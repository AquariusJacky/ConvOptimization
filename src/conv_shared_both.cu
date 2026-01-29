#include <stdio.h>

#include <algorithm>

#include "conv_interface.h"

namespace shared_both {

// Kernel parameters optimized for your dimensions

#define TILE_SIZE 16    // thread tile size
#define C_PER_BLOCK 32  // Input channels per iteration
#define K_PER_BLOCK 4   // Output channels per block

// Forward convolution kernel with shared memory tiling and register blocking
__global__ void conv_shared_both_forward(const float* input,   // [N, C, H, W]
                                         const float* kernel,  // [K, C, R, S]
                                         float* output,  // [N, K, H_out, W_out]
                                         size_t N, size_t C, size_t H, size_t W,
                                         size_t K, size_t R, size_t S,
                                         size_t pad, size_t stride,
                                         size_t H_out, size_t W_out) {
  // Block indices
  size_t bx = blockIdx.x;
  size_t by = blockIdx.y;
  size_t bz = blockIdx.z;
  size_t n = bz / ((K + K_PER_BLOCK - 1) / K_PER_BLOCK);
  size_t k_base = (bz % ((K + K_PER_BLOCK - 1) / K_PER_BLOCK)) * K_PER_BLOCK;

  size_t tx = threadIdx.x;
  size_t ty = threadIdx.y;
  size_t tz = threadIdx.z;

  // Output element indices
  size_t out_row = by * TILE_SIZE + ty;
  size_t out_col = bx * TILE_SIZE + tx;
  size_t k = k_base + tz;

  // Dynamic shared memory allocation
  extern __shared__ float smem[];

  // Calculate input tile size (output tile + halo)
  size_t input_h = (TILE_SIZE - 1) * stride + R;
  size_t input_w = (TILE_SIZE - 1) * stride + S;

  // Partition shared memory
  // Input: [C_PER_BLOCK][TILE_SIZE][TILE_SIZE]
  // Kernel: [K_PER_BLOCK][C_PER_BLOCK][R][S]
  float* smem_input = smem;
  float* smem_kernel = smem + C_PER_BLOCK * input_h * input_w;

  float sum = 0.0f;

  // Loop over all input channels in chunks of C_PER_BLOCK
  // There are a total of K_PER_BLOCK * C kernels to load
  for (size_t c_base = 0; c_base < C; c_base += C_PER_BLOCK) {
    // Load kernel
    // We load K_PER_BLOCK * C_PER_BLOCK kernels each iteration
    size_t tid = tz * (TILE_SIZE * TILE_SIZE) + ty * TILE_SIZE + tx;
    size_t total_threads = K_PER_BLOCK * TILE_SIZE * TILE_SIZE;
    size_t elem_per_k = C_PER_BLOCK * R * S;  // Elements per output channel

    // Since the block shape is not aligned with the shape of the kernel, so
    // each thread gets assigned according to the total thread count
    for (size_t w_idx = tid; w_idx < K_PER_BLOCK * elem_per_k;
         w_idx += total_threads) {
      size_t ker_k_off = w_idx / elem_per_k;  // output channel offset
      size_t ker_k_pos =
          w_idx % elem_per_k;  // Position within the output kernel
      size_t ker_c_off = ker_k_pos / (R * S);  // input channel offset
      size_t ker_c_pos =
          ker_k_pos % (R * S);   // Position within the output kernel
      size_t r = ker_c_pos / S;  // row number
      size_t s = ker_c_pos % S;  // col number

      float val = 0.0f;
      if (k_base + ker_k_off < K && c_base + ker_c_off < C) {
        val = kernel[(k_base + ker_k_off) * C * R * S +
                     (c_base + ker_c_off) * R * S + (r)*S + (s)];
      }
      smem_kernel[(ker_k_off)*C_PER_BLOCK * R * S + (ker_c_off)*R * S + (r)*S +
                  (s)] = val;
    }

    // Synchronization is not needed here since loading kernel and input use
    // different shared memory regions

    // Load input tile
    size_t in_row_base = by * TILE_SIZE * stride - pad;
    size_t in_col_base = bx * TILE_SIZE * stride - pad;

    // Block shape is aligned with input tile shape, so it's easier to index
    // this than the kernel
    for (size_t in_c_off = tz; in_c_off < C_PER_BLOCK;
         in_c_off += K_PER_BLOCK) {
      size_t c_idx = c_base + in_c_off;
      if (c_idx >= C) continue;

      for (size_t i = ty; i < input_h; i += TILE_SIZE) {
        for (size_t j = tx; j < input_w; j += TILE_SIZE) {
          int in_row = (int)in_row_base + (int)i;
          int in_col = (int)in_col_base + (int)j;

          float val = 0.0f;
          if (in_row >= 0 && in_row < H && in_col >= 0 && in_col < W) {
            val = input[(n)*C * H * W + (c_idx)*H * W + (in_row)*W + (in_col)];
          }
          smem_input[(in_c_off)*input_h * input_w + (i)*input_w + (j)] = val;
        }
      }
    }

    // Synchronize to make sure all elements are loaded
    __syncthreads();

    // Compute convolution
    if (k < K && ty < TILE_SIZE && tx < TILE_SIZE) {
      for (size_t c_off = 0; c_off < C_PER_BLOCK; c_off++) {
        if (c_base + c_off >= C) break;

        for (size_t r = 0; r < R; r++) {
          for (size_t s = 0; s < S; s++) {
            // Access smem_kernel[tz][c_off][r][s]
            float w = smem_kernel[(tz)*C_PER_BLOCK * R * S + (c_off)*R * S +
                                  (r)*S + (s)];
            // Access smem_input[c_off][ty*stride + r][tx*stride + s]
            size_t in_i = ty * stride + r;
            size_t in_j = tx * stride + s;
            float inp =
                smem_input[(c_off)*input_h * input_w + (in_i)*input_w + (in_j)];
            sum += inp * w;
          }
        }
      }
    }

    __syncthreads();
  }

  // Write result
  if (out_row < H_out && out_col < W_out && n < N && k < K) {
    output[(n)*K * H_out * W_out + (k)*H_out * W_out + (out_row)*W_out +
           (out_col)] = sum;
  }
}

/*
  @brief Convolution forward pass using a bigger shared memory that includes
  both input and kernel with a 3D block shape (shared memory can't fit all K
  kernels). Since the shared version is worse than naive, this is an attempt to
  see if moving kernel to shared memory helps.
  From here it's much harder to do indexing
  @param input Pointer to input tensor [N, C, H, W]
  @param kernel Pointer to kernel tensor [K, C, R, S]
  @param output Pointer to output tensor [N, K, out_h, out_w]
  @param params Convolution parameters
*/
void conv_forward(const float* input, const float* kernel, float* output,
                  const ConvParams& params) {
  size_t H_out = (params.H + 2 * params.pad - params.R) / params.stride + 1;
  size_t W_out = (params.W + 2 * params.pad - params.S) / params.stride + 1;

  // Calculate shared memory size
  size_t input_h = TILE_SIZE + (params.R - 1);
  size_t input_w = TILE_SIZE + (params.S - 1);
  // Shared memory can only fit C_PER_BLOCK channels at a time
  size_t smem_input_size = C_PER_BLOCK * input_h * input_w * sizeof(float);
  size_t smem_kernel_size =
      K_PER_BLOCK * C_PER_BLOCK * params.R * params.S * sizeof(float);
  size_t total_smem = smem_input_size + smem_kernel_size;

  // Grid and block dimensions
  // Each block computes only K_PER_BLOCK output channels but still all C
  // channels
  dim3 block(TILE_SIZE, TILE_SIZE, K_PER_BLOCK);
  dim3 grid((W_out + TILE_SIZE - 1) / TILE_SIZE,
            (H_out + TILE_SIZE - 1) / TILE_SIZE,
            params.N * ((params.K + K_PER_BLOCK - 1) / K_PER_BLOCK));

  conv_shared_both_forward<<<grid, block, total_smem>>>(
      input, kernel, output, params.N, params.C, params.H, params.W, params.K,
      params.R, params.S, params.pad, params.stride, H_out, W_out);
}
}  // namespace shared_both