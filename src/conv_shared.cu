#include <stdio.h>

#include <algorithm>

#include "conv_interface.h"

namespace shared {

#define TILE_SIZE 16
#define C_PER_BLOCK 8

// To solve coalescing issues, adjacent threads will load adjacent elements
__global__ void conv_shared_kernel(const float* input, const float* kernel,
                                   float* output, size_t N, size_t C, size_t H,
                                   size_t W, size_t K, size_t R, size_t S,
                                   size_t pad, size_t stride) {
  size_t bx = blockDim.x * blockIdx.x;
  size_t by = blockDim.y * blockIdx.y;
  size_t bz = blockIdx.z;
  size_t n = bz / K;
  size_t k = bz % K;

  size_t tx = bx + threadIdx.x;
  size_t ty = by + threadIdx.y;

  size_t out_w = (W + 2 * pad - S) / stride + 1;
  size_t out_h = (H + 2 * pad - R) / stride + 1;

  extern __shared__ float smem[];

  size_t input_h = (blockDim.y - 1) * stride + R;
  size_t input_w = (blockDim.x - 1) * stride + S;

  float sum = 0.0f;

  // Loop over input channels in chunks of C_PER_BLOCK
  for (size_t c_start = 0; c_start < C; c_start += C_PER_BLOCK) {
    size_t c_end = min(c_start + C_PER_BLOCK, C);
    size_t num_c = c_end - c_start;

    // Load channel tile into shared memory
    for (size_t c_idx = 0; c_idx < num_c; c_idx++) {
      size_t c = c_start + c_idx;
      for (size_t i = threadIdx.y; i < input_h; i += blockDim.y) {
        for (size_t j = threadIdx.x; j < input_w; j += blockDim.x) {
          int h = (int)by * stride + (int)i - (int)pad;
          int w = (int)bx * stride + (int)j - (int)pad;

          if (w >= 0 && w < W && h >= 0 && h < H) {
            smem[(c_idx)*input_h * input_w + (i)*input_w + (j)] =
                input[(n)*C * H * W + (c)*H * W + (h)*W + (w)];
          } else {
            smem[(c_idx)*input_h * input_w + (i)*input_w + (j)] = 0.0f;
          }
        }
      }
    }
    __syncthreads();

    // Compute with this channel tile
    for (size_t c_idx = 0; c_idx < num_c; c_idx++) {
      size_t c = c_start + c_idx;
      for (size_t r = 0; r < R; r++) {
        for (size_t s = 0; s < S; s++) {
          int shared_h = threadIdx.y * stride + r;
          int shared_w = threadIdx.x * stride + s;
          sum += smem[(c_idx)*input_h * input_w + (shared_h)*input_w +
                      (shared_w)] *
                 kernel[(k)*C * R * S + (c)*R * S + (r)*S + (s)];
        }
      }
    }
    __syncthreads();
  }

  if (ty < out_h && tx < out_w) {
    output[(n)*K * out_h * out_w + (k)*out_h * out_w + (ty)*out_w + (tx)] = sum;
  }
}

/*
  @brief Convolution forward pass using shared memory optimization
  for input tiles. Uncoalesced global memory access may be the biggest
  optimization according to ncu. Also, since we can't fit all input channels in
  shared memory, we have to split the channels internally during calculation.
  @param input Pointer to input tensor [N, C, H, W]
  @param kernel Pointer to kernel tensor [K, C, R, S]
  @param output Pointer to output tensor [N, K, out_h, out_w]
  @param params Convolution parameters
*/
void conv_forward(const float* input, const float* kernel, float* output,
                  const ConvParams& params) {
  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((params.out_w() + block.x - 1) / block.x,
            (params.out_h() + block.y - 1) / block.y, params.N * params.K);

  size_t input_h = (block.y - 1) * params.stride + params.R;
  size_t input_w = (block.x - 1) * params.stride + params.S;
  size_t shared_mem_size = C_PER_BLOCK * input_h * input_w * sizeof(float);

  conv_shared_kernel<<<grid, block, shared_mem_size>>>(
      input, kernel, output, params.N, params.C, params.H, params.W, params.K,
      params.R, params.S, params.pad, params.stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Shared kernel error: %s", cudaGetErrorString(err));
  }
}

}  // namespace shared