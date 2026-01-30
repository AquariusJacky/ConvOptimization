#include <mma.h>
#include <stdio.h>

#include <algorithm>

#include "conv_interface.h"

// CUDA error checking macro
#define CUDA_CHECK(call)                                               \
  do {                                                                 \
    cudaError_t err = call;                                            \
    if (err != cudaSuccess) {                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err));                                \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

namespace im2col_NHWC {

#define NCHW_TO_NHWC_BLOCK_SIZE 256

// This kernel must be launched for both input and kernel matrices
__global__ void NCHW_to_NHWC(const float* input, half* output, size_t N,
                             size_t C, size_t H, size_t W) {
  size_t total_elements = N * C * H * W;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx; i < total_elements; i += stride) {
    // Compute NHWC layout indices
    size_t n = i / (H * W * C);
    size_t h = (i / (W * C)) % H;
    size_t w = (i / C) % W;
    size_t c = i % C;

    output[i] = __float2half(input[(n)*C * H * W + (c)*H * W + (h)*W + (w)]);
  }
}

#define BLOCK_SIZE 128

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define TILE_SIZE 64       // thread tile size
#define WARP_TILE_SIZE 32  // number of output elements per warp

__global__ void conv_im2col_NHWC_forward(const half* input,   // [N, C, H, W]
                                         const half* kernel,  // [K, C, R, S]
                                         float* output,  // [N, K, H_out, W_out]
                                         size_t N, size_t C, size_t H, size_t W,
                                         size_t K, size_t R, size_t S,
                                         size_t pad, size_t stride,
                                         size_t H_out, size_t W_out) {
  __shared__ half tile_kernel[TILE_SIZE][WMMA_K];
  __shared__ half tile_im2col[WMMA_K][TILE_SIZE];
  __shared__ half tile_output[TILE_SIZE][TILE_SIZE];

  int warpId = threadIdx.x / 32;
  int warp_row = (warpId / 2) * WARP_TILE_SIZE;
  int warp_col = (warpId % 2) * WARP_TILE_SIZE;

  int spatial_block_idx = blockIdx.x * TILE_SIZE;
  int channel_block_idx = blockIdx.y * TILE_SIZE;

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::row_major>
      a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                         nvcuda::wmma::row_major>
      b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K,
                         half>
      c_frag[2][2];

#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      nvcuda::wmma::fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  int CRS = C * R * S;

  // ===== Main computation loop =====
  for (int k_block = 0; k_block < CRS; k_block += WMMA_K) {
// Load kernel tile
#pragma unroll
    for (int idx = threadIdx.x; idx < (TILE_SIZE * WMMA_K); idx += BLOCK_SIZE) {
      int tile_row = idx / WMMA_K;
      int tile_col = idx % WMMA_K;

      int k_out = channel_block_idx + tile_row;
      int crs_idx = k_block + tile_col;

      if (k_out < K && crs_idx < CRS) {
        // Decompose crs_idx into c, r, s
        int c = crs_idx / (R * S);
        int rs = crs_idx % (R * S);
        int r = rs / S;
        int s = rs % S;

        // KRSC: recompose as k, r, s, c
        int krsc_idx = k_out * (R * S * C) + r * (S * C) + s * C + c;

        tile_kernel[tile_row][tile_col] = kernel[krsc_idx];
      } else {
        tile_kernel[tile_row][tile_col] = __float2half(0.0f);
      }
    }

// Load im2col tile (row-major)
#pragma unroll
    for (int idx = threadIdx.x; idx < (WMMA_K * TILE_SIZE); idx += BLOCK_SIZE) {
      int tile_row = idx / TILE_SIZE;
      int tile_col = idx % TILE_SIZE;

      int crs_idx = k_block + tile_row;
      int spatial_idx = spatial_block_idx + tile_col;

      half value = __float2half(0.0f);

      if (crs_idx < CRS && spatial_idx < N * H_out * W_out) {
        int c = crs_idx / (R * S);
        int rs = crs_idx % (R * S);
        int r = rs / S;
        int s = rs % S;

        int n = spatial_idx / (H_out * W_out);
        int hw = spatial_idx % (H_out * W_out);
        int out_h = hw / W_out;
        int out_w = hw % W_out;

        int in_h = out_h * stride - pad + r;
        int in_w = out_w * stride - pad + s;

        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
          value = input[(n)*H * W * C + (in_h)*W * C + (in_w)*C + (c)];
        }
      }

      tile_im2col[tile_row][tile_col] = value;
    }

    __syncthreads();

// WMMA computation
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
      for (int j = 0; j < 2; j++) {
        int a_row = warp_row + i * WMMA_M;
        int b_col = warp_col + j * WMMA_N;

        nvcuda::wmma::load_matrix_sync(a_frag, &tile_kernel[a_row][0], WMMA_K);
        nvcuda::wmma::load_matrix_sync(b_frag, &tile_im2col[0][b_col],
                                       TILE_SIZE);

        nvcuda::wmma::mma_sync(c_frag[i][j], a_frag, b_frag, c_frag[i][j]);
      }
    }

    __syncthreads();
  }

// ===== Store to shared memory first =====
#pragma unroll
  for (int i = 0; i < 2; i++) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
      int smem_row = warp_row + i * WMMA_M;
      int smem_col = warp_col + j * WMMA_N;

      // Store fragment to shared memory
      nvcuda::wmma::store_matrix_sync(&tile_output[smem_row][smem_col],
                                      c_frag[i][j], TILE_SIZE,
                                      nvcuda::wmma::mem_row_major);
    }
  }

  __syncthreads();

// ===== Copy from shared memory to global memory =====
#pragma unroll
  for (int idx = threadIdx.x; idx < (TILE_SIZE * TILE_SIZE);
       idx += BLOCK_SIZE) {
    int tile_row = idx / TILE_SIZE;
    int tile_col = idx % TILE_SIZE;

    int out_k = channel_block_idx + tile_row;
    int spatial_idx = spatial_block_idx + tile_col;

    if (out_k < K && spatial_idx < N * H_out * W_out) {
      // Decompose spatial index
      int n = spatial_idx / (H_out * W_out);
      int hw = spatial_idx % (H_out * W_out);
      int out_h = hw / W_out;
      int out_w = hw % W_out;

      // Write to global memory: output[N, K, H_out, W_out]
      int out_idx = ((n * K + out_k) * H_out + out_h) * W_out + out_w;
      output[out_idx] = __half2float(tile_output[tile_row][tile_col]);
    }
  }
}

void conv_forward(const float* input, const float* kernel, float* output,
                  const ConvParams& params) {
  size_t H_out = (params.H + 2 * params.pad - params.R) / params.stride + 1;
  size_t W_out = (params.W + 2 * params.pad - params.S) / params.stride + 1;

  dim3 block(NCHW_TO_NHWC_BLOCK_SIZE);
  dim3 grid((params.input_size() + NCHW_TO_NHWC_BLOCK_SIZE - 1) /
            NCHW_TO_NHWC_BLOCK_SIZE);

  half *d_input_nhwc, *d_kernel_nhwc;
  CUDA_CHECK(cudaMalloc(&d_input_nhwc, params.input_size() * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_kernel_nhwc, params.kernel_size() * sizeof(half)));

  NCHW_to_NHWC<<<grid, block>>>(input, d_input_nhwc, params.N, params.C,
                                params.H, params.W);

  grid = dim3((params.kernel_size() + NCHW_TO_NHWC_BLOCK_SIZE - 1) /
              NCHW_TO_NHWC_BLOCK_SIZE);
  NCHW_to_NHWC<<<grid, block>>>(kernel, d_kernel_nhwc, params.K, params.C,
                                params.R, params.S);

  block = dim3(BLOCK_SIZE);
  grid = dim3(((params.N * H_out * W_out) + TILE_SIZE - 1) / TILE_SIZE),
  (params.K + TILE_SIZE - 1) / TILE_SIZE;
  conv_im2col_NHWC_forward<<<grid, block>>>(
      d_input_nhwc, d_kernel_nhwc, output, params.N, params.C, params.H,
      params.W, params.K, params.R, params.S, params.pad, params.stride, H_out,
      W_out);

  CUDA_CHECK(cudaFree(d_input_nhwc));
  CUDA_CHECK(cudaFree(d_kernel_nhwc));
}
}  // namespace im2col_NHWC