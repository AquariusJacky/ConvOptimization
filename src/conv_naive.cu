#include <algorithm>
#include <iostream>

#include "conv_interface.h"

namespace naive {

__global__ void conv_naive_kernel(const float* input, const float* kernel,
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

  if (tx >= out_w || ty >= out_h) {
    return;
  }

  float sum = 0.0f;
  for (size_t c = 0; c < C; c++) {
    for (size_t r = 0; r < R; r++) {
      for (size_t s = 0; s < S; s++) {
        int h = (int)(ty * stride) - (int)pad + (int)r;
        int w = (int)(tx * stride) - (int)pad + (int)s;
        if (w >= 0 && w < W && h >= 0 && h < H) {
          sum += input[(n)*C * H * W + (c)*H * W + (h)*W + (w)] *
                 kernel[(k)*C * R * S + (c)*R * S + (r)*S + (s)];
        }
      }
    }
  }

  output[(n)*K * out_h * out_w + (k)*out_h * out_w + (ty)*out_w + (tx)] = sum;
}

void conv_forward(const float* input,   // [N, C, H, W]
                  const float* kernel,  // [K, C, R, S]
                  float* output,        // [N, K, out_h, out_w]
                  const ConvParams& params) {
  dim3 block(16, 16);
  dim3 grid((params.out_w() + block.x - 1) / block.x,
            (params.out_h() + block.y - 1) / block.y, params.N * params.K);

  conv_naive_kernel<<<grid, block>>>(input, kernel, output, params.N, params.C,
                                     params.H, params.W, params.K, params.R,
                                     params.S, params.pad, params.stride);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Naive kernel error: %s", cudaGetErrorString(err));
  }
}

}  // namespace naive