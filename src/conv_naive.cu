#include <algorithm>
#include <iostream>

#include "conv_interface.h"

namespace naive {

__global__ void conv_naive_kernel(const float* input, const float* kernel,
                                  float* output, ConvParams params) {
  size_t bx = blockDim.x * blockIdx.x;
  size_t by = blockDim.y * blockIdx.y;
  size_t bz = blockIdx.z;
  size_t tx = bx + threadIdx.x;
  size_t ty = by + threadIdx.y;
}

void conv_forward(const float* input,   // [N, C, H, W]
                  const float* kernel,  // [K, C, R, S]
                  float* output,        // [N, K, out_h, out_w]
                  const ConvParams& params) {
  int out_h = params.out_h();
  int out_w = params.out_w();

  dim3 block(16, 16);
  dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
            params.N * params.K);

  conv_naive_kernel<<<grid, block>>>(input, kernel, output, params);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Naive kernel error: " << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace naive