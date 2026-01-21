#include <algorithm>

#include "conv_interface.h"

namespace shared {

__global__ void conv_forward_kernel(const float* input, const float* kernel,
                                    float* output, ConvParams params) {
  // Kernel implementation (not used in naive CPU version)
}

void conv_forward(const float* input,   // [N, C, H, W]
                  const float* kernel,  // [K, C, R, S]
                  float* output,        // [N, K, out_h, out_w]
                  const ConvParams& params) {}

}  // namespace shared