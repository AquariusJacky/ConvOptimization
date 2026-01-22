#include <algorithm>

#include "conv_interface.h"

namespace cpu {

#define input(n, c, h, w)                                                  \
  input[(n) * params.C * params.H * params.W + (c) * params.H * params.W + \
        (h) * params.W + (w)]

#define kernel(k, c, r, s)                                                  \
  kernel[(k) * params.C * params.R * params.S + (c) * params.R * params.S + \
         (r) * params.S + (s)]

#define output(n, k, h, w)                                  \
  output[(n) * params.K * params.out_h() * params.out_w() + \
         (k) * params.out_h() * params.out_w() + (h) * params.out_w() + (w)]

void conv_forward(const float* input,   // [N, C, H, W]
                  const float* kernel,  // [K, C, R, S]
                  float* output,        // [N, K, out_h, out_w]
                  const ConvParams& params) {
  size_t out_h = params.out_h();
  size_t out_w = params.out_w();

  for (size_t n = 0; n < params.N; n++) {
    for (size_t k = 0; k < params.K; k++) {
      for (size_t p = 0; p < out_h; p++) {
        for (size_t q = 0; q < out_w; q++) {
          float sum = 0.0f;
          for (size_t c = 0; c < params.C; c++) {
            for (size_t r = 0; r < params.R; r++) {
              for (size_t s = 0; s < params.S; s++) {
                int in_h = (int)(p * params.stride) - (int)params.pad + (int)r;
                int in_w = (int)(q * params.stride) - (int)params.pad + (int)s;
                if (in_h >= 0 && in_h < (int)params.H && in_w >= 0 &&
                    in_w < (int)params.W) {
                  sum += input(n, c, in_h, in_w) * kernel(k, c, r, s);
                }
              }
            }
          }
          output(n, k, p, q) = sum;
        }
      }
    }
  }
}

}  // namespace cpu