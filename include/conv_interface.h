#ifndef CONV_INTERFACE_H
#define CONV_INTERFACE_H

#include <cstddef>

// Parameters for 2D convolution
struct ConvParams {
  // Input dimensions
  size_t N;  // Batch size
  size_t C;  // Input channels
  size_t H;  // Input height
  size_t W;  // Input width

  // Output dimensions
  size_t K;  // Output channels (number of filters)

  // Kernel dimensions
  size_t R;  // Kernel height
  size_t S;  // Kernel width

  // Convolution parameters
  size_t pad;     // Padding
  size_t stride;  // Stride

  // Computed output dimensions
  // Because (out_h - 1) * stride + R = (H + (2 * pad)) = original size possibly
  // with padding
  size_t out_h() const { return (H + 2 * pad - R) / stride + 1; }
  size_t out_w() const { return (W + 2 * pad - S) / stride + 1; }

  // Size calculations
  size_t input_size() const { return N * C * H * W; }
  size_t kernel_size() const { return K * C * R * S; }
  size_t output_size() const { return N * K * out_h() * out_w(); }
};

// Standard interface that ALL implementations must follow
// This makes benchmarking fair and consistent
void conv_forward(const float* input,   // Input tensor [N, C, H, W]
                  const float* kernel,  // Kernel weights [K, C, R, S]
                  float* output,        // Output tensor [N, K, out_h, out_w]
                  const ConvParams& params);

#endif