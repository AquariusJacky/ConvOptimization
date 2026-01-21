#include "benchmark_utils.h"
#include "conv_interface.h"

// Declare implementations (each in separate .cu file)
namespace cpu {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}
namespace naive {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}
namespace shared {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}

int main() {
  // Define problem size
  ConvParams params;
  params.N = 32;
  params.C = 128;
  params.H = 28;
  params.W = 28;
  params.K = 256;
  params.R = 3;
  params.S = 3;
  params.pad = 1;
  params.stride = 1;

  // Allocate all buffers once (via RAII)
  ConvBuffers buffers(params);

  printf("Convolution: N=%d, C=%d, H=%d×%d, K=%d, Kernel=%dx%d\n", params.N,
         params.C, params.H, params.W, params.K, params.R, params.S);
  printf("Output: %d×%d, Total FLOPs: %.2f GFLOP\n\n", params.out_h(),
         params.out_w(),
         calculate_gflops(params, 1000.0f));  // GFLOPS at 1 second

  // Benchmark CPU
  auto cpu_result = time_cpu_convolution(cpu::conv_forward, buffers);
  cpu_result.print("CPU Baseline");

  // Benchmark GPU implementations
  auto naive_result = time_gpu_convolution(naive::conv_forward, buffers);
  naive_result.print("CUDA Naive", cpu_result.total_time);

  auto shared_result = time_gpu_convolution(shared::conv_forward, buffers);
  shared_result.print("CUDA Shared Memory", cpu_result.total_time);

  printf("\nPerformance:\n");
  printf("  CPU:    %.2f GFLOPS\n",
         calculate_gflops(params, cpu_result.total_time));
  printf("  Naive:  %.2f GFLOPS\n",
         calculate_gflops(params, naive_result.kernel_time));
  printf("  Shared: %.2f GFLOPS\n",
         calculate_gflops(params, shared_result.kernel_time));

  return 0;
}