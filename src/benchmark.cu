#include <string>

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
namespace shared_both {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}
namespace what {
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

  printf("Convolution: N=%zd, C=%zd, H=%zd×%zd, K=%zd, Kernel=%zdx%zd\n",
         params.N, params.C, params.H, params.W, params.K, params.R, params.S);
  printf("Output: %zd×%zd, Total FLOPs: %.2f GFLOP\n\n", params.out_h(),
         params.out_w(),
         calculate_gflops(params, 1000.0f));  // GFLOPS at 1 second

  // Benchmark CPU
  auto cpu_result = time_cpu_convolution(cpu::conv_forward, buffers);
  float* cpu_output = new float[params.output_size()];
  memcpy(cpu_output, buffers.h_output, params.output_size() * sizeof(float));

  // Unoptimized GPU implementations
  auto naive_result = time_gpu_convolution(naive::conv_forward, buffers);
  float* naive_output = new float[params.output_size()];
  memcpy(naive_output, buffers.h_output, params.output_size() * sizeof(float));

  printf("Comparing naive output to CPU reference...\n");
  compare_outputs(cpu_output, naive_output, params.output_size());

  // Shared GPU implementations
  auto shared_result =
      time_gpu_convolution(shared::conv_forward, buffers, true);
  float* shared_output = new float[params.output_size()];
  memcpy(shared_output, buffers.h_output, params.output_size() * sizeof(float));

  printf("Comparing shared output to CPU reference...\n");
  compare_outputs(cpu_output, shared_output, params.output_size());

  // Shared_both GPU implementations
  auto shared_both_result =
      time_gpu_convolution(shared_both::conv_forward, buffers, true);
  float* shared_both_output = new float[params.output_size()];
  memcpy(shared_both_output, buffers.h_output,
         params.output_size() * sizeof(float));

  printf("Comparing shared_both output to CPU reference...\n");
  compare_outputs(cpu_output, shared_both_output, params.output_size());

  // cuDNN implementations
  auto cudnn_result = time_cudnn_convolution(buffers);
  float* cudnn_output = new float[params.output_size()];
  memcpy(cudnn_output, buffers.h_output, params.output_size() * sizeof(float));

  printf("Comparing cuDNN output to CPU reference...\n");
  compare_outputs(cpu_output, cudnn_output, params.output_size());

  //////////////////////////////////////////////////////////////////////////////////////////

  printf("\nPerformance:\n");

  // Print results
  cpu_result.print("Baseline");
  naive_result.print("Naive", cpu_result.total_time);
  shared_result.print("Shared", cpu_result.total_time);
  shared_both_result.print("Shared_both", cpu_result.total_time);
  cudnn_result.print("cuDNN", cpu_result.total_time);

  // Print GFLOPS
  printf("  CPU:    %.2f GFLOPS\n",
         calculate_gflops(params, cpu_result.total_time));
  printf("  Naive:  %.2f GFLOPS\n",
         calculate_gflops(params, naive_result.kernel_time));
  printf("  Shared: %.2f GFLOPS\n",
         calculate_gflops(params, shared_result.kernel_time));
  printf("  Shared_both: %.2f GFLOPS\n",
         calculate_gflops(params, shared_both_result.kernel_time));
  printf("  cuDNN: %.2f GFLOPS\n",
         calculate_gflops(params, cudnn_result.kernel_time));

  delete[] cpu_output;
  delete[] naive_output;
  delete[] shared_output;
  delete[] shared_both_output;
  delete[] cudnn_output;

  return 0;
}