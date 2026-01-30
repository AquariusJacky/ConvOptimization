#include <string>

#include "benchmark_utils.h"
#include "conv_interface.h"

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
namespace im2col {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}
namespace im2col_NHWC {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}
namespace im2col_NHWC_restrict {
void conv_forward(const float*, const float*, float*, const ConvParams&);
}

int main(int argc, char** argv) {
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

  // Report mode or comparison mode
  bool report_mode = false;
  bool naive_gpu = true;
  bool shared_gpu = true;
  bool shared_both_gpu = true;
  bool im2col_gpu = true;
  bool im2col_NHWC_gpu = true;
  bool cudnn = true;

  int warmup_iters, timing_iters;
  if (report_mode) {
    warmup_iters = 2;
    timing_iters = 5;
  } else {
    warmup_iters = 5;
    timing_iters = 20;
  }

  // Allocate all buffers once (via RAII)
  ConvBuffers buffers(params);

  printf("Convolution: N=%zd, C=%zd, H=%zd×%zd, K=%zd, Kernel=%zdx%zd\n",
         params.N, params.C, params.H, params.W, params.K, params.R, params.S);
  printf("Output: %zd×%zd, Total FLOPs: %.2f GFLOP\n\n", params.out_h(),
         params.out_w(),
         calculate_gflops(params, 1000.0f));  // GFLOPS at 1 second

  TimingResult cpu_result, naive_result, shared_result, shared_both_result,
      im2col_result, im2col_NHWC_result, im2col_NHWC_restrict_result,
      cudnn_result;
  float *cpu_output, *naive_output, *shared_output, *shared_both_output,
      *im2col_output, *im2col_NHWC_output, *im2col_NHWC_restrict_output,
      *cudnn_output;

  // Benchmark CPU
  cpu_result = time_cpu_convolution(cpu::conv_forward, buffers);
  cpu_output = new float[params.output_size()];
  memcpy(cpu_output, buffers.h_output, params.output_size() * sizeof(float));

  if (naive_gpu) {  // Unoptimized GPU implementations
    naive_result = time_gpu_convolution(naive::conv_forward, buffers,
                                        warmup_iters, timing_iters);
    float* naive_output = new float[params.output_size()];
    memcpy(naive_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing naive output to CPU reference...\n");
    compare_outputs(cpu_output, naive_output, params.output_size());
  }

  if (shared_gpu) {
    // Shared GPU implementations
    shared_result = time_gpu_convolution(shared::conv_forward, buffers,
                                         warmup_iters, timing_iters);
    float* shared_output = new float[params.output_size()];
    memcpy(shared_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing shared output to CPU reference...\n");
    compare_outputs(cpu_output, shared_output, params.output_size());
  }

  if (shared_both_gpu) {
    // Shared_both GPU implementations
    shared_both_result = time_gpu_convolution(
        shared_both::conv_forward, buffers, warmup_iters, timing_iters);
    float* shared_both_output = new float[params.output_size()];
    memcpy(shared_both_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing shared_both output to CPU reference...\n");
    compare_outputs(cpu_output, shared_both_output, params.output_size());
  }

  if (im2col_gpu) {
    // im2col GPU implementations
    im2col_result = time_gpu_convolution(im2col::conv_forward, buffers,
                                         warmup_iters, timing_iters);
    float* im2col_output = new float[params.output_size()];
    memcpy(im2col_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing im2col output to CPU reference...\n");
    compare_outputs(cpu_output, im2col_output, params.output_size());
  }

  if (im2col_NHWC_gpu) {
    // im2col_NHWC GPU implementations
    im2col_NHWC_result = time_gpu_convolution(
        im2col_NHWC::conv_forward, buffers, warmup_iters, timing_iters);
    float* im2col_NHWC_output = new float[params.output_size()];
    memcpy(im2col_NHWC_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing im2col_NHWC output to CPU reference...\n");
    compare_outputs(cpu_output, im2col_NHWC_output, params.output_size());
  }

  if (cudnn) {
    // cuDNN implementations
    cudnn_result = time_cudnn_convolution(buffers);
    float* cudnn_output = new float[params.output_size()];
    memcpy(cudnn_output, buffers.h_output,
           params.output_size() * sizeof(float));

    printf("Comparing cuDNN output to CPU reference...\n");
    compare_outputs(cpu_output, cudnn_output, params.output_size());
  }

  //////////////////////////////////////////////////////////////////////////////////////////

  printf("\nPerformance:\n");

  // Print results
  cpu_result.print("Baseline");
  if (naive_gpu) naive_result.print("Naive", cpu_result.total_time);
  if (shared_gpu) shared_result.print("Shared", cpu_result.total_time);
  if (shared_both_gpu)
    shared_both_result.print("Shared_both", cpu_result.total_time);
  if (im2col_gpu) im2col_result.print("im2col", cpu_result.total_time);
  if (im2col_NHWC_gpu)
    im2col_NHWC_result.print("im2col_NHWC", cpu_result.total_time);
  if (cudnn) cudnn_result.print("cuDNN", cpu_result.total_time);

  // Print GFLOPS

  printf("  CPU:    %.2f GFLOPS\n",
         calculate_gflops(params, cpu_result.total_time));
  if (naive_gpu)
    printf("  Naive:  %.2f GFLOPS\n",
           calculate_gflops(params, naive_result.kernel_time));
  if (shared_gpu)
    printf("  Shared: %.2f GFLOPS\n",
           calculate_gflops(params, shared_result.kernel_time));
  if (shared_both_gpu)
    printf("  Shared_both: %.2f GFLOPS\n",
           calculate_gflops(params, shared_both_result.kernel_time));
  if (im2col_gpu)
    printf("  im2col: %.2f GFLOPS\n",
           calculate_gflops(params, im2col_result.kernel_time));
  if (im2col_NHWC_gpu)
    printf("  im2col_NHWC: %.2f GFLOPS\n",
           calculate_gflops(params, im2col_NHWC_result.kernel_time));
  if (cudnn)
    printf("  cuDNN: %.2f GFLOPS\n",
           calculate_gflops(params, cudnn_result.kernel_time));

  delete[] cpu_output;
  if (naive_gpu) delete[] naive_output;
  if (shared_gpu) delete[] shared_output;
  if (shared_both_gpu) delete[] shared_both_output;
  if (im2col_gpu) delete[] im2col_output;
  if (im2col_NHWC_gpu) delete[] im2col_NHWC_output;
  if (cudnn) delete[] cudnn_output;

  return 0;
}