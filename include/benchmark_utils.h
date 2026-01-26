#ifndef BENCHMARK_UTILS_H
#define BENCHMARK_UTILS_H

#include <cuda_runtime.h>
#include <cudnn.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>

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

#define CUDNN_CHECK(call)                                               \
  do {                                                                  \
    cudnnStatus_t status = call;                                        \
    if (status != CUDNN_STATUS_SUCCESS) {                               \
      fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudnnGetErrorString(status));                             \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

// Data structure to hold all buffers
struct ConvBuffers {
  // Host buffers
  float* h_input;
  float* h_kernel;
  float* h_output;

  // Device buffers
  float* d_input;
  float* d_kernel;
  float* d_output;

  ConvParams params;

  // Constructor: allocate all buffers
  ConvBuffers(const ConvParams& p) : params(p) {
    size_t input_bytes = params.input_size() * sizeof(float);
    size_t kernel_bytes = params.kernel_size() * sizeof(float);
    size_t output_bytes = params.output_size() * sizeof(float);

    // Allocate host memory
    h_input = new float[params.input_size()];
    h_kernel = new float[params.kernel_size()];
    h_output = new float[params.output_size()];

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // Initialize with random data
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < params.input_size(); i++) {
      h_input[i] = dis(gen);
    }
    for (size_t i = 0; i < params.kernel_size(); i++) {
      h_kernel[i] = dis(gen);
    }
  }

  // Destructor: free all buffers
  ~ConvBuffers() {
    delete[] h_input;
    delete[] h_kernel;
    delete[] h_output;

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
  }

  // Prevent copying (expensive)
  ConvBuffers(const ConvBuffers&) = delete;
  ConvBuffers& operator=(const ConvBuffers&) = delete;
};

void compare_outputs(const float* output1, const float* output2, size_t size,
                     float tol = 1e-3f) {
  float max_diff = 0.0f;
  float rel_error = 0.0f;
  for (int i = 0; i < size; i++) {
    float diff = std::abs(output1[i] - output2[i]);
    max_diff = std::max(max_diff, diff);

    // If value is very small, avoid division by zero
    rel_error =
        std::max(rel_error, diff / std::max(std::abs(output2[i]), 1e-5f));
  }

  printf("  Max absolute difference: %.2e\n", max_diff);
  printf("  Max relative error: %.2e\n", rel_error);
}

// Timing result with breakdown
struct TimingResult {
  float h2d_time;     // Host to Device transfer
  float kernel_time;  // Kernel execution
  float d2h_time;     // Device to Host transfer
  float total_time;   // Total end-to-end time

  void print(const char* name, float baseline_time = 0) const {
    float speedup = (baseline_time > 0) ? baseline_time / total_time : 1.0f;

    printf("%-25s: %8.3f ms", name, total_time);
    if (baseline_time > 0) {
      printf("  (%6.2fx speedup)", speedup);
    }
    printf("\n");

    printf("  └─ H2D: %.3f ms, Kernel: %.3f ms, D2H: %.3f ms\n", h2d_time,
           kernel_time, d2h_time);
  }
};

// Time a GPU convolution with full memory transfer
template <typename Func>
TimingResult time_gpu_convolution(Func conv_func, ConvBuffers& buffers,
                                  bool use_constant_memory = false,
                                  int warmup_iters = 3, int timing_iters = 10) {
  TimingResult result = {0, 0, 0, 0};

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_kernel, buffers.h_kernel,
                          buffers.params.kernel_size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    conv_func(buffers.d_input, buffers.d_kernel, buffers.d_output,
              buffers.params);
    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  for (int i = 0; i < timing_iters; i++) {
    printf("Iteration %d/%d\r", i + 1, timing_iters);
    float h2d, kernel, d2h;

    // Time H2D transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_kernel, buffers.h_kernel,
                          buffers.params.kernel_size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));

    // Time kernel
    CUDA_CHECK(cudaEventRecord(start));
    conv_func(buffers.d_input, buffers.d_kernel, buffers.d_output,
              buffers.params);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel, start, stop));

    // Time D2H transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));

    result.h2d_time += h2d;
    result.kernel_time += kernel;
    result.d2h_time += d2h;
  }

  // Average over iterations
  result.h2d_time /= timing_iters;
  result.kernel_time /= timing_iters;
  result.d2h_time /= timing_iters;
  result.total_time = result.h2d_time + result.kernel_time + result.d2h_time;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return result;
}

// Time a CPU convolution
template <typename Func>
TimingResult time_cpu_convolution(Func conv_func, ConvBuffers& buffers,
                                  int warmup_iters = 2, int timing_iters = 10) {
  TimingResult result = {0, 0, 0, 0};

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    conv_func(buffers.h_input, buffers.h_kernel, buffers.h_output,
              buffers.params);
  }

  // Timed runs
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < timing_iters; i++) {
    conv_func(buffers.h_input, buffers.h_kernel, buffers.h_output,
              buffers.params);
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;

  result.total_time = duration.count() / timing_iters;
  result.kernel_time = result.total_time;  // CPU has no transfer overhead

  return result;
}

// Time a cuDNN convolution with full memory transfer
TimingResult time_cudnn_convolution(ConvBuffers& buffers, int warmup_iters = 3,
                                    int timing_iters = 10) {
  TimingResult result = {0, 0, 0, 0};

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // Create cuDNN handle
  cudnnHandle_t cudnn;
  CUDNN_CHECK(cudnnCreate(&cudnn));

  // Create tensor descriptors
  cudnnTensorDescriptor_t input_desc, output_desc;
  cudnnFilterDescriptor_t kernel_desc;
  cudnnConvolutionDescriptor_t conv_desc;

  CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CHECK(cudnnCreateFilterDescriptor(&kernel_desc));
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

  // Set input descriptor (NCHW format)
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(
      input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, buffers.params.N,
      buffers.params.C, buffers.params.H, buffers.params.W));

  // Set filter descriptor (KCRS format: out_channels, in_channels, height,
  // width)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(
      kernel_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, buffers.params.K,
      buffers.params.C, buffers.params.R, buffers.params.S));

  // Set convolution descriptor
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
      conv_desc, buffers.params.pad, buffers.params.pad, buffers.params.stride,
      buffers.params.stride, 1, 1,  // dilation
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  // Get output dimensions
  int out_n, out_c, out_h, out_w;
  CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
      conv_desc, input_desc, kernel_desc, &out_n, &out_c, &out_h, &out_w));

  // Set output descriptor
  CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                         CUDNN_DATA_FLOAT, out_n, out_c, out_h,
                                         out_w));

  // Find best algorithm
  cudnnConvolutionFwdAlgoPerf_t perf_results;
  int returned_algo_count;
  CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
      cudnn, input_desc, kernel_desc, conv_desc, output_desc, 1,
      &returned_algo_count, &perf_results));

  cudnnConvolutionFwdAlgo_t algo = perf_results.algo;

  // Get workspace size
  size_t workspace_size;
  CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn, input_desc, kernel_desc, conv_desc, output_desc, algo,
      &workspace_size));

  void* d_workspace = nullptr;
  if (workspace_size > 0) {
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
  }

  const float alpha = 1.0f, beta = 0.0f;

  // Warmup
  for (int i = 0; i < warmup_iters; i++) {
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_kernel, buffers.h_kernel,
                          buffers.params.kernel_size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn, &alpha, input_desc, buffers.d_input, kernel_desc,
        buffers.d_kernel, conv_desc, algo, d_workspace, workspace_size, &beta,
        output_desc, buffers.d_output));

    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Timed runs
  for (int i = 0; i < timing_iters; i++) {
    printf("Iteration %d/%d\r", i + 1, timing_iters);
    float h2d, kernel, d2h;

    // Time H2D transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(buffers.d_input, buffers.h_input,
                          buffers.params.input_size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buffers.d_kernel, buffers.h_kernel,
                          buffers.params.kernel_size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d, start, stop));

    // Time kernel
    CUDA_CHECK(cudaEventRecord(start));
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn, &alpha, input_desc, buffers.d_input, kernel_desc,
        buffers.d_kernel, conv_desc, algo, d_workspace, workspace_size, &beta,
        output_desc, buffers.d_output));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel, start, stop));

    // Time D2H transfer
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(buffers.h_output, buffers.d_output,
                          buffers.params.output_size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h, start, stop));

    result.h2d_time += h2d;
    result.kernel_time += kernel;
    result.d2h_time += d2h;
  }

  // Average over iterations
  result.h2d_time /= timing_iters;
  result.kernel_time /= timing_iters;
  result.d2h_time /= timing_iters;
  result.total_time = result.h2d_time + result.kernel_time + result.d2h_time;

  // Cleanup
  if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CHECK(cudnnDestroyFilterDescriptor(kernel_desc));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDNN_CHECK(cudnnDestroy(cudnn));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return result;
}

// Calculate GFLOPS for convolution
inline float calculate_gflops(const ConvParams& params, float time_ms) {
  // Operations per output element: 2 * C * R * S (MAC operations)
  // Total outputs: N * K * out_h * out_w
  long long ops = 2LL * params.N * params.K * params.out_h() * params.out_w() *
                  params.C * params.R * params.S;
  return (ops / 1e9) / (time_ms / 1e3);  // GFLOPS
}

#endif