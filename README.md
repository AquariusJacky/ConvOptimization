# CUDA Convolution Optimization Practice

This project demonstrates progressive optimization of CUDA convolution kernels using Nsight Compute (ncu) for profiling and performance analysis. Each implementation stage builds understanding of what to look for when profiling CUDA kernels.

## Project Overview

The project implements 2D convolution through different optimization stages, comparing performance metrics at each step. All kernel implementations use consistent parameters and inputs for fair comparison.

### Input Specifications
- **Input tensor**: 4D shape in NCHW format (Batch, Channels, Height, Width)
- **Kernel (weights)**: KCRS format (output channels, input channels, height, width)
- **Convolution parameters**: Supports configurable stride and padding
- **Memory management**: RAII-style allocation for both CPU and GPU
- **Timing measurement**: CUDA events record host-to-device (H2D), kernel execution, and device-to-host (D2H) times

## Implementation Stages

### 1. CPU Version
Baseline implementation for performance comparison. All GPU kernels are benchmarked against this reference implementation.

### 2. Naive GPU
First GPU implementation without optimizations. Each block calculates `C × TILE_WIDTH × TILE_HEIGHT` outputs using only register files. Provides baseline GPU performance metrics.

### 3. Shared GPU
Introduces shared memory optimization for the input matrix. Since the entire input matrix cannot fit in shared memory simultaneously, the kernel iterates through input channels (C dimension) in multiple passes to compute all outputs assigned to each block.
Note: Uncoalesced global memory access may be the biggest optimization according to ncu. Also, since we can't fit all input channels in shared memory, we have to split the channels internally during calculation.

### 4. Shared_both GPU
Extends shared memory usage to both input and kernel (weight) matrices. Modified to use 3D block shapes due to shared memory size constraints. This version was developed because the Shared GPU version performed worse than Naive GPU.

Note: Since the shared version is worse than naive, this is an attempt to see if moving kernel to shared memory helps. From here it's much harder to do indexing.

### 5. im2col + Tensor core GPU 
Algorithmic transformation using the im2col (image-to-column) method. Unlike previous versions which are iterative optimizations, this approach fundamentally changes the algorithm by transforming convolution into matrix multiplication to reduce computational complexity.

Note: At this point probably try out im2col + gemm approach because ncu shows that the kernel is compute bound.

Note 2: After I learned more about im2col, I found out that the previous methods were all thrown away. It's not an improvement over the   previous methods, it's a completely different approach. Hopefully this one will work better.

### 6. im2col + Tensor core GPU + NCHW to NHWC conversion
Learning that NHWC layout decreases memory read uncoalesce, I attempted to do it by myself.

Note: I am amazed by how fast im2col can be when implemented correctly. This implementation is an attempt to convert the input and kernel tensors from NCHW to NHWC format, then perform im2col + GEMM using WMMA. This should improve memory coalescing during the im2col step. The conversion kernels are simple and straightforward.
  
Note 2: The main convolution kernel is similar to the previous im2col + WMMA kernel, except that the indexing for input tensor is changed to NHWC format. This should yield better performance due to improved memory access patterns.
  Note that the input and kernel tensors are converted to half-precision before being passed to the convolution kernel. This reduces memory bandwidth requirements and speeds up computation. The output tensor remains in single-precision for accuracy. Overall, this approach leverages the strengths of im2col and WMMA while optimizing memory access patterns through NHWC layout. Hopefully, this will lead to significant performance improvements.
Note 3: (Spoiler, it didn't)

### 7. cuDNN
Comparison implementation using NVIDIA's cuDNN library. This serves as a benchmark against production-grade, highly optimized convolution implementations. Apparently, the speedup is not possible for me at my current level.

## Build Instructions

### Prerequisites
- CUDA Toolkit (with nvcc compiler)
- g++ compiler
- Nsight Compute (ncu) for profiling
- Nsight Systems (nsys) for system-wide profiling (optional)
- GNU Make

### Building the Project

```bash
# Build all targets
make

# Run the benchmark
make run
```

### Profiling

```bash
# Profile with Nsight Compute
ncu ./bin/conv_benchmark

# Profile with Nsight Systems (optional)
nsys profile ./bin/conv_benchmark
```

### Available Targets

```bash
make          # Build the project
make run      # Build and run the benchmark
make clean    # Remove build artifacts
```

## Performance Analysis Workflow

1. Run baseline CPU and Naive GPU implementations
2. Profile with `ncu` to identify bottlenecks
3. Implement optimization based on profiling insights
4. Compare performance metrics with previous versions
5. Iterate based on findings

## Key Learning Points

- Not all optimizations improve performance (e.g., Shared GPU vs Naive GPU)
- Shared memory size constraints affect block configuration choices
- Different algorithms (like im2col) can fundamentally change performance characteristics
- Profiling tools like ncu are essential for understanding actual vs. expected performance
- Roofline model is your friend.

## Future Work

- Add additional optimization techniques (e.g., register tiling, warp-level optimizations)
- Add support for dilation parameter
- Performance analysis across different input sizes and kernel configurations
