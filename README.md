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

### 4. Shared_both GPU
Extends shared memory usage to both input and kernel (weight) matrices. Modified to use 3D block shapes due to shared memory size constraints. This version was developed because the Shared GPU version performed worse than Naive GPU.

### 5. im2col + Tensor core GPU *(In Progress)*
Algorithmic transformation using the im2col (image-to-column) method. Unlike previous versions which are iterative optimizations, this approach fundamentally changes the algorithm by transforming convolution into matrix multiplication to reduce computational complexity.

### 6. cuDNN
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

- Complete im2col implementation
- Add additional optimization techniques (e.g., register tiling, warp-level optimizations)
- Explore tensor core usage for mixed-precision convolutions
- Add support for dilation parameter
- Performance analysis across different input sizes and kernel configurations
