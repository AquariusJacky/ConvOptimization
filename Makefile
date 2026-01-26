# Compiler and tools
NVCC = nvcc
CXX = g++
ARCH = sm_89 # RTX 4060 ti

# Directories
SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
BIN_DIR = bin

# Compiler flags
NVCCFLAGS = -g -lineinfo -O3 --use_fast_math -arch=$(ARCH) -std=c++17
CXXFLAGS = -O3 -std=c++17 -Wall
INCLUDES = -I$(INC_DIR)
LIBS = -lcudart -lcudnn

# Optional: Enable line info for Nsight Compute profiling
# Uncomment the next line when profiling
# NVCCFLAGS += -lineinfo

# Optional: Enable debug mode
# Uncomment for debugging
# NVCCFLAGS += -g -G
# CXXFLAGS += -g

# Source files
CPU_SRC = $(SRC_DIR)/conv_cpu.cpp
CUDA_SRC = $(SRC_DIR)/conv_naive.cu \
           $(SRC_DIR)/conv_shared.cu \
           $(SRC_DIR)/conv_shared_both.cu \
           $(CUTLASS_SRC)
MAIN_SRC = $(SRC_DIR)/benchmark.cu

# Object files
CPU_OBJ = $(BUILD_DIR)/conv_cpu.o
CUDA_OBJ = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SRC))
MAIN_OBJ = $(BUILD_DIR)/benchmark.o

ALL_OBJ = $(MAIN_OBJ) $(CPU_OBJ) $(CUDA_OBJ)

# Output executable
TARGET = $(BIN_DIR)/conv_benchmark

# Default target
all: dirs $(TARGET)

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Link executable
$(TARGET): $(ALL_OBJ)
	@echo "Linking $@..."
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LIBS)
	@echo "Build complete: $@"

# Compile CUDA source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@echo "Compiling $<..."
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -dc $< -o $@

# Compile C++ source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Run benchmark
run: $(TARGET)
	@echo "Running benchmark..."
	@./$(TARGET)

# Profile with Nsight Compute (detailed kernel analysis)
profile: $(TARGET)
	@echo "Profiling with Nsight Compute..."
	ncu --set full -o profile_ncu $(TARGET)
	@echo "Profile saved to profile_ncu.ncu-rep"
	@echo "Open with: ncu-ui profile_ncu.ncu-rep"

# Profile specific kernel with Nsight Compute
profile-kernel: $(TARGET)
	@echo "Profiling kernels matching 'conv_'..."
	ncu --kernel-name regex:"conv_.*" --launch-skip 5 --launch-count 10 -o profile_kernel $(TARGET)

# Profile with Nsight Systems (timeline view)
profile-sys: $(TARGET)
	@echo "Profiling with Nsight Systems..."
	nsys profile -o profile_nsys --trace=cuda,nvtx $(TARGET)
	@echo "Profile saved to profile_nsys.nsys-rep"
	@echo "Open with: nsys-ui profile_nsys.nsys-rep"

# Quick profile (less detail, faster)
profile-quick: $(TARGET)
	@echo "Quick profiling..."
	ncu --set basic -o profile_quick $(TARGET)

# Check CUDA installation and GPU info
info:
	@echo "=== CUDA Information ==="
	@$(NVCC) --version
	@echo ""
	@echo "=== GPU Information ==="
	@nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
	@echo ""
	@echo "=== Build Configuration ==="
	@echo "Architecture: $(ARCH)"
	@echo "Optimization: -O3"
	@echo "cuDNN: Enabled"
	@echo "CUTLASS: $(if $(CUTLASS_SRC),Enabled,Disabled)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f *.o

# Clean everything including profile results
cleanall: clean
	@echo "Cleaning profile results..."
	rm -rf *.ncu-rep *.nsys-rep profile_* *.qdrep *.sqlite

# Help target
help:
	@echo "Available targets:"
	@echo "  make              - Build the project (default)"
	@echo "  make run          - Build and run benchmark"
	@echo "  make profile      - Profile with Nsight Compute (full metrics)"
	@echo "  make profile-quick- Profile with Nsight Compute (basic metrics)"
	@echo "  make profile-sys  - Profile with Nsight Systems (timeline)"
	@echo "  make profile-kernel - Profile specific kernels"
	@echo "  make info         - Show CUDA and GPU information"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make cleanall     - Remove build artifacts and profiles"
	@echo "  make help         - Show this help message"
	@echo ""
	@echo "Build options (set as environment variables):"
	@echo "  ARCH=sm_XX        - Set GPU architecture (default: sm_80)"
	@echo ""
	@echo "Examples:"
	@echo "  make ARCH=sm_89   - Build for RTX 4090"
	@echo "  make run          - Build and run"
	@echo "  make profile      - Build and profile"

# Phony targets (not actual files)
.PHONY: all dirs run profile profile-kernel profile-sys profile-quick info clean cleanall help

# Print variables (for debugging Makefile)
print-%:
	@echo $* = $($*)