# --- Configuration ---
CXX := clang++  # GCC-compatible, but clang generates more efficient code.
NVCC ?= nvcc
CXX_STANDARD ?= c++17

# Set USE_GPU=0 to build for CPU only.
# USE_GPU defaults to 1 if an NVIDIA GPU and CUDA are likely available.
ifeq ($(shell which nvcc 2>/dev/null),)
    USE_GPU ?= 0
else
    USE_GPU ?= 1
endif

# --- Output Directories ---
LIB_DIR := lib
BIN_DIR := bin

# --- Base Flags ---
CXXFLAGS := -std=$(CXX_STANDARD) -Wall -Werror -O3 \
            -Wno-sign-compare \
            -Wno-parentheses \
            -Wno-ignored-attributes

# --- Architecture-Specific Setup ---
TARGET_ARCH := $(shell uname -m)

ifeq ($(TARGET_ARCH),x86_64)
  ARCH_FLAGS   = -mavx2 -mbmi -mbmi2
else ifeq ($(filter aarch64 arm64,$(TARGET_ARCH)),$(TARGET_ARCH))
  TARGET_ARCH := aarch64
  ARCH_FLAGS  =
  ifneq ($(USE_GPU),0)
    $(info INFO: Disabling GPU build for aarch64 architecture.)
  endif
  override USE_GPU = 0
else
  TARGET_ARCH := generic
  ARCH_FLAGS  =
endif

# Appends architecture-specific flags to the main CXXFLAGS.
CXXFLAGS += $(ARCH_FLAGS)

# --- Source File Definitions ---
SRC_ORD_TRANSFORM_CPU     := ordinal_transform.cc
SRC_ISO_FILTER_CPU        := fast_isotropic_median_filter.cc
SRC_ISO_FILTER_CPU_SIMPLE := fast_isotropic_median_filter_simple.cc
SRC_ISO_FILTER_CUDA       := fast_isotropic_median_filter_cuda.cu
SRC_MAIN                  := run.cc
SRC_TEST                  := test.cc
SRC_BENCHMARK             := benchmark.cc

# Adds architecture-specific sources to the lists.
ifeq ($(TARGET_ARCH),x86_64)
  SRC_ORD_TRANSFORM_CPU += ordinal_transform_avx2.cc
  SRC_ISO_FILTER_CPU    += fast_isotropic_median_filter_avx2.cc
else ifeq ($(TARGET_ARCH),aarch64)
  SRC_ORD_TRANSFORM_CPU += ordinal_transform_neon.cc
  SRC_ISO_FILTER_CPU    += fast_isotropic_median_filter_neon.cc
endif

# --- Library and Executable Definitions ---
OBJS_ORD_TRANSFORM_CPU     := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_ORD_TRANSFORM_CPU))
OBJS_ISO_FILTER_CPU        := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_ISO_FILTER_CPU))
OBJS_ISO_FILTER_CPU_SIMPLE := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_ISO_FILTER_CPU_SIMPLE))
OBJS_ISO_FILTER_CUDA       := $(patsubst %.cu,$(BIN_DIR)/%.o,$(SRC_ISO_FILTER_CUDA))
OBJS_MAIN                  := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_MAIN))
OBJS_TEST                  := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_TEST))
OBJS_BENCHMARK             := $(patsubst %.cc,$(BIN_DIR)/%.o,$(SRC_BENCHMARK))

# Static libraries
LIB_ORD_TRANSFORM                 := $(LIB_DIR)/libordinal_transform.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER  := $(LIB_DIR)/libfast_isotropic_median_filter.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE := $(LIB_DIR)/libfast_isotropic_median_filter_simple.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA   := $(LIB_DIR)/libfast_isotropic_median_filter_cuda.a

# Final executables
RUN_EXE       := $(BIN_DIR)/run
TEST_EXE      := $(BIN_DIR)/test
BENCHMARK_EXE := $(BIN_DIR)/benchmark

# --- GPU / CUDA Configuration ---
ifeq ($(USE_GPU),1)
  # Try to find CUDA_DIR automatically, but allow user override.
  CUDA_DIR ?= $(shell dirname $(shell dirname $(shell which nvcc)))
  CUDA_INCLUDES := -I$(CUDA_DIR)/include

  # NVIDIA compute capabilities to build for.
  # Valid architectures are { 61 70 75 80 86 89 90 101 120 }.
  # [86 => RTX 30x0,  89 => RTX 40x0,  120 => RTX 50x0]
  GPU_ARCHS ?= 86 89 120
  GENCODE   := $(foreach arch,$(GPU_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
  NVCCFLAGS := $(GENCODE) --expt-relaxed-constexpr -Wno-deprecated-gpu-targets

  # --- Benchmark Target (GPU version) ---
  BENCHMARK_COMPILER := $(NVCC)
  NVCC_HOST_FLAGS    := -std=$(CXX_STANDARD) -O3 -Wno-sign-compare -Wno-parentheses -Wno-ignored-attributes
  NVCC_COMPILE_FLAGS := $(NVCCFLAGS) --compiler-bindir=$(CXX) $(foreach flag,$(NVCC_HOST_FLAGS),-Xcompiler $(flag))
  BENCHMARK_FLAGS    := $(NVCC_COMPILE_FLAGS) -DBENCHMARK_GPU=1
  BENCHMARK_DEPS     := $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA)
  BENCHMARK_LIBS     := -lfast_isotropic_median_filter_cuda
else
  # --- Benchmark Target (CPU version) ---
  BENCHMARK_COMPILER := $(CXX)
  BENCHMARK_FLAGS    := $(CXXFLAGS) -DBENCHMARK_GPU=0
  BENCHMARK_DEPS     :=
  BENCHMARK_LIBS     :=
  CUDA_INCLUDES      :=
endif

# --- External Libraries Setup ---
ABSL_COMPONENTS := absl_log absl_base absl_check absl_flags absl_flags_parse absl_log_initialize absl_span absl_status absl_statusor
_ABSL_LIBS      := $(shell pkg-config --libs $(ABSL_COMPONENTS) | tr ' ' '\n' | awk '!seen[$$0]++' | tr '\n' ' ')

OPENCV_LIBS   := $(shell pkg-config --libs opencv4)

INCLUDES      := -I. $(CUDA_INCLUDES) $(shell pkg-config --cflags $(ABSL_COMPONENTS) opencv4)
COMMON_LIBS   := $(_ABSL_LIBS) $(OPENCV_LIBS) -lpthread -ldl -lz -lm

# For NVCC, filters out linker state flags that it may not support.
_ABSL_LIBS_FOR_NVCC := $(shell echo "$(_ABSL_LIBS)" | sed 's/-Wl,--push-state,--as-needed//g' | sed 's/-Wl,--pop-state//g')
NVCC_COMMON_LIBS := $(_ABSL_LIBS_FOR_NVCC) $(OPENCV_LIBS) -lpthread -ldl -lz -lm

# --- Main Targets ---
all: $(RUN_EXE) $(TEST_EXE) $(BENCHMARK_EXE)

run: $(RUN_EXE)
test: $(TEST_EXE)
benchmark: $(BENCHMARK_EXE)

# --- Linking Executables ---
$(RUN_EXE): $(OBJS_MAIN) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_ORD_TRANSFORM) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_MAIN) -L$(LIB_DIR) -lfast_isotropic_median_filter -lordinal_transform $(COMMON_LIBS)

$(TEST_EXE): $(OBJS_TEST) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE) $(LIB_ORD_TRANSFORM) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS_TEST) -L$(LIB_DIR) -lfast_isotropic_median_filter -lfast_isotropic_median_filter_simple -lordinal_transform $(COMMON_LIBS)

$(BENCHMARK_EXE): $(OBJS_BENCHMARK) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_ORD_TRANSFORM) $(BENCHMARK_DEPS) | $(BIN_DIR)
	$(BENCHMARK_COMPILER) -o $@ $(OBJS_BENCHMARK) -L$(LIB_DIR) -lfast_isotropic_median_filter -lordinal_transform $(BENCHMARK_LIBS) $(NVCC_COMMON_LIBS)

# --- Archiving Static Libraries ---
$(LIB_ORD_TRANSFORM): $(OBJS_ORD_TRANSFORM_CPU) | $(LIB_DIR)
	ar rcs $@ $^

$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER): $(OBJS_ISO_FILTER_CPU) | $(LIB_DIR)
	ar rcs $@ $^

$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE): $(OBJS_ISO_FILTER_CPU_SIMPLE) | $(LIB_DIR)
	ar rcs $@ $^

ifeq ($(USE_GPU),1)
$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA): $(OBJS_ISO_FILTER_CUDA) | $(LIB_DIR)
	ar rcs $@ $^
endif

# --- Compilation Rules ---
# Rule for compiling the benchmark source, which uses a conditional compiler.
$(BIN_DIR)/benchmark.o: $(SRC_BENCHMARK) | $(BIN_DIR)
	$(BENCHMARK_COMPILER) $(BENCHMARK_FLAGS) $(INCLUDES) -c $< -o $@

# Generic rule for C++ files (.cc -> .o).
$(BIN_DIR)/%.o: %.cc | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Generic rule for CUDA files (.cu -> .o), only active if GPU is used.
ifeq ($(USE_GPU),1)
$(BIN_DIR)/%.o: %.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_COMPILE_FLAGS) $(INCLUDES) -c $< -o $@
endif

# --- Housekeeping ---
# Order-only prerequisites to create directories.
$(BIN_DIR) $(LIB_DIR):
	@mkdir -p $@

clean:
	@echo "Cleaning up generated files..."
	@rm -rf $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean help run test benchmark

# Sets the default goal to 'all'.
.DEFAULT_GOAL := all

help:
	@echo "Usage: make [TARGET] [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Builds all executables (default)."
	@echo "  run        Builds the 'run' executable."
	@echo "  test       Builds the 'test' executable."
	@echo "  benchmark  Builds the 'benchmark' executable."
	@echo "  clean      Removes all generated files."
	@echo "  help       Prints this help message."
	@echo ""
	@echo "Options:"
	@echo "  USE_GPU=<0|1>       Force CPU-only (0) or enable GPU (1) build."
	@echo "                      (Default is 1 if nvcc is found, 0 otherwise)."
	@echo "  CXX=<compiler>      Specify C++ compiler (e.g., CXX=g++)."
	@echo "  CUDA_DIR=<path>     Specify path to CUDA toolkit."
