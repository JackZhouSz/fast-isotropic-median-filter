CXX_STANDARD ?= c++17
CXX := clang++
NVCC := nvcc
CXXFLAGS := -std=$(CXX_STANDARD) -Wall -Werror -O3 -v

# NVIDIA compute capabilities to build for. { 61 70 75 80 86 89 90 101 120 }
GPU_ARCHS ?= 86 89 120  # 86 => RTX 30x0,  89 => RTX 40x0,  120 => RTX 50x0
CUDA_DIR = /usr/local/cuda
CUDA_INCLUDES = -I$(CUDA_DIR)/include
GENCODE = $(foreach arch,$(GPU_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NVCCFLAGS = $(CUDA_INCLUDES) $(GENCODE) --std=$(CXX_STANDARD) -O3 --expt-relaxed-constexpr -v

# By default, the benchmark is built as ordinary C++.
BENCHMARK_CXX := $(CXX)
BENCHMARK_CXXFLAGS := $(CXXFLAGS)

ifeq ($(shell uname -m),x86_64)
  ARCH_FLAGS = -mavx2 -mbmi2
  TARGET_ARCH = x86_64
  BENCHMARK_CXX := $(NVCC)  # On x86_64, the benchmark is built as CUDA.
  BENCHMARK_CXXFLAGS := $(NVCCFLAGS)
else ifeq ($(shell uname -m),aarch64)
  ARCH_FLAGS =
  TARGET_ARCH = aarch64
else ifeq ($(shell uname -m),arm64)
  ARCH_FLAGS =
  TARGET_ARCH = aarch64
else
  ARCH_FLAGS =  # Default flags (no specific architecture optimizations)
  TARGET_ARCH = generic
endif

# Linking type: static or dynamic (default: static)
LINK_TYPE ?= static
ifeq ($(LINK_TYPE),static)
  LINK_FLAG = -static
  ABSL_LINK_TYPE_FLAG = -Wl,-Bstatic # Force static linking for Abseil
  OPENCV_LINK_TYPE_FLAG = -Wl,-Bstatic # Force static linking for OpenCV
else
  LINK_FLAG = 
  ABSL_LINK_TYPE_FLAG = -Wl,-Bdynamic # Force dynamic linking for Abseil
  OPENCV_LINK_TYPE_FLAG = -Wl,-Bdynamic # Force dynamic linking for OpenCV
endif

ABSL_CFLAGS = $(shell pkg-config --cflags absl_base absl_check absl_flags absl_flags_parse absl_span absl_status absl_statusor)
ABSL_LIBS = $(shell pkg-config --libs absl_base absl_check absl_flags absl_flags_parse absl_span absl_status absl_statusor  | tr ' ' '\n' | awk '!x[$$0]++' | tr '\n' ' ')

# OpenCV is used only for file I/O and image padding.
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)

LIB_DIR = lib
BIN_DIR = bin

INCLUDES =  -I. $(ABSL_CFLAGS) $(OPENCV_CFLAGS)
LIBS = -L$(LIB_DIR)  $(ABSL_LINK_TYPE_FLAG) $(ABSL_LIBS) $(OPENCV_LINK_TYPE_FLAG) $(OPENCV_LIBS) -lpthread -ldl -lz -lm
LDFLAGS = $(LINK_FLAG) # General linker flags

MKDIR_P = mkdir -p

SRCS_ORDINAL_TRANSFORM_AVX2 = ordinal_transform_avx2.cc
SRCS_ORDINAL_TRANSFORM_NEON = ordinal_transform_neon.cc
SRCS_ORDINAL_TRANSFORM = ordinal_transform.cc
SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_AVX2 = fast_isotropic_median_filter_avx2.cc
SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_NEON = fast_isotropic_median_filter_neon.cc
SRCS_FAST_ISOTROPIC_MEDIAN_FILTER = fast_isotropic_median_filter.cc
SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE = fast_isotropic_median_filter_simple.cc
SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA = fast_isotropic_median_filter_cuda.cu
SRCS_RUN = run.cc
SRCS_TEST = test.cc
SRCS_BENCHMARK = benchmark.cc

OBJS_ORDINAL_TRANSFORM_AVX2 = $(addprefix $(BIN_DIR)/,$(SRCS_ORDINAL_TRANSFORM_AVX2:.cc=.o))
OBJS_ORDINAL_TRANSFORM_NEON = $(addprefix $(BIN_DIR)/,$(SRCS_ORDINAL_TRANSFORM_NEON:.cc=.o))
OBJS_ORDINAL_TRANSFORM = $(addprefix $(BIN_DIR)/,$(SRCS_ORDINAL_TRANSFORM:.cc=.o))
OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_AVX2 = $(addprefix $(BIN_DIR)/,$(SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_AVX2:.cc=.o))
OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_NEON = $(addprefix $(BIN_DIR)/,$(SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_NEON:.cc=.o))
OBJS_FAST_ISOTROPIC_MEDIAN_FILTER = $(addprefix $(BIN_DIR)/,$(SRCS_FAST_ISOTROPIC_MEDIAN_FILTER:.cc=.o))
OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE = $(addprefix $(BIN_DIR)/,$(SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE:.cc=.o))
OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA = $(addprefix $(BIN_DIR)/,$(SRCS_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA:.cu=.o))
OBJS_RUN = $(BIN_DIR)/run.o
OBJS_TEST = $(BIN_DIR)/test.o
OBJS_BENCHMARK = $(BIN_DIR)/benchmark.o

LIB_ORDINAL_TRANSFORM = $(LIB_DIR)/libordinal_transform.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER = $(LIB_DIR)/libfast_isotropic_median_filter.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE = $(LIB_DIR)/libfast_isotropic_median_filter_simple.a
LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA = $(LIB_DIR)/libfast_isotropic_median_filter_cuda.a

RUN_EXE = $(BIN_DIR)/run
TEST_EXE = $(BIN_DIR)/test
BENCHMARK_EXE = $(BIN_DIR)/benchmark

all: $(RUN_EXE) $(TEST_EXE) $(BENCHMARK_EXE)

ifeq ($(TARGET_ARCH),x86_64)
  ORDINAL_TRANSFORM_OBJS = $(OBJS_ORDINAL_TRANSFORM) $(OBJS_ORDINAL_TRANSFORM_AVX2)
else ifeq ($(TARGET_ARCH),aarch64)
  ORDINAL_TRANSFORM_OBJS = $(OBJS_ORDINAL_TRANSFORM) $(OBJS_ORDINAL_TRANSFORM_NEON)
else
  ORDINAL_TRANSFORM_OBJS = $(OBJS_ORDINAL_TRANSFORM)
endif

$(LIB_ORDINAL_TRANSFORM): $(ORDINAL_TRANSFORM_OBJS)
	$(MKDIR_P) $(@D)
	ar rcs $@ $(ORDINAL_TRANSFORM_OBJS)

ifeq ($(TARGET_ARCH),x86_64)
  FAST_ISOTROPIC_MEDIAN_FILTER_OBJS = $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER) $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_AVX2)
else ifeq ($(TARGET_ARCH),aarch64)
  FAST_ISOTROPIC_MEDIAN_FILTER_OBJS = $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER) $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_NEON)
else
  FAST_ISOTROPIC_MEDIAN_FILTER_OBJS = $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER)
endif

$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER): $(FAST_ISOTROPIC_MEDIAN_FILTER_OBJS)
	$(MKDIR_P) $(@D)
	ar rcs $@ $(FAST_ISOTROPIC_MEDIAN_FILTER_OBJS)

$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE): $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE)
	$(MKDIR_P) $(@D)
	ar rcs $@ $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE)

FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_OBJS = $(OBJS_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA)

$(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA): $(FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_OBJS)
	$(MKDIR_P) $(@D)
	ar rcs $@ $(FAST_ISOTROPIC_MEDIAN_FILTER_CUDA_OBJS)

$(RUN_EXE): $(OBJS_RUN) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_ORDINAL_TRANSFORM)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) $(INCLUDES) $(LDFLAGS) -o $(RUN_EXE) $(OBJS_RUN) -L$(LIB_DIR) -lfast_isotropic_median_filter -lordinal_transform $(LIBS)

$(TEST_EXE): $(OBJS_TEST) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_SIMPLE) $(LIB_ORDINAL_TRANSFORM)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) $(INCLUDES) $(LDFLAGS) -o $(TEST_EXE) $(OBJS_TEST) -L$(LIB_DIR) -lfast_isotropic_median_filter -lfast_isotropic_median_filter_simple -lordinal_transform $(LIBS)

ifeq ($(shell uname -m),x86_64)
$(BENCHMARK_EXE): $(OBJS_BENCHMARK) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER_CUDA) $(LIB_ORDINAL_TRANSFORM)
	mkdir -p $(@D)
	$(BENCHMARK_CXX) $(BENCHMARK_CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $(BENCHMARK_EXE) $(OBJS_BENCHMARK) -L$(LIB_DIR) -lfast_isotropic_median_filter -lfast_isotropic_median_filter_cuda -lordinal_transform $(LIBS)
else  # Exclude the CUDA library on non-x86_64.
$(BENCHMARK_EXE): $(OBJS_BENCHMARK) $(LIB_FAST_ISOTROPIC_MEDIAN_FILTER) $(LIB_ORDINAL_TRANSFORM)
	mkdir -p $(@D)
	$(BENCHMARK_CXX) $(BENCHMARK_CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $(BENCHMARK_EXE) $(OBJS_BENCHMARK) -L$(LIB_DIR) -lfast_isotropic_median_filter -lordinal_transform $(LIBS)
endif

$(BIN_DIR)/%.o: %.cc
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(ARCH_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/benchmark.o: benchmark.cc
	mkdir -p $(@D)
	$(BENCHMARK_CXX) $(BENCHMARK_CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(LIB_DIR) $(BIN_DIR)

.PHONY: all clean
