TARGET = mobula_op
BUILD_PATH = build
USING_OPENMP = 0
HOST_NUM_THREADS = 8
USING_HIGH_LEVEL_WARNINGS = 1
USING_OPTIMIZATION = 1

CXX = g++
NVCC = nvcc
CUDA_DIR = /opt/cuda

COMMON_FLAGS := -D HOST_NUM_THREADS=$(HOST_NUM_THREADS)
ifeq ($(USING_OPTIMIZATION), 1)
	COMMON_FLAGS += -O3
endif

CFLAGS := -std=c++11 -Iinc -fPIC -DUSING_CUDA=0 -DUSING_OPENMP=$(USING_OPENMP) $(COMMON_FLAGS)
LDFLAGS := -lpthread -shared

CU_CFLAGS := -std=c++11 -Iinc -DUSING_CUDA=1 -Wno-deprecated-gpu-targets -dc --compiler-options "-fPIC" $(COMMON_FLAGS)
CU_LDFLAGS := -lpthread -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas -shared

ifeq ($(USING_OPENMP), 1)
	CFLAGS += -fopenmp
	LDFLAGS += -fopenmp
endif

ifeq ($(USING_HIGH_LEVEL_WARNINGS), 1)
	CFLAGS += -Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wundef -fdiagnostics-show-option
endif

SRCS := $(wildcard src/*.cpp src/operators/*.cpp)
OBJS := $(patsubst %cpp,%o,$(SRCS))

CU_SRCS := $(patsubst %cpp,%cu,$(SRCS))
CU_OBJS := $(patsubst %cu,%cu.o,$(CU_SRCS))

all: $(OBJS)
	mkdir -p $(BUILD_PATH) 
	$(CXX) $^ $(LDFLAGS) -o $(BUILD_PATH)/$(TARGET)_cpu.so
cuda: $(CU_OBJS)
	mkdir -p $(BUILD_PATH)
	$(NVCC) $^ $(CU_LDFLAGS) -o $(BUILD_PATH)/$(TARGET)_gpu.so
clean:
	rm -f $(BUILD_PATH)/$(TARGET)_?pu.so $(CU_SRCS) src/*.o
clean_all:  
	rm -f $(BUILD_PATH)/$(TARGET)_?pu.so $(CU_SRCS) $(OBJS) $(CU_OBJS)
rebuild: clean_all all
%.o:%.cpp  
	$(CXX) $< $(CFLAGS) -c -o $@ 
%.cu.o:%.cu
	$(NVCC) $< $(CU_CFLAGS) -c -o $@ 
%.cu:%.cpp
	ln $< $@
