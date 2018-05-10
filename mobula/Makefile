TARGET = mobula_op
BUILD_PATH = build
USING_OPENMP = 0
HOST_NUM_THREADS = 8

CXX = g++
NVCC = nvcc
CUDA_DIR = /opt/cuda

COMMON_FLAGS := -D HOST_NUM_THREADS=$(HOST_NUM_THREADS)

CFLAGS := -std=c++11 -Iinc -fPIC -D USING_OPENMP=$(USING_OPENMP) $(COMMON_FLAGS)
LDFLAGS := -lpthread -shared 

CU_CFLAGS := -std=c++11 -Iinc -DUSING_CUDA=1 -Wno-deprecated-gpu-targets -dc --compiler-options "-fPIC" $(COMMON_FLAGS)
CU_LDFLAGS := -lpthread -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas -shared

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
%.o:%.cpp  
	$(CXX) $< $(CFLAGS) -c -o $@ 
%.cu.o:%.cu
	$(NVCC) $< $(CU_CFLAGS) -c -o $@ 
%.cu:%.cpp
	ln $< $@
