MOBULA_HOME = /mnt/wkcn/proj/MobulaOP

all: context.o defines.o im2col.o mobula_cpp.o
	g++ mobula_cpp.o $(MOBULA_HOME)/mobula/build/cpu/src/context.o $(MOBULA_HOME)/mobula/build/cpu/src/defines.o $(MOBULA_HOME)/mobula/build/cpu/src/im2col.o -lpthread -o test

clean:
	rm $(MOBULA_HOME)/mobula/build/cpu/src/context.o
	rm $(MOBULA_HOME)/mobula/build/cpu/src/defines.o
	rm $(MOBULA_HOME)/mobula/build/cpu/src/im2col.o
	rm mobula_cpp.o

context.o: $(MOBULA_HOME)/mobula/src/context.cpp
	g++ $(MOBULA_HOME)/mobula/src/context.cpp -std=c++11 -DUSING_CUDA=0 -DUSING_HIP=0 -DUSING_OPENMP=0  -DHOST_NUM_THREADS=4 -DMXNET_GCC_VERSION=4 -O3 -DUSING_CBLAS=0 -I$(MOBULA_HOME)/mobula/./ -I$(MOBULA_HOME)/mobula/./inc -I$(MOBULA_HOME)/mobula/../3rdparty/dlpack -I$(MOBULA_HOME)/mobula/../3rdparty/tvm_packed_func -fPIC -Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wundef -fdiagnostics-show-option -c -o $(MOBULA_HOME)/mobula/build/cpu/src/context.o

defines.o: $(MOBULA_HOME)/mobula/src/defines.cpp
	g++ $(MOBULA_HOME)/mobula/src/defines.cpp -std=c++11 -DUSING_CUDA=0 -DUSING_HIP=0 -DUSING_OPENMP=0  -DHOST_NUM_THREADS=4 -DMXNET_GCC_VERSION=4 -O3 -DUSING_CBLAS=0 -I$(MOBULA_HOME)/mobula/./ -I$(MOBULA_HOME)/mobula/./inc -I$(MOBULA_HOME)/mobula/../3rdparty/dlpack -I$(MOBULA_HOME)/mobula/../3rdparty/tvm_packed_func -fPIC -Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wundef -fdiagnostics-show-option -c -o $(MOBULA_HOME)/mobula/build/cpu/src/defines.o

im2col.o: $(MOBULA_HOME)/mobula/src/im2col.cpp
	g++ $(MOBULA_HOME)/mobula/src/im2col.cpp -std=c++11 -DUSING_CUDA=0 -DUSING_HIP=0 -DUSING_OPENMP=0  -DHOST_NUM_THREADS=4 -DMXNET_GCC_VERSION=4 -O3 -DUSING_CBLAS=0 -I$(MOBULA_HOME)/mobula/./ -I$(MOBULA_HOME)/mobula/./inc -I$(MOBULA_HOME)/mobula/../3rdparty/dlpack -I$(MOBULA_HOME)/mobula/../3rdparty/tvm_packed_func -fPIC -Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wundef -fdiagnostics-show-option -c -o $(MOBULA_HOME)/mobula/build/cpu/src/im2col.o

mobula_cpp.o: mobula_cpp.cpp
	g++ mobula_cpp.cpp -std=c++11 -DUSING_CUDA=0 -DUSING_HIP=0 -DUSING_OPENMP=0  -DHOST_NUM_THREADS=4 -DMXNET_GCC_VERSION=4 -O3 -DUSING_CBLAS=0 -I$(MOBULA_HOME)/mobula/./ -I$(MOBULA_HOME)/mobula/./inc -I$(MOBULA_HOME)/mobula/inc/capi -I$(MOBULA_HOME)/mobula/../3rdparty/dlpack -I$(MOBULA_HOME)/mobula/../3rdparty/tvm_packed_func -fPIC -Werror -Wall -Wextra -pedantic -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wmissing-include-dirs -Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-promo -Wundef -fdiagnostics-show-option -c -o mobula_cpp.o 
