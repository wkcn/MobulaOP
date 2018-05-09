#ifndef _MOBULA_DEFINES_
#define _MOBULA_DEFINES_

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <utility>
#include <typeinfo>
#include <cassert>
#include <memory>
#include <thread>
#include <mutex>
using namespace std;

namespace mobula {

#ifdef USING_CUDA

#include <cuda_runtime.h>
#define CUDA_NUM_THREADS 512
#define CUDA_GET_BLOCKS(n) ((n) + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS

#define MOBULA_KERNEL __global__ void 
#define KERNEL_LOOP(i,n) for (int i = blockIdx.x * blockDim.x + threadIdx.x;i < (n);i += blockDim.x * gridDim.x)
#define KERNEL_RUN(a, n) (a)<<<CUDA_GET_BLOCKS(n), CUDA_NUM_THREADS>>>

#else

extern map<thread::id, pair<int, int> > MOBULA_KERNEL_INFOS;
extern mutex MOBULA_KERNEL_MUTEX;
#define HOST_NUM_THREADS 8

template<typename Func>
class KernelRunner{
public:
	KernelRunner(Func func, int n):_func(func), _n(n <= HOST_NUM_THREADS ? n : HOST_NUM_THREADS){};
	template<typename ...Args>
	void operator()(Args... args){
		vector<thread> threads(_n);
		MOBULA_KERNEL_MUTEX.lock();
		for (int i = 0;i < _n;++i){
			threads[i] = thread(_func, args...);
			thread::id id = threads[i].get_id();
			MOBULA_KERNEL_INFOS[id] = make_pair(i, _n);
		}
		MOBULA_KERNEL_MUTEX.unlock();
		for (int i = 0;i < _n;++i){
			threads[i].join();
		}
	}
private:
	Func _func;
	int _n;
};

#define MOBULA_KERNEL void
#define KERNEL_LOOP(i,n) MOBULA_KERNEL_MUTEX.lock(); \
						 const pair<int, int> MOBULA_KERNEL_INFO = MOBULA_KERNEL_INFOS[this_thread::get_id()]; \
						 MOBULA_KERNEL_INFOS.erase(this_thread::get_id()); \
						 MOBULA_KERNEL_MUTEX.unlock(); \
						 const int MOBULA_KERNEL_START = MOBULA_KERNEL_INFO.first; \
						 const int MOBULA_KERNEL_STEP = MOBULA_KERNEL_INFO.second; \
						 for(int i = MOBULA_KERNEL_START;i < (n);i += MOBULA_KERNEL_STEP)
#define KERNEL_RUN(a, n) (KernelRunner<decltype(&a)>(&a, (n)))

#endif

};

#endif
