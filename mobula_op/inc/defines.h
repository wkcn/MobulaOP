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
#include <cmath>
#include <memory>
#include <cfloat>

#include "context/context.h"

namespace mobula {

typedef float DType;

template<typename T = DType>
struct CArray{
    int size;
    T* data;
};

// Allocate Memory
template<typename T = DType>
T* xnew(int size) {
#if USING_CUDA
	T *p;
	cudaMalloc((void **)&p, sizeof(T) * size);
	return p;
#else
	return new T[size];
#endif
}

template<typename T = DType>
void xdel(T* p) {
#if USING_CUDA
	cudaFree(p);
#else
	delete []p;
#endif
}

template<typename F, typename T = DType>
inline MOBULA_DEVICE void mobula_map(F func, const T *data, const int n, const int stride = 1, T *out = nullptr) {
    if (out == nullptr) out = const_cast<T*>(data);
    for (int i = 0, j = 0; i < n; ++i, j += stride) {
        out[j] = func(data[j]);
    }
}

template<typename F, typename T = DType>
inline MOBULA_DEVICE void mobula_reduce(F func, const T *data, const int n, const int stride = 1, T *out = nullptr) {
    if (out == nullptr) out = const_cast<T*>(data);
    T &val = out[0];
    val = data[0];
    for (int i = 1, j = stride; i < n; ++i, j += stride) {
        val = func(val, data[j]);
    }
}

inline MOBULA_DEVICE int get_middle_loop_offset(const int i, const int middle_size, const int inner_size) {
    // &a[outer_size][0][inner_size] = &a[j]
    const int inner_i = i % inner_size;
    const int outer_i = i / inner_size;
    return outer_i * middle_size * inner_size + inner_i; // j
}

template<typename T = DType>
MOBULA_DEVICE T ADD_FUNC(const T &a, const T &b) {
    return a + b;
}
template<typename T = DType>
MOBULA_DEVICE T MAX_FUNC(const T &a, const T &b) {
    return a > b ? a : b;
}

} // namespace mobula

#endif
