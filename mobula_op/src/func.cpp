#include "func.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL add_kernel(const int n, const T *a, const T *b, T *out){
	KERNEL_LOOP(i, n) {
		out[i] = a[i] + b[i];
	}
}

template <typename T>
MOBULA_KERNEL sub_kernel(const int n, const T *a, const T *b, T *out){
	parfor(n, [&](int i){
		out[i] = a[i] - b[i];
	});
}

}

void add(const int n, const DType *a, const DType *b, DType *out){
	KERNEL_RUN(add_kernel<DType>, n)(n, a, b, out);
}

void sub(const int n, const DType *a, const DType *b, DType *out){
	KERNEL_RUN(sub_kernel<DType>, n)(n, a, b, out);
}

