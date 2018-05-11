#include "kernels.h"
#include <iostream>
using namespace std;

namespace mobula{

template <typename T>
MOBULA_KERNEL add_kernel(const int n, const T *a, const T *b, T *out){
	KERNEL_LOOP(i, n) {
		out[i] = a[i] + b[i];
	}
}

template MOBULA_KERNEL add_kernel(const int n, const float *a, const float *b, float *out);

};
