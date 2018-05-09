#include "kernels.h"

namespace mobula{

MOBULA_KERNEL add_kernel(const int n, const float *a, const float *b, float *out){
	KERNEL_LOOP(i, n) {
		out[i] = a[i] + b[i];
	}
}

};
