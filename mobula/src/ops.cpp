#include "ops.h"
#include "kernels.h"

namespace mobula{

void add(const int n, const float *a, const float *b, float *out){
	KERNEL_RUN(add_kernel, n)(n, a, b, out);
}

};
