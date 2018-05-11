#include "ops.h"
#include "kernels.h"

void add(const int n, const DType *a, const DType *b, DType *out){
	KERNEL_RUN(add_kernel<DType>, n)(n, a, b, out);
}
