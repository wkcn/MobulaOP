#ifndef _MOBULA_FUNC_
#define _MOBULA_FUNC_

#include "defines.h"

extern "C" {
using namespace mobula;

void add(const int n, const DType *a, const DType *b, DType *out);
void sub(const int n, const DType *a, const DType *b, DType *out);

}

#endif
