#ifndef _CONTEXT_H_
#define _CONTEXT_H_

#if USING_CUDA
#include "context/cuda_ctx.h"
#else
#include "context/cpu_ctx.h"
#endif // USING_CUDA

// C API
extern "C" {

void set_device(const int device_id);

}

#endif
