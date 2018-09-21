#ifndef MOBULA_INC_CONTEXT_CONTEXT_H_
#define MOBULA_INC_CONTEXT_CONTEXT_H_

#include <cstring>

#include "./api.h"

#if USING_CUDA
#include "context/cuda_ctx.h"
#elif USING_HIP
#include "context/hip_ctx.h"
#else
#include "context/cpu_ctx.h"
#endif  // USING_CUDA

// C API
extern "C" {
MOBULA_DLL void set_device(const int device_id);
}

#endif  // MOBULA_INC_CONTEXT_CONTEXT_H_
