#ifndef MOBULA_INC_CONTEXT_CONTEXT_H_
#define MOBULA_INC_CONTEXT_CONTEXT_H_

#include <cstring>

#include "../api.h"

#define MOBULA_FUNC
#if USING_HIP || USING_CUDA
#include "./hip_ctx.h"
#else
#include "./cpu_ctx.h"
#endif  // USING_CUDA

// C API
extern "C" {
MOBULA_DLL void set_device(const int device_id);
}

#endif  // MOBULA_INC_CONTEXT_CONTEXT_H_
