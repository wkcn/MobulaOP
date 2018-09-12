#ifndef _CONTEXT_H_
#define _CONTEXT_H_

#include <cstring>

#if USING_CUDA
#include "context/cuda_ctx.h"
#elif USING_HIP
#include "context/hip_ctx.h"
#else
#include "context/cpu_ctx.h"
#endif // USING_CUDA

namespace mobula {

} // namespace mobula

// C API
extern "C" {

void set_device(const int device_id);

}

#endif
