#include "context/context.h"

#include <stdexcept>

#include "logging.h"

namespace mobula {}

#if USING_HIP || USING_CUDA
void set_device(const int device_id) {
  int current_device;
  CHECK_HIP(hipGetDevice(&current_device));
  if (current_device != device_id) {
    CHECK_HIP(hipSetDevice(device_id));
  }
}
#else
void set_device(const int /*device_id*/) {
  LOG(FATAL) << "Doesn't support setting device on CPU mode";
}
#endif  // USING_HIP || USING_CUDA
