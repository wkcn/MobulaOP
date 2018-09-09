#include "context/context.h"

namespace mobula{

}

#if USING_CUDA
void set_device(const int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}
#else // USING_CUDA else
void set_device(const int /*device_id*/) {
    throw "Doesn't support setting device on CPU mode";
}
#endif // USING_CUDA endif
