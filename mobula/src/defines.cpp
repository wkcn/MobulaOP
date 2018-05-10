#include "defines.h"

namespace mobula{

#if not USING_CUDA
#if not USING_OPENMP
map<thread::id, pair<int, int> > MOBULA_KERNEL_INFOS;
mutex MOBULA_KERNEL_MUTEX;

mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
#endif // USING_OPENMP endif
#endif // USING_CUDA endif

};

#if USING_CUDA
void set_device(const int device_id) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));
    if (current_device != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
    }
}
#else // USING_CUDA else
void set_device(const int device_id) {
    throw "Doesn't support setting device on CPU mode";
}
#endif // USING_CUDA endif
