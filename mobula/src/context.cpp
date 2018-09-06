#include "context/context.h"

namespace mobula{

#if not USING_CUDA and HOST_NUM_THREADS > 1
std::mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
#if not USING_OPENMP
std::map<std::thread::id, std::pair<int, int> > MOBULA_KERNEL_INFOS;
std::mutex MOBULA_KERNEL_MUTEX;

#endif // USING_OPENMP endif
#endif // USING_CUDA endif

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
