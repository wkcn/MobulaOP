#ifndef _CPU_ENGINE_
#define _CPU_ENGINE_

namespace mobula {

#include <cmath>
#include <thread>
#include <mutex>

#define MOBULA_KERNEL void
#define MOBULA_DEVICE

using std::max;
using std::min;

constexpr int NUM_MOBULA_ATOMIC_ADD_MUTEXES = HOST_NUM_THREADS * 8;
extern std::mutex MOBULA_ATOMIC_ADD_MUTEXES[NUM_MOBULA_ATOMIC_ADD_MUTEXES];
inline MOBULA_DEVICE float atomic_add(const float val, float* address) {
    long id = (reinterpret_cast<long>(address) / sizeof(float)) % NUM_MOBULA_ATOMIC_ADD_MUTEXES;
    MOBULA_ATOMIC_ADD_MUTEXES[id].lock();
    *address += val;
    MOBULA_ATOMIC_ADD_MUTEXES[id].unlock();
    return *address;
}

}

#if USING_OPENMP
#include "openmp_engine.h"
#else
#include "naive_engine.h"
#endif


#endif
