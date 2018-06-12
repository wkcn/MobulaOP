#ifndef _NAIVE_CTX_
#define _NAIVE_CTX_

#include <vector>
#include <map>
#include <utility>
#include <thread>
#include <mutex>
#include <future>

namespace mobula {

#define KERNEL_RUN(a, n) a

#if HOST_NUM_THREADS > 1

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    const int nthreads = std::min(n, HOST_NUM_THREADS);
    std::vector<std::future<void> > futures;
    const int step = (n + nthreads - 1) / nthreads;
    int blockBegin = 0;
    int blockEnd;
    for (int t = 0; t < nthreads; ++t) {
        blockEnd = std::min(blockBegin + step, n);
        futures.push_back(std::move(std::async(std::launch::async, [blockBegin, blockEnd, &F]{
            for (int i = blockBegin; i < blockEnd; ++i) {
                F(i);
            }
        })));
        blockBegin = blockEnd;
    }
    for (auto &f : futures) f.wait();
}

#else // HOST_NUM_THREADS > 1 else

template <typename Func>
MOBULA_DEVICE void parfor(const int n, Func F) {
    for (int i = 0; i < n; ++i) {
        F(i);
    }
}

#endif // HOST_NUM_THREADS > 1

} // namespace mobula

#endif
