#ifndef _MOBULA_SOFTMAX_LOSS_
#define _MOBULA_SOFTMAX_LOSS_

#include "defines.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL SoftmaxLossForward(
    const int nthreads,
    const T *data,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    T *probs);

}

extern "C" {
using namespace mobula;

void softmax_loss_forward(
    const DType *data,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *probs);

}

#endif
