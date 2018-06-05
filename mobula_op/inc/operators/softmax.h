#ifndef _MOBULA_SOFTMAX_
#define _MOBULA_SOFTMAX_

#include "defines.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL SoftmaxForward(
    const int nthreads,
    const T *data,
    const int num_classes,
    const int inner_size,
    T *probs);

template <typename T>
MOBULA_KERNEL SoftmaxLossForward(
    const int nthreads,
    const T *probs,
    const T *labels,
    const int num_classes,
    const int inner_size,
    T *losses);

}

extern "C" {
using namespace mobula;

void softmax_forward(
    const DType *data,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *probs);


void softmax_loss_forward(
    const DType *probs,
    const DType *labels,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *losses);


void softmax_loss_backward(
    const DType *probs,
    const DType *labels,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *dX);

}

#endif
