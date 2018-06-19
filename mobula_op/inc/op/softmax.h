#ifndef _MOBULA_SOFTMAX_
#define _MOBULA_SOFTMAX_

#include "defines.h"

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
