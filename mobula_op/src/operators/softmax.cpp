#include "operators/softmax.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL SoftmaxForward(
    const int nthreads,
    const T *data,
    const int num_classes,
    const int inner_size,
    T *probs) {
    KERNEL_LOOP(index, nthreads) {
        int j = get_middle_loop_offset(index, num_classes, inner_size);
        const T *data_i = data + j;
        T *probs_i = probs + j;
        // get maximum
        T max_val;
        mobula_reduce(std::max<T>, data_i, num_classes, inner_size, &max_val);
        // exp(x - max(x))
        mobula_map([&max_val](const T &a){return std::exp(a - max_val);}, data_i, num_classes, inner_size, probs_i);
        // sum
        T sum_val;
        mobula_reduce([](const T &a, const T &b){return a + b;}, probs_i, num_classes, inner_size, &sum_val);
        // result
        mobula_map([&sum_val](const T &a){return a / sum_val;}, probs_i, num_classes, inner_size);
    }
}

template <typename T>
MOBULA_KERNEL SoftmaxLossForward(
    const int nthreads,
    const T *probs,
    const T *labels,
    const int num_classes,
    const int inner_size,
    T *losses) {
    KERNEL_LOOP(index, nthreads) {
        int j = get_middle_loop_offset(index, num_classes, inner_size);
        const int label = static_cast<int>(labels[index]);
        if (label >= 0)
            losses[index] = - log(probs[j + label * inner_size] + FLT_MIN);
        else
            losses[index] = 0;
    }
}

template <typename T>
MOBULA_KERNEL SoftmaxLossBackward(
    const int nthreads,
    const T *probs,
    const T *labels,
    const int num_classes,
    const int inner_size,
    const T grad_scale,
    T *dX) {

    KERNEL_LOOP(index, nthreads) {
        const int i = index / (num_classes * inner_size);
        const int j = (index / inner_size) % num_classes;
        const int k = index % inner_size;
        const int label = static_cast<int>(labels[i * inner_size + k]);
        if (label >= 0) {
            T grad = probs[index];
            if (label == j) --grad;
            dX[index] += grad * grad_scale;
        }
    }
}

} // namespace mobula

void softmax_forward(
    const DType *data,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *probs) {
    const int nthreads = outer_size * inner_size;
    KERNEL_RUN(SoftmaxForward<DType>, nthreads)(nthreads, data, num_classes, inner_size, probs);
}

void softmax_loss_forward(
    const DType *probs,
    const DType *labels,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    DType *losses) {
    const int nthreads = outer_size * inner_size;
    KERNEL_RUN(SoftmaxLossForward<DType>, nthreads)(nthreads, probs, labels,  num_classes, inner_size, losses);
}

void softmax_loss_backward(
    const DType *probs,
    const DType *labels,
    const int num_classes,
    const int outer_size,
    const int inner_size,
    const DType grad_scale,
    DType *dX) {
    const int nthreads = outer_size * num_classes * inner_size;
    KERNEL_RUN(SoftmaxLossBackward<DType>, nthreads)(nthreads, probs, labels,  num_classes, inner_size, grad_scale, dX);
}
