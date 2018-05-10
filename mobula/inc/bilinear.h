/*
    Adapted from caffe2: github.com/caffe2/caffe2
*/
#ifndef _MOBULA_BILINEAR_
#define _MOBULA_BILINEAR_
#include "defines.h"

namespace mobula {

template <typename T>
MOBULA_DEVICE T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index);

template <typename T>
MOBULA_DEVICE void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index);

}

#endif
