/*
    Adapted from caffe2: github.com/caffe2/caffe2
*/
#ifndef _MOBULA_ROI_ALIGN_
#define _MOBULA_ROI_ALIGN_

#include "defines.h"

namespace mobula {

template <typename T>
MOBULA_KERNEL RoIAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data);

template <typename T>
MOBULA_KERNEL RoIAlignBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois);

}

#endif
