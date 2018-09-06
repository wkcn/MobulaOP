/*
    Adapted from caffe2: github.com/caffe2/caffe2
*/
#ifndef _MOBULA_ROI_ALIGN_
#define _MOBULA_ROI_ALIGN_

#include "defines.h"

extern "C" {
using namespace mobula;

void roi_align_forward(
    const int nthreads,
    const DType* bottom_data,
    const DType spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const DType* bottom_rois,
    DType* top_data);

void roi_align_backward(
    const int nthreads,
    const DType* top_diff,
    const int num_rois,
    const DType spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    DType* bottom_diff,
    const float* bottom_rois);

}

#endif
