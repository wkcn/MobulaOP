#ifndef _IM2COL_H_
#define _IM2COL_H_

#include "defines.h"

extern "C" {
using namespace mobula;

void im2col(const DType *data_im, const int channels,
            const int height, const int width,
            const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            DType *data_col);

void col2im(const DType *data_col, const int channels,
            const int height, const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            DType *data_im);
 
}

#endif
