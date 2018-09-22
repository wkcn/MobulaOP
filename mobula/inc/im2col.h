#ifndef MOBULA_INC_IM2COL_H_
#define MOBULA_INC_IM2COL_H_

#include "./api.h"

extern "C" {

/*
 * data_im: (channels, height, width)
 * data_col: (channels, kernel_h, kernel_w, height_col, width_col)
 */
typedef float DType;
MOBULA_DLL void im2col(const DType *data_im, const int channels,
                       const int height, const int width, const int kernel_h,
                       const int kernel_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       DType *data_col);

/*
 * data_col: (channels, kernel_h, kernel_w, height_col, width_col)
 * data_im: (channels, height, width)
 */

typedef float DType;
MOBULA_DLL void col2im(const DType *data_col, const int channels,
                       const int height, const int width, const int kernel_h,
                       const int kernel_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       DType *data_im);
}

#endif  // MOBULA_INC_IM2COL_H_
