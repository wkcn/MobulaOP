/*
 * Adapted from MXNet: github.com/apache/incubator-mxnet
 */

#include "im2col.h"

#include "defines.h"

namespace mobula {

inline MOBULA_DEVICE bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
MOBULA_KERNEL im2col_kernel(const int n, const T* data_im, const int height,
                            const int width, const int kernel_h,
                            const int kernel_w, const int pad_h,
                            const int pad_w, const int stride_h,
                            const int stride_w, const int dilation_h,
                            const int dilation_w, const int height_col,
                            const int width_col, T* data_col) {
  const int channel_size = height * width;
  // (channels, height_col, width_col)
  parfor(n, [&](int index) {
    /*
    const int channel = index / (kernel_h * kernel_w * height_col * width_col);
    const int kernel_row = (index / (kernel_w * height_col * width_col)) %
    kernel_h; const int kernel_col = (index / (height_col * width_col)) %
    kernel_w; const int output_row = (index / width_col) % height_col; const int
    output_col = index % width_col;
    */
    int tmp_index = index;
    const int output_col = tmp_index % width_col;
    tmp_index /= width_col;
    const int output_row = tmp_index % height_col;
    tmp_index /= height_col;
    const int kernel_col = tmp_index % kernel_w;
    tmp_index /= kernel_w;
    const int kernel_row = tmp_index % kernel_h;
    tmp_index /= kernel_h;
    const int channel = tmp_index;

    const int input_row =
        -pad_h + kernel_row * dilation_h + stride_h * output_row;
    const int input_col =
        -pad_w + kernel_col * dilation_w + stride_w * output_col;
    data_col[index] =
        (is_a_ge_zero_and_a_lt_b(input_row, height) &&
         is_a_ge_zero_and_a_lt_b(input_col, width))
            ? data_im[channel * channel_size + input_row * width + input_col]
            : static_cast<T>(0);
  });
}

template <typename T>
MOBULA_KERNEL col2im_kernel(const int n, const T* data_col,
                            const int /*channels*/, const int height,
                            const int width, const int kernel_h,
                            const int kernel_w, const int pad_h,
                            const int pad_w, const int stride_h,
                            const int stride_w, const int dilation_h,
                            const int dilation_w, const int height_col,
                            const int width_col, T* data_im) {
  parfor(n, [&](int index) {
    T val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    const int w_col_end = min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    const int h_col_end = min(h_im / stride_h + 1, height_col);
    // TODO(caffe): use LCM of stride and dilation to avoid unnecessary loops
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
          h_k /= dilation_h;
          w_k /= dilation_w;
          int data_col_index =
              (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col +
               h_col) *
                  width_col +
              w_col;
          val += data_col[data_col_index];
        }
      }
    }
    data_im[index] = val;
  });
}

typedef float DType;
void im2col(const DType* data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            DType* data_col) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * kernel_h * kernel_w * height_col * width_col;
  KERNEL_RUN(im2col_kernel<DType>)
  (num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w,
   stride_h, stride_w, dilation_h, dilation_w, height_col, width_col, data_col);
}

typedef float DType;
void col2im(const DType* data_col, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, const int dilation_h, const int dilation_w,
            DType* data_im) {
  int height_col =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int width_col =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  int num_kernels = channels * height * width;
  KERNEL_RUN(col2im_kernel<DType>)
  (num_kernels, data_col, channels, height, width, kernel_h, kernel_w, pad_h,
   pad_w, stride_h, stride_w, dilation_h, dilation_w, height_col, width_col,
   data_im);
}

}  // namespace mobula
