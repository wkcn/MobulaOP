/*
 * author: kohill
 * github: https://github.com/kohillyang
 */
#include "mobula_op.h"

#if USING_CUDA
#include <cuda.h>
#else
#include <algorithm>
#include <cmath>
using std::exp;
using std::log;
using std::max;
using std::min;
using std::pow;
#endif  // USING_CUDA

namespace mobula {

template <typename T>
MOBULA_KERNEL iou_loss_forward_kernel(const int out_size, const T* preds,
                                      const T* targets, T* outputs) {
  parfor(out_size, [&](int index) {
    const int l = 0, t = 1, r = 2, b = 3;
    T targets_l = targets[index * 4 + l];
    T targets_t = targets[index * 4 + t];
    T targets_r = targets[index * 4 + r];
    T targets_b = targets[index * 4 + b];
    T preds_l = preds[index * 4 + l];
    T preds_t = preds[index * 4 + t];
    T preds_r = preds[index * 4 + r];
    T preds_b = preds[index * 4 + b];
    if (targets_l > 0 && targets_t > 0 && targets_r > 0 && targets_b > 0) {
      targets_l = log(targets_l);
      targets_t = log(targets_t);
      targets_r = log(targets_r);
      targets_b = log(targets_b);
      T tl = targets_t + targets_l;
      T tr = targets_t + targets_r;
      T bl = targets_b + targets_l;
      T br = targets_b + targets_r;
      T tl_hat = preds_t + preds_l;
      T tr_hat = preds_t + preds_r;
      T bl_hat = preds_b + preds_l;
      T br_hat = preds_b + preds_r;
      T x_t_i = min(targets_t, preds_t);
      T x_b_i = min(targets_b, preds_b);
      T x_l_i = min(targets_l, preds_l);
      T x_r_i = min(targets_r, preds_r);
      T tl_i = x_t_i + x_l_i;
      T tr_i = x_t_i + x_r_i;
      T bl_i = x_b_i + x_l_i;
      T br_i = x_b_i + x_r_i;
      T max_v = tl;
      max_v = max(max_v, tr);
      max_v = max(max_v, bl);
      max_v = max(max_v, br);
      max_v = max(max_v, tl_hat);
      max_v = max(max_v, tr_hat);
      max_v = max(max_v, bl_hat);
      max_v = max(max_v, br_hat);
      max_v = max(max_v, tl_i);
      max_v = max(max_v, tr_i);
      max_v = max(max_v, bl_i);
      max_v = max(max_v, br_i);
      T I = exp(tl_i - max_v) + exp(tr_i - max_v) + exp(bl_i - max_v) +
            exp(br_i - max_v);
      T X =
          exp(tl - max_v) + exp(tr - max_v) + exp(bl - max_v) + exp(br - max_v);
      T X_hat = exp(tl_hat - max_v) + exp(tr_hat - max_v) +
                exp(bl_hat - max_v) + exp(br_hat - max_v);
      T I_over_U = I / (X + X_hat - I);
      T loss = log(I_over_U) * -1;
      outputs[index] = loss;
    }
  });
}  // iou_loss_forward_kernel

template <typename T>
MOBULA_KERNEL iou_loss_backward_kernel(const int out_size, const T* preds,
                                       const T* targets, T* outputs) {
  parfor(out_size, [&](int index) {
    const int l = 0, t = 1, r = 2, b = 3;
    T targets_l = targets[index * 4 + l];
    T targets_t = targets[index * 4 + t];
    T targets_r = targets[index * 4 + r];
    T targets_b = targets[index * 4 + b];
    T preds_l = preds[index * 4 + l];
    T preds_t = preds[index * 4 + t];
    T preds_r = preds[index * 4 + r];
    T preds_b = preds[index * 4 + b];
    if (targets_l > 0 && targets_t > 0 && targets_r > 0 && targets_b > 0) {
      targets_l = log(targets_l);
      targets_t = log(targets_t);
      targets_r = log(targets_r);
      targets_b = log(targets_b);
      T tl = targets_t + targets_l;
      T tr = targets_t + targets_r;
      T bl = targets_b + targets_l;
      T br = targets_b + targets_r;
      T tl_hat = preds_t + preds_l;
      T tr_hat = preds_t + preds_r;
      T bl_hat = preds_b + preds_l;
      T br_hat = preds_b + preds_r;
      T x_t_i = min(targets_t, preds_t);
      T x_b_i = min(targets_b, preds_b);
      T x_l_i = min(targets_l, preds_l);
      T x_r_i = min(targets_r, preds_r);
      T tl_i = x_t_i + x_l_i;
      T tr_i = x_t_i + x_r_i;
      T bl_i = x_b_i + x_l_i;
      T br_i = x_b_i + x_r_i;
      T max_v = tl;
      max_v = max(max_v, tr);
      max_v = max(max_v, bl);
      max_v = max(max_v, br);
      max_v = max(max_v, tl_hat);
      max_v = max(max_v, tr_hat);
      max_v = max(max_v, bl_hat);
      max_v = max(max_v, br_hat);
      max_v = max(max_v, tl_i);
      max_v = max(max_v, tr_i);
      max_v = max(max_v, bl_i);
      max_v = max(max_v, br_i);
      T I = exp(tl_i - max_v) + exp(tr_i - max_v) + exp(bl_i - max_v) +
            exp(br_i - max_v);
      T X =
          exp(tl - max_v) + exp(tr - max_v) + exp(bl - max_v) + exp(br - max_v);
      T X_hat = exp(tl_hat - max_v) + exp(tr_hat - max_v) +
                exp(bl_hat - max_v) + exp(br_hat - max_v);

      T partial_l;
      if (targets_l > preds_l) {
        T partial_item_1 = (exp(tl_i - max_v) + exp(bl_i - max_v)) / I;
        T partial_item_2 = exp(tl_hat - max_v) + exp(bl_hat - max_v) -
                           exp(tl_i - max_v) - exp(bl_i - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_l = partial_item_1 - partial_item_2;
      } else {
        T partial_item_1 = 0;
        T partial_item_2 = exp(tl_hat - max_v) + exp(bl_hat - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_l = partial_item_1 - partial_item_2;
      }
      T partial_t;
      if (targets_t > preds_t) {
        T partial_item_1 = (exp(tl_i - max_v) + exp(tr_i - max_v)) / I;
        T partial_item_2 = exp(tl_hat - max_v) + exp(tr_hat - max_v) -
                           exp(tl_i - max_v) - exp(tr_i - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_t = partial_item_1 - partial_item_2;
      } else {
        T partial_item_1 = 0;
        T partial_item_2 = exp(tl_hat - max_v) + exp(tr_hat - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_t = partial_item_1 - partial_item_2;
      }

      T partial_r;
      if (targets_r > preds_r) {
        T partial_item_1 = (exp(tr_i - max_v) + exp(br_i - max_v)) / I;
        T partial_item_2 = exp(tr_hat - max_v) + exp(br_hat - max_v) -
                           exp(tr_i - max_v) - exp(br_i - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_r = partial_item_1 - partial_item_2;
      } else {
        T partial_item_1 = 0;
        T partial_item_2 = exp(tr_hat - max_v) + exp(br_hat - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_r = partial_item_1 - partial_item_2;
      }

      T partial_b;
      if (targets_b > preds_b) {
        T partial_item_1 = (exp(bl_i - max_v) + exp(br_i - max_v)) / I;
        T partial_item_2 = exp(bl_hat - max_v) + exp(br_hat - max_v) -
                           exp(bl_i - max_v) - exp(br_i - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_b = partial_item_1 - partial_item_2;
      } else {
        T partial_item_1 = 0;
        T partial_item_2 = exp(bl_hat - max_v) + exp(br_hat - max_v);
        partial_item_2 /= (X + X_hat - I);
        partial_b = partial_item_1 - partial_item_2;
      }
      outputs[index * 4 + l] = -1 * partial_l;
      outputs[index * 4 + t] = -1 * partial_t;
      outputs[index * 4 + r] = -1 * partial_r;
      outputs[index * 4 + b] = -1 * partial_b;
    }
  });
}  // iou_loss_backward_kernel

}  // namespace mobula
