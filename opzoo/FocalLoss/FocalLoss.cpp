/*
 * author: kohill
 */
#include "mobula_op.h"

#if USING_CUDA
#include<cuda.h>
#else
#define __device__
#include <cmath>
#include <algorithm>
using std::exp;
using std::max;
using std::pow;
using std::log;
#endif

namespace mobula {

#define UNUSED(expr) do { (void)(expr); } while (0)

template <typename T>
__device__ inline T sigmoid(T x){
	T max_val = max(static_cast<T>(0), -1 * x);
	T v0 = exp(0-max_val);
	return v0 / (v0 + exp(-x - max_val));
}

template <typename T>
__device__ inline T log_sigmoid(T x){
	T max_val = max(static_cast<T>(0), -1 * x);
	return -1 * max_val - log(exp(0-max_val) + exp(-x - max_val));
}

template <typename T>
MOBULA_KERNEL focal_loss_forward_kernel(const int out_size, T alpha, T gamma, T* logits, T* targets, T* outputs) {
	parfor(out_size, [&](int index){
		T y = targets[index];
		T x = logits[index];
		T sigmoid_x = sigmoid(x);
		T sigmoid_neg_x = 1 - sigmoid_x;
		T output = alpha * y * pow(sigmoid_neg_x, gamma) * log_sigmoid(x);
		output += (1 - alpha) * (1 - y) * log_sigmoid(-x) * pow(sigmoid_x, gamma);
		output *= -1;
		outputs[index] = output;
	});
} // focal_loss_forward_kernel

template <typename T>
MOBULA_KERNEL focal_loss_backward_kernel(const int out_size, T alpha, T gamma, T* logits, T* targets, T* outputs) {
	parfor(out_size, [&](int index){
		T y = targets[index];
		T x = logits[index];
		T sigmoid_x = sigmoid(x);
		T sigmoid_neg_x = 1 - sigmoid_x;
		T output = (alpha- 1 - alpha * y) * pow(sigmoid_x, 1 + gamma);
		output += alpha * y * pow(sigmoid_neg_x, gamma + 1);
		output += (alpha - 1) * gamma * (y - 1) * sigmoid_neg_x * pow(sigmoid_x, gamma) * log_sigmoid(-x);
		output -= alpha * gamma * sigmoid_x * y * pow(sigmoid_neg_x, gamma)  * log_sigmoid(x);
		output += sigmoid_x * y * pow(sigmoid_x, gamma);
		output *= -1;
 		outputs[index] = output;
	});
} // focal_loss_backward_kernel


}  // namespace mobula
