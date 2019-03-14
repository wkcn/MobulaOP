/*
 * Adapted from [TVM](https://github.com/dmlc/tvm)
 * License:
 * © Contributors Licensed under an Apache-2.0 license.
 * https://github.com/dmlc/tvm/blob/master/LICENSE
 */
#ifndef TVM_PACKED_FUNC_H_
#define TVM_PACKED_FUNC_H_

#include <functional>

#include <limits>
#include "dlpack.h"

namespace tvm {

// internal namespace
namespace detail {

template <bool stop, std::size_t I, typename F>
struct for_each_dispatcher {
  template <typename T, typename... Args>
  static void run(const F& f, T&& value, Args&&... args) {  // NOLINT(*)
    f(I, std::forward<T>(value));
    for_each_dispatcher<sizeof...(Args) == 0, (I + 1), F>::run(
        f, std::forward<Args>(args)...);
  }
};

template <std::size_t I, typename F>
struct for_each_dispatcher<true, I, F> {
  static void run(const F&) {}  // NOLINT(*)
};

template <typename F, typename... Args>
inline void for_each(const F& f, Args&&... args) {  // NOLINT(*)
  for_each_dispatcher<sizeof...(Args) == 0, 0, F>::run(
      f, std::forward<Args>(args)...);
}
}  // namespace detail

class NDArray {};
typedef NDArray* NDArrayHandle;

typedef enum {
  kHandle = 3U,
  kNull = 4U,
  kTVMType = 5U,
  kTVMContext = 6U,
  kArrayHandle = 7U,
  kFuncHandle = 10U,
  kStr = 11U,
  kTVMNDArrayTypeCode = 19U,
} TVMTypeCode;

typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  DLDataType v_type;
  DLContext v_ctx;
} TVMValue;

class TVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  TVMArgs(const TVMValue* _values, const int* _type_codes, int _num_args)
      : values(_values), type_codes(_type_codes), num_args(_num_args) {}
  inline int size() const { return num_args; }
};

class TVMRetValue {
 public:
  TVMValue value_;
  int type_code_;
  TVMRetValue() : type_code_(kNull) {}
};

class PackedFunc;

class TVMArgsSetter {
 public:
  TVMArgsSetter(TVMValue* values, int* type_codes)
      : values_(values), type_codes_(type_codes) {}
  // setters for POD types
  template <typename T, typename = typename std::enable_if<
                            std::is_integral<T>::value>::type>
  void operator()(size_t i, T value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, uint64_t value) const {
    values_[i].v_int64 = static_cast<int64_t>(value);
    // CHECK_LE(value,
    //         static_cast<uint64_t>(std::numeric_limits<int64_t>::max()));
    type_codes_[i] = kDLInt;
  }
  void operator()(size_t i, double value) const {
    values_[i].v_float64 = value;
    type_codes_[i] = kDLFloat;
  }
  void operator()(size_t i, std::nullptr_t value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kNull;
  }
  void operator()(size_t i, void* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kHandle;
  }
  void operator()(size_t i, DLTensor* value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kArrayHandle;
  }
  void operator()(size_t i, const char* value) const {
    values_[i].v_str = value;
    type_codes_[i] = kStr;
  }
  // setters for container type
  // They must be reference(instead of const ref)
  // to make sure they are alive in the tuple(instead of getting converted)
  void operator()(size_t i, const std::string& value) const {  // NOLINT(*)
    values_[i].v_str = value.c_str();
    type_codes_[i] = kStr;
  }
  void operator()(size_t i, const PackedFunc& value) const {  // NOLINT(*)
    values_[i].v_handle = const_cast<PackedFunc*>(&value);
    type_codes_[i] = kFuncHandle;
  }
  void operator()(size_t i, const NDArrayHandle value) const {  // NOLINT(*)
    values_[i].v_handle = static_cast<void*>(value);
    type_codes_[i] = kTVMNDArrayTypeCode;
  }

 private:
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

class PackedFunc {
 public:
  using FType = std::function<void(TVMArgs args, TVMRetValue* rv)>;
  PackedFunc(){};
  explicit PackedFunc(FType body) : body_(body) {}
  inline void CallPacked(TVMArgs args, TVMRetValue* rv) const {
    body_(args, rv);
  }
  template <typename... Args>
  inline TVMRetValue operator()(Args&&... args) const {
    const int kNumArgs = sizeof...(Args);
    const int kArraySize = kNumArgs > 0 ? kNumArgs : 1;
    TVMValue values[kArraySize];
    int type_codes[kArraySize];
    detail::for_each(TVMArgsSetter(values, type_codes),
                     std::forward<Args>(args)...);
    TVMRetValue rv;
    body_(TVMArgs(values, type_codes, kNumArgs), &rv);
    return rv;
  }

 private:
  FType body_;
};

}  // namespace tvm

#endif
