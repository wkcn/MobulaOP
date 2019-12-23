/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

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

#include "dlpack/dlpack.h"

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

// NDArray
class MXNDArray {};
typedef MXNDArray* NDArrayHandle;

/*!
 * \brief Union type of values
 *  being passed through API and function calls.
 */
typedef union {
  int64_t v_int64;
  double v_float64;
  void* v_handle;
  const char* v_str;
  // TVMType v_type;
  // TVMContext v_ctx;
} TVMValue;

/*!
 * \brief The type code in TVMType
 * \note TVMType is used in two places.
 */
typedef enum {
  // The type code of other types are compatible with DLPack.
  // The next few fields are extension types
  // that is used by TVM API calls.
  kHandle = 3U,
  kNull = 4U,
  kTVMType = 5U,
  kTVMContext = 6U,
  kArrayHandle = 7U,
  kNodeHandle = 8U,
  kModuleHandle = 9U,
  kFuncHandle = 10U,
  kStr = 11U,
  kBytes = 12U,
  kNDArrayContainer = 13U,
  kObjectCell = 14U,
  // Extension codes for other frameworks to integrate TVM PackedFunc.
  // To make sure each framework's id do not conflict, use first and
  // last sections to mark ranges.
  // Open an issue at the repo if you need a section of code.
  kExtBegin = 15U,
  kNNVMFirst = 16U,
  kNNVMLast = 20U,
  // The following section of code is used for non-reserved types.
  kExtReserveEnd = 64U,
  kExtEnd = 128U,
  // The rest of the space is used for custom, user-supplied datatypes
  kCustomBegin = 129U,
} TVMTypeCode;
// Pick code 19 for MXNet NDArray
constexpr const int kTVMNDArrayTypeCode = 19;

namespace runtime {
#define TVM_CHECK_TYPE_CODE(a, b)

class TVMRetValue {
 public:
  TVMValue value_;
  int type_code_;
  TVMRetValue() : type_code_(kNull) {}
};

class TVMArgsSetter;
/*!
 * \brief Internal base class to
 *  handle conversion to POD values.
 */
class TVMPODValue_ {
 public:
  operator double() const {
    // Allow automatic conversion from int to float
    // This avoids errors when user pass in int from
    // the frontend while the API expects a float.
    if (type_code_ == kDLInt) {
      return static_cast<double>(value_.v_int64);
    }
    TVM_CHECK_TYPE_CODE(type_code_, kDLFloat);
    return value_.v_float64;
  }
  operator int64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator uint64_t() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64;
  }
  operator int() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    CHECK_LE(value_.v_int64, std::numeric_limits<int>::max());
    return static_cast<int>(value_.v_int64);
  }
  operator bool() const {
    TVM_CHECK_TYPE_CODE(type_code_, kDLInt);
    return value_.v_int64 != 0;
  }
  operator void*() const {
    if (type_code_ == kNull) return nullptr;
    if (type_code_ == kArrayHandle) return value_.v_handle;
    TVM_CHECK_TYPE_CODE(type_code_, kHandle);
    return value_.v_handle;
  }
  operator DLTensor*() const {
    if (type_code_ == kArrayHandle || type_code_ == kNDArrayContainer) {
      return static_cast<DLTensor*>(value_.v_handle);
    } else {
      if (type_code_ == kNull) return nullptr;
      /*
      LOG(FATAL) << "Expected "
                 << "DLTensor* or NDArray but get "
                 << TypeCode2Str(type_code_);
      */
      return nullptr;
    }
  }
  int type_code() const { return type_code_; }
  /*!
   * \brief return handle as specific pointer type.
   * \tparam T the data type.
   * \return The pointer type.
   */
  template <typename T>
  T* ptr() const {
    return static_cast<T*>(value_.v_handle);
  }

 protected:
  friend class TVMArgsSetter;
  friend class TVMRetValue;
  TVMPODValue_() : type_code_(kNull) {}
  TVMPODValue_(TVMValue _value, int _type_code)
      : value_(_value), type_code_(_type_code) {}

  /*! \brief The value */
  TVMValue value_;
  /*! \brief the type code */
  int type_code_;
};

class TVMArgValue;
/*! \brief Arguments into TVM functions. */
class TVMArgs {
 public:
  const TVMValue* values;
  const int* type_codes;
  int num_args;
  /*!
   * \brief constructor
   * \param values The argument values
   * \param type_codes The argument type codes
   * \param num_args number of arguments.
   */
  TVMArgs(const TVMValue* values_, const int* type_codes_, int num_args_)
      : values(values_), type_codes(type_codes_), num_args(num_args_) {}
  /*! \return size of the arguments */
  inline int size() const { return num_args; }
  /*!
   * \brief Get i-th argument
   * \param i the index.
   * \return the ith argument.
   */
  inline TVMArgValue operator[](int i) const;
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
  inline TVMRetValue operator()(Args&&... args) const;

 private:
  FType body_;
};

/*!
 * \brief A single argument value to PackedFunc.
 *  Containing both type_code and TVMValue
 *
 *  Provides utilities to do type cast into other types.
 */
class TVMArgValue : public TVMPODValue_ {
 public:
  /*! \brief default constructor */
  TVMArgValue() {}
  /*!
   * \brief constructor
   * \param value of the function
   * \param type_code The type code.
   */
  TVMArgValue(TVMValue _value, int _type_code)
      : TVMPODValue_(_value, _type_code) {}
  // reuse converter from parent
  using TVMPODValue_::operator double;
  using TVMPODValue_::operator int64_t;
  using TVMPODValue_::operator uint64_t;
  using TVMPODValue_::operator int;
  using TVMPODValue_::operator bool;
  using TVMPODValue_::operator void*;
  using TVMPODValue_::operator DLTensor*;
  operator PackedFunc() const {
    if (type_code_ == kNull) return PackedFunc();
    TVM_CHECK_TYPE_CODE(type_code_, kFuncHandle);
    return *ptr<PackedFunc>();
  }
  const TVMValue& value() const { return value_; }
};

inline TVMArgValue TVMArgs::operator[](int i) const {
  return TVMArgValue(values[i], type_codes[i]);
}

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
  void operator()(size_t i, NDArrayHandle value) const {
    values_[i].v_handle = value;
    type_codes_[i] = kTVMNDArrayTypeCode;
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

 private:
  /*! \brief The values fields */
  TVMValue* values_;
  /*! \brief The type code fields */
  int* type_codes_;
};

template <typename... Args>
inline TVMRetValue PackedFunc::operator()(Args&&... args) const {
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

}  // namespace runtime

}  // namespace tvm

#endif
