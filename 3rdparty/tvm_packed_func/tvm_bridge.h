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

/*!
 * \file tvm_bridge.h
 * \brief Bridge to run TVM's PackedFunc in MXNet's async engine.
 *
 *  This bridge is mainly used to expose MXNet's async engine push to
 *  TVM. It only uses TVM runtime in aheader only mode, which means
 *  there is no link dependencies.
 *
 *  Support for TVM is optional even when this code
 *  is always compiled and built with the project.
 *  We choose this strategy because we do not yet want
 *  llvm as dependency(which TVM uses). So instead we expose hook
 *  to TVM and let user use this feature when they have TVM installed.
 *
 *  We do require TVM and MXNet to be built with same C++ ABI of std::function
 *
 *  Adapted from
 * https://github.com/apache/incubator-mxnet/blob/master/src/nnvm/tvm_bridge.cc
 */
#pragma once

#include <iostream>

#include "tvm_packed_func.h"

namespace tvm_bridge {
using namespace tvm;
using namespace tvm::runtime;

using TVMFunc = std::function<void(TVMArgs args, TVMRetValue* rv)>;

typedef DLManagedTensor* DLManagedTensorHandle;
/*! \brief handle to Context */
typedef const void* ContextHandle;
/*! \brief handle to Engine FnProperty */
typedef const void* EngineFnPropertyHandle;
/*! \brief handle to Engine VarHandle */
typedef void* EngineVarHandle;

/*! \brief Engine asynchronous operation */
typedef void (*EngineAsyncFunc)(void*, void*, void*);
/*! \brief Engine synchronous operation */
typedef void (*EngineSyncFunc)(void*, void*);
/*! \brief Callback to free the param for EngineAsyncFunc/EngineSyncFunc */
typedef void (*EngineFuncParamDeleter)(void*);

int (*MXShallowCopyNDArray)(NDArrayHandle src_handle, NDArrayHandle* out);
int (*MXNDArrayFree)(NDArrayHandle handle);
int (*MXNDArrayGetContext)(NDArrayHandle handle, int* out_dev_type,
                           int* out_dev_id);
int (*MXNDArrayToDLPack)(NDArrayHandle handle, DLManagedTensorHandle*);
int (*MXEnginePushSyncND)(
    EngineSyncFunc sync_func, void* func_param, EngineFuncParamDeleter deleter,
    ContextHandle ctx_handle, NDArrayHandle* const_nds_handle,
    int num_const_nds, NDArrayHandle* mutable_nds_handle, int num_mutable_nds,
    EngineFnPropertyHandle prop_handle, int priority, const char* opr_name);

struct Context {
  enum DeviceType { kCPU = 1 << 0, kGPU = 1 << 1, kCPUPinned = 3 };
  DeviceType dev_type;
  int32_t dev_id;
};

struct RunContext {
  Context ctx;
  void* stream;
};

thread_local int DEV_ID;
thread_local void* STRM;

void set_stream(TVMArgs args, TVMRetValue*) {
  DEV_ID = args.values[1].v_int64;
  STRM = args.values[2].v_handle;
}

/*!
 * \brief Async functor object
 *  calling argument of the function.
 */
class TVMFunctor {
 public:
  // constructor
  explicit TVMFunctor(PackedFunc func, PackedFunc fset_stream)
      : func_(func), fset_stream_(fset_stream) {}

  void Init(const TVMArgs& args, const std::vector<int>& const_loc,
            std::vector<NDArrayHandle>* const_nds,
            std::vector<NDArrayHandle>* mutate_nds) {
    values_.clear();
    type_codes_.clear();
    values_.insert(values_.end(), args.values, args.values + args.size());
    type_codes_.insert(type_codes_.end(), args.type_codes,
                       args.type_codes + args.size());

    size_t const_loc_ptr = 0;
    int dev_type, dev_id;
    ctx_.dev_id = -1;
    for (int i = 0; i < args.size(); ++i) {
      if (args.type_codes[i] == kTVMNDArrayTypeCode) {
        NDArrayHandle nd_handle =
            static_cast<NDArrayHandle>(args.values[i].v_handle);
        NDArrayHandle nd;
        MXShallowCopyNDArray(nd_handle, &nd);
        type_codes_[i] = kArrayHandle;
        array_handle_.push_back(nd);
        MXNDArrayGetContext(nd, &dev_type, &dev_id);
        if (ctx_.dev_id != -1) {
          if (dev_type != ctx_.dev_type || dev_id != ctx_.dev_id) {
            std::cout << "Inconsistent context: source(" << int(ctx_.dev_type)
                      << ":" << ctx_.dev_id << ") vs target: (" << dev_type
                      << ":" << dev_id << ")" << std::endl;
            exit(-1);
          }
        }
        ctx_ = {Context::DeviceType(dev_type), dev_id};
        array_loc_.push_back(i);
        // check if there is read or mutate
        // by default assume we mutate the array.
        if (const_loc_ptr < const_loc.size() && i == const_loc[const_loc_ptr]) {
          const_nds->push_back(nd);
          ++const_loc_ptr;
        } else {
          mutate_nds->push_back(nd);
        }
      } else {
        CHECK_LT(args.type_codes[i], int(kTVMType))
            << "Only allow POD type in mxnet async call";
      }
    }
  }

  void Run(const RunContext& rctx) {
    // setup DLTensor
    std::vector<DLManagedTensorHandle> dlms(array_loc_.size());
    for (size_t i = 0; i < array_loc_.size(); ++i) {
      DLManagedTensorHandle& dlm = dlms[i];
      MXNDArrayToDLPack(array_handle_[i], &dlm);
      values_[array_loc_[i]].v_handle = static_cast<void*>(&dlm->dl_tensor);
    }
    // run the packed function
    TVMRetValue rv;
    TVMArgs args(&values_[0], &type_codes_[0], values_.size());
    if (ctx_.dev_type == Context::kGPU) {
      // pass stream via last argument.
      void* strm = reinterpret_cast<void**>(rctx.stream)[0];
      int dev_type = kDLGPU;
      fset_stream_(dev_type, rctx.ctx.dev_id, strm);
      func_.CallPacked(args, &rv);
      fset_stream_(dev_type, rctx.ctx.dev_id, nullptr);
    } else {
      func_.CallPacked(args, &rv);
    }
    for (DLManagedTensorHandle dlm : dlms) {
      dlm->deleter(dlm);
    }
  }

  inline const Context& ctx() { return ctx_; }

  ~TVMFunctor() {
    for (NDArrayHandle handle : array_handle_) {
      MXNDArrayFree(handle);
    }
  }

 private:
  /*! \brief The function */
  PackedFunc func_;
  /*! \brief Set stream */
  PackedFunc fset_stream_;
  /*! \brief Values field */
  std::vector<TVMValue> values_;
  /*! \brief type code field */
  std::vector<int> type_codes_;
  /*! \brief NDArrayHandles field */
  std::vector<NDArrayHandle> array_handle_;
  /*! \brief position of array in arguments */
  std::vector<int> array_loc_;
  /*! \brief context */
  Context ctx_;
};

inline void DeduplicateNDArrayHandle(std::vector<NDArrayHandle>* read_nds,
                                     std::vector<NDArrayHandle>* write_nds) {
  std::sort(write_nds->begin(), write_nds->end());
  write_nds->resize(std::unique(write_nds->begin(), write_nds->end()) -
                    write_nds->begin());
  std::sort(read_nds->begin(), read_nds->end());
  read_nds->resize(std::unique(read_nds->begin(), read_nds->end()) -
                   read_nds->begin());
  auto wit = write_nds->begin();
  auto rtop = read_nds->begin();
  for (auto rit = read_nds->begin(); rit != read_nds->end(); ++rit) {
    while (wit != write_nds->end() && *wit < *rit) ++wit;
    if (wit == write_nds->end() || *wit != *rit) {
      *rtop = *rit;
      ++rtop;
    }
  }
  read_nds->resize(rtop - read_nds->begin());
}

struct SyncFuncParams {
  Context ctx;
  std::shared_ptr<TVMFunctor> func;
};

void sync_func_inst(void* rctx, void* param) {
  SyncFuncParams* ps = static_cast<SyncFuncParams*>(param);
  const RunContext* pctx = static_cast<RunContext*>(rctx);
  ps->func->Run(*pctx);
}

void deleter_inst(void* param) {
  SyncFuncParams* ps = static_cast<SyncFuncParams*>(param);
  delete ps;
}

PackedFunc WrapAsyncCall(TVMArgs wrap_args) {
  PackedFunc f = wrap_args[0];
  PackedFunc fset_stream = wrap_args[1];
  int num_const = wrap_args[2];

  // sorted position of constant arguments
  std::vector<int> const_loc;
  for (int i = 0; i < num_const; ++i) {
    const_loc.push_back(wrap_args[i + 3].operator int());
  }
  std::sort(const_loc.begin(), const_loc.end());
  // wrapped function
  // This is the function that called by the user.
  auto wrapped = [f, fset_stream, const_loc](TVMArgs args,
                                             TVMRetValue* /*rv*/) {
    std::shared_ptr<TVMFunctor> func =
        std::make_shared<TVMFunctor>(f, fset_stream);
    std::vector<NDArrayHandle> const_nds, mutate_nds;
    func->Init(args, const_loc, &const_nds, &mutate_nds);
    DeduplicateNDArrayHandle(&const_nds, &mutate_nds);
    SyncFuncParams* ps = new SyncFuncParams();
    ps->ctx = func->ctx();
    ps->func = func;
    NDArrayHandle* const_nds_handle = const_nds.data();
    int num_const_nds = const_nds.size();
    NDArrayHandle* mutate_nds_handle = mutate_nds.data();
    int num_mutate_nds = mutate_nds.size();

    MXEnginePushSyncND(sync_func_inst, static_cast<void*>(ps), deleter_inst,
                       &ps->ctx, const_nds_handle, num_const_nds,
                       mutate_nds_handle, num_mutate_nds, nullptr, 0, nullptr);
  };
  return PackedFunc(wrapped);
}

}  // namespace tvm_bridge
