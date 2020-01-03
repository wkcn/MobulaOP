// Adapted from
// https://github.com/apache/incubator-mxnet/blob/master/src/nnvm/tvm_bridge.cc
#ifndef MOBULA_INC_GLUE_MX_H_
#define MOBULA_INC_GLUE_MX_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "api.h"
#include "tvm_bridge.h"

extern "C" {
using namespace mobula;
using namespace tvm_bridge;

MOBULA_DLL PackedFunc* RegisterTVMFunc(const char*, TVMFunc pfunc,
                                       int num_const, int* const_loc) {
  PackedFunc func(pfunc);
  PackedFunc fset_stream(set_stream);
  const int num_args = 3 + num_const;
  std::vector<TVMValue> values(num_args);
  std::vector<int> type_codes(num_args);
  values[0].v_handle = &func;
  type_codes[0] = kFuncHandle;
  values[1].v_handle = &fset_stream;
  type_codes[1] = kFuncHandle;
  values[2].v_int64 = num_const;
  type_codes[2] = kDLInt;
  for (int i = 0; i < num_const; ++i) {
    values[i + 3].v_int64 = const_loc[i];
    type_codes[i + 3] = kDLInt;
  }
  TVMArgs args(&values[0], &type_codes[0], num_args);
  PackedFunc* p_rtn_func = new PackedFunc(WrapAsyncCall(args));
  return p_rtn_func;
}

MOBULA_DLL void RegisterMXAPI(
    decltype(MXShallowCopyNDArray) shallow_copy_ndarray,
    decltype(MXNDArrayFree) ndarray_free,
    decltype(MXNDArrayGetContext) ndarray_get_context,
    decltype(MXNDArrayToDLPack) ndarray_to_dlpack,
    decltype(MXEnginePushSyncND) engine_push_sync_nd) {
  MXShallowCopyNDArray = shallow_copy_ndarray;
  MXNDArrayFree = ndarray_free;
  MXNDArrayGetContext = ndarray_get_context;
  MXNDArrayToDLPack = ndarray_to_dlpack;
  MXEnginePushSyncND = engine_push_sync_nd;
}
}

#endif  // MOBULA_INC_GLUE_MX_H_
