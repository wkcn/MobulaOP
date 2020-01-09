MOBULA_DLL void ${func_idcode_hash}(const int device_id, ${args_def}) {
  KERNEL_RUN_BEGIN(device_id);
  KERNEL_RUN(${func_name})(${args_inst});
  KERNEL_RUN_END();
}
