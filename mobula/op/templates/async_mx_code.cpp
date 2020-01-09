MOBULA_DLL PackedFunc* ${func_idcode_hash}_register_mx() {
  return RegisterTVMFunc(
      "${func_idcode_hash}",
      [](TVMArgs args, TVMRetValue*) {
        KERNEL_RUN_BEGIN(DEV_ID);
        KERNEL_RUN_STREAM(${func_name}, STRM)(${args_inst_mx}
        );
        KERNEL_RUN_END();
      },
      ${num_const}, ${const_loc_code});
}

MOBULA_DLL void ${func_idcode_hash}_async_mx(
    PackedFunc* packed_func,
    ${args_def_async_mx}) {
  (*packed_func)(${args_inst});
}
