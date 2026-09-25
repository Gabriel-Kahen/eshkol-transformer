#ifndef ET_TR3_C_PRIVATE_INITIALIZER_BRIDGE_H
#define ET_TR3_C_PRIVATE_INITIALIZER_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Private package ABI. Zero also means an earlier call completed successfully. */
enum {
  ET_TR3_C_INIT_READY_V1 = 0,
  ET_TR3_C_INIT_BUSY_V1 = 1,
  ET_TR3_C_INIT_RUNTIME_V1 = 2,
  ET_TR3_C_INIT_ARENA_V1 = 3,
  ET_TR3_C_INIT_HANDLER_V1 = 4,
  ET_TR3_C_INIT_EXCEPTION_V1 = 5,
};

#if defined(__GNUC__)
#define ET_TR3_C_PRIVATE_EXPORT __attribute__((visibility("default")))
#else
#define ET_TR3_C_PRIVATE_EXPORT
#endif
ET_TR3_C_PRIVATE_EXPORT int et_tr3_c_private_initialize_v1(void);

#ifdef __cplusplus
}
#endif

#endif
