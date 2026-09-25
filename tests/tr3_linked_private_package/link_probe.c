#if defined(TR3_HOSTILE_LEASE)
extern void tr3_private_lease(void) __asm__("tr3-lease-create-internal");
#define TARGET tr3_private_lease
typedef void (*target_fn)(void);
#elif defined(TR3_HOSTILE_NATIVE)
extern void et_tr3_c_private_i2_restore_create_v1(void);
#define TARGET et_tr3_c_private_i2_restore_create_v1
typedef void (*target_fn)(void);
#else
extern void __eshkol_lib_init__(void *);
#define TARGET __eshkol_lib_init__
typedef void (*target_fn)(void *);
#endif

int main(void) {
  target_fn volatile target = TARGET;
  return target == 0;
}
