#include "e1b_error_consumer_bridge.h"

/* D2 extends the accepted T2 wrappers and compiles I2 helpers privately. */
#include "t2_wave2_package_bridge.c"
#define ET_I2_NATIVE_HELPERS_ONLY 1
#include "i2_wave2_package_bridge.c"
#undef ET_I2_NATIVE_HELPERS_ONLY

#define ET_D2_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_D2_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,             \
                                     eshkol_tagged_value_t right)

ET_D2_DECLARE_BINARY(et_e1b_private_d2_token_dataset_open_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_dataset_next_batch_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_dataset_cursor_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_dataset_end_cabi_v1);
ET_D2_DECLARE_BINARY(et_e1b_private_d2_token_dataset_seek_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_dataset_close_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_batch_inputs_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_batch_targets_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_batch_loss_mask_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_batch_validate_cabi_v1);
ET_D2_DECLARE_UNARY(et_e1b_private_d2_token_batch_release_cabi_v1);

#define ET_D2_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_D2_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

/*
 * The accepted batch accessors and exact repeated release are allocation-free.
 * Their Eshkol facade first proves procedure?, so these four private package
 * entries reconstruct the callable tag on the C stack instead of allocating a
 * one-element argument vector. No caller capability is retained here.
 */
#define ET_D2_PUBLIC_CALLABLE_UNARY(name, target)                            \
  void name(void *callable, void *output) {                                 \
    eshkol_tagged_value_t input;                                             \
    et_e1b_ensure_private_initialized_v1();                                 \
    input = eshkol_make_ptr((uint64_t)(uintptr_t)callable,                   \
                            ESHKOL_VALUE_CALLABLE);                          \
    *et_e1b_box_value_v1(output) = target(input);                            \
  }

/* Wrong-kind values still require their original operation-specific E1 error. */
void et_e1b_public_d2_token_batch_validate_v1(void *input, int64_t operation,
                                               void *output) {
  eshkol_tagged_value_t value;
  et_e1b_ensure_private_initialized_v1();
  value = *et_e1b_box_value_v1(input);
  switch (operation) {
  case 0:
    value = et_e1b_private_d2_token_batch_validate_cabi_v1(value);
    break;
  case 1:
    value = et_e1b_private_d2_token_batch_inputs_cabi_v1(value);
    break;
  case 2:
    value = et_e1b_private_d2_token_batch_targets_cabi_v1(value);
    break;
  case 3:
    value = et_e1b_private_d2_token_batch_loss_mask_cabi_v1(value);
    break;
  case 4:
    value = et_e1b_private_d2_token_batch_release_cabi_v1(value);
    break;
  default:
    __builtin_trap();
  }
  *et_e1b_box_value_v1(output) = value;
}

ET_D2_PUBLIC_BINARY(et_e1b_public_d2_token_dataset_open_v1,
                    et_e1b_private_d2_token_dataset_open_cabi_v1)
ET_D2_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_next_batch_v1,
                   et_e1b_private_d2_token_dataset_next_batch_cabi_v1)
ET_D2_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_cursor_v1,
                   et_e1b_private_d2_token_dataset_cursor_cabi_v1)
ET_D2_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_end_v1,
                   et_e1b_private_d2_token_dataset_end_cabi_v1)
ET_D2_PUBLIC_BINARY(et_e1b_public_d2_token_dataset_seek_v1,
                    et_e1b_private_d2_token_dataset_seek_cabi_v1)
ET_D2_PUBLIC_UNARY(et_e1b_public_d2_token_dataset_close_v1,
                   et_e1b_private_d2_token_dataset_close_cabi_v1)
ET_D2_PUBLIC_CALLABLE_UNARY(et_e1b_public_d2_token_batch_inputs_v1,
                            et_e1b_private_d2_token_batch_inputs_cabi_v1)
ET_D2_PUBLIC_CALLABLE_UNARY(et_e1b_public_d2_token_batch_targets_v1,
                            et_e1b_private_d2_token_batch_targets_cabi_v1)
ET_D2_PUBLIC_CALLABLE_UNARY(et_e1b_public_d2_token_batch_loss_mask_v1,
                            et_e1b_private_d2_token_batch_loss_mask_cabi_v1)
ET_D2_PUBLIC_CALLABLE_UNARY(et_e1b_public_d2_token_batch_release_v1,
                            et_e1b_private_d2_token_batch_release_cabi_v1)
