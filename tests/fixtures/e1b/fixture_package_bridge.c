#include "e1b_error_consumer_bridge.h"

/* Renamed compiler C-ABI thunks; both remain local in the final artifact. */
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_consumer_a_raise_cabi_v1(
    eshkol_tagged_value_t path);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_consumer_b_raise_cabi_v1(
    eshkol_tagged_value_t cause);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_category_probe_cabi_v1(eshkol_tagged_value_t category);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_invalid_category_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_invalid_operation_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_invalid_message_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_invalid_details_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_duplicate_details_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_opaque_details_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t
et_e1b_private_fixture_invalid_cause_cabi_v1(void);
extern _Noreturn eshkol_tagged_value_t et_e1b_private_fixture_owned_cabi_v1(
    eshkol_tagged_value_t message, eshkol_tagged_value_t detail);

_Noreturn void et_e1b_public_fixture_consumer_a_raise_v1(void *path_box) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_e1b_private_fixture_consumer_a_raise_cabi_v1(
      *et_e1b_box_value_v1(path_box));
  __builtin_unreachable();
}

_Noreturn void et_e1b_public_fixture_consumer_b_raise_v1(void *cause_box) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_e1b_private_fixture_consumer_b_raise_cabi_v1(
      *et_e1b_box_value_v1(cause_box));
  __builtin_unreachable();
}

_Noreturn void et_e1b_public_fixture_category_probe_v1(void *category_box) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_e1b_private_fixture_category_probe_cabi_v1(
      *et_e1b_box_value_v1(category_box));
  __builtin_unreachable();
}

#define E1B_NOARG_PUBLIC(name, target) \
  _Noreturn void name(void) {          \
    et_e1b_ensure_private_initialized_v1(); \
    (void)target();                    \
    __builtin_unreachable();           \
  }

E1B_NOARG_PUBLIC(et_e1b_public_fixture_invalid_category_v1,
                 et_e1b_private_fixture_invalid_category_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_invalid_operation_v1,
                 et_e1b_private_fixture_invalid_operation_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_invalid_message_v1,
                 et_e1b_private_fixture_invalid_message_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_invalid_details_v1,
                 et_e1b_private_fixture_invalid_details_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_duplicate_details_v1,
                 et_e1b_private_fixture_duplicate_details_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_opaque_details_v1,
                 et_e1b_private_fixture_opaque_details_cabi_v1)
E1B_NOARG_PUBLIC(et_e1b_public_fixture_invalid_cause_v1,
                 et_e1b_private_fixture_invalid_cause_cabi_v1)

_Noreturn void et_e1b_public_fixture_owned_v1(void *message_box,
                                               void *detail_box) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_e1b_private_fixture_owned_cabi_v1(
      *et_e1b_box_value_v1(message_box), *et_e1b_box_value_v1(detail_box));
  __builtin_unreachable();
}
