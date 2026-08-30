#include <stddef.h>
#include <stdint.h>

#include "e1b_error_consumer_bridge.h"

typedef struct et_e1b_box_v1 {
  int64_t length;
  eshkol_tagged_value_t value;
} et_e1b_box_v1;

_Static_assert(sizeof(eshkol_tagged_value_t) == 16,
               "E1B requires the pinned 16-byte tagged value ABI");
_Static_assert(_Alignof(eshkol_tagged_value_t) == 8,
               "E1B requires the pinned tagged-value alignment");
_Static_assert(offsetof(et_e1b_box_v1, value) == 8,
               "E1B requires the pinned heterogeneous-vector layout");
_Static_assert(sizeof(et_e1b_box_v1) == 24,
               "E1B requires one length and one tagged vector element");

extern void __eshkol_lib_init__(void *arena);
extern void *get_global_arena(void);

extern eshkol_tagged_value_t et_e1b_private_error_predicate_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_error_category_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_error_operation_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_error_message_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_error_details_cabi_v1(
    eshkol_tagged_value_t value);
extern eshkol_tagged_value_t et_e1b_private_error_cause_cabi_v1(
    eshkol_tagged_value_t value);
extern _Noreturn eshkol_tagged_value_t et_e1b_private_raise_cabi_v1(
    eshkol_tagged_value_t category, eshkol_tagged_value_t operation,
    eshkol_tagged_value_t message, eshkol_tagged_value_t details,
    eshkol_tagged_value_t cause);

eshkol_tagged_value_t *et_e1b_box_value_v1(void *opaque) {
  et_e1b_box_v1 *box = (et_e1b_box_v1 *)opaque;
  if (box == NULL || ESHKOL_GET_SUBTYPE(box) != HEAP_SUBTYPE_VECTOR ||
      box->length != 1) {
    __builtin_trap();
  }
  return &box->value;
}

void et_e1b_ensure_private_initialized_v1(void) {
  static int initialized = 0;
  if (!initialized) {
    __eshkol_lib_init__(get_global_arena());
    initialized = 1;
  }
}

#define ET_E1B_ACCESSOR(name, target)                                      \
  void name(void *input, void *output) {                                   \
    et_e1b_ensure_private_initialized_v1();                                \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));     \
  }

ET_E1B_ACCESSOR(et_e1b_error_predicate_v1,
                et_e1b_private_error_predicate_cabi_v1)
ET_E1B_ACCESSOR(et_e1b_error_category_v1,
                et_e1b_private_error_category_cabi_v1)
ET_E1B_ACCESSOR(et_e1b_error_operation_v1,
                et_e1b_private_error_operation_cabi_v1)
ET_E1B_ACCESSOR(et_e1b_error_message_v1,
                et_e1b_private_error_message_cabi_v1)
ET_E1B_ACCESSOR(et_e1b_error_details_v1,
                et_e1b_private_error_details_cabi_v1)
ET_E1B_ACCESSOR(et_e1b_error_cause_v1,
                et_e1b_private_error_cause_cabi_v1)

_Noreturn void et_e1b_consumer_raise_v1(void *category, void *operation,
                                        void *message, void *details,
                                        void *cause) {
  et_e1b_ensure_private_initialized_v1();
  (void)et_e1b_private_raise_cabi_v1(
      *et_e1b_box_value_v1(category), *et_e1b_box_value_v1(operation),
      *et_e1b_box_value_v1(message), *et_e1b_box_value_v1(details),
      *et_e1b_box_value_v1(cause));
  __builtin_unreachable();
}
