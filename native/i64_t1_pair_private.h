#ifndef ET_I64_T1_PAIR_PRIVATE_H
#define ET_I64_T1_PAIR_PRIVATE_H

#ifndef ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE
#error "I1 exact T1 pair inspection is private to the G3-C4 linked feature"
#endif

#include "eshkol_transformer/i64_tensor.h"

enum {
  ET_I64_T1_PAIR_OK = 0,
  ET_I64_T1_PAIR_INVALID_STATE = 1,
  ET_I64_T1_PAIR_SHAPE = 2,
  ET_I64_T1_PAIR_DTYPE = 3,
  ET_I64_T1_PAIR_DEVICE = 4,
  ET_I64_T1_PAIR_LAYOUT = 5,
  ET_I64_T1_PAIR_STORAGE = 6
};

/* Inspects this I1 owner's live registries; never acquires a borrow. */
int32_t et_i64_tensor_private_t1_pair_validate_v1(
    const et_i64_tensor *tensor, const et_i64_tensor_borrow *borrow,
    const et_kernel_tensor_view_v1 *view, int64_t expected_length);

#endif
