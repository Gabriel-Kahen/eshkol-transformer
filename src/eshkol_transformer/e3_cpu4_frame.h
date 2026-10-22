#ifndef ET_E3_CPU4_FRAME_H
#define ET_E3_CPU4_FRAME_H
#include "e3_frame_internal.h"
#ifdef ET_CPU4_TRAINING_PRIVATE
/* Borrow existing lease and four caller-owned canonical scalar destinations. */
void *et_e3_private_cpu4_frame_create_v1(void *existing_lease,
    et_f32_tensor *loss, et_f32_tensor *weight,
    et_f32_tensor *perplexity, et_f32_tensor *accuracy);

#ifdef ET_E3_CPU4_TESTING
/* Raw downstream failure survives cleanup; selectors 0/1/2=domain/category/code.
 * C4 native domain is 4; K1/I1/I2 retain domains 1/2/3. */
int64_t et_e3_cpu4_private_raw_error_ref_v1(int64_t selector);
void et_e3_cpu4_test_fail_after(size_t count);
#endif
#endif
#endif
