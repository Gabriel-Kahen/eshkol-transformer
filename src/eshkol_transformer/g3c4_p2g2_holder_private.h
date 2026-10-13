#ifndef ET_G3C4_P2G2_HOLDER_PRIVATE_H
#define ET_G3C4_P2G2_HOLDER_PRIVATE_H

#include <stdint.h>

/* Source-private, fixed-profile held-lease capability. Not a public G3 ABI.
 * 1 = pending I1, 2 = staged A2; phase 2 = first READY, 3 = second READY.
 * Generation is positive on success, zero on error. All later operations
 * authenticate pointer and generation before dereference. A successful dispose
 * consumes the capability. At the fixed 1,048,576 lifetime-token limit,
 * creation returns NULL with C4 INTERNAL/ALLOCATION; tokens are never reused. */
#ifdef ET_G3C4_P2_G2_HELD_HOLDER_PRIVATE
void *et_g3c4_private_p2g2_holder_new_v1(int64_t kind, int64_t phase);
int64_t et_g3c4_private_p2g2_holder_generation_v1(void *holder);
int64_t et_g3c4_private_p2g2_holder_bind_v1(
    void *holder, int64_t generation, void *context, void *pending);
int64_t et_g3c4_private_p2g2_holder_begin_capture_v1(
    void *holder, int64_t generation, void *context, void *pending);
int64_t et_g3c4_private_p2g2_holder_witness_v1(
    void *holder, int64_t generation, void *context, void *pending);
int64_t et_g3c4_private_p2g2_holder_normal_close_v1(
    void *holder, int64_t generation, void *context, void *pending);
int64_t et_g3c4_private_p2g2_holder_abort_attempt_v1(
    void *holder, int64_t generation, void *context, void *pending,
    int64_t ordinal);
int64_t et_g3c4_private_p2g2_holder_end_after_busy_v1(
    void *holder, int64_t generation, void *context, void *pending);
int64_t et_g3c4_private_p2g2_holder_dispose_v1(
    void *holder, int64_t generation);
#endif

#endif
