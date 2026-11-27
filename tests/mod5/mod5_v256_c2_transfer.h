#ifndef ESHKOL_TEST_MOD5_V256_C2_TRANSFER_H
#define ESHKOL_TEST_MOD5_V256_C2_TRANSFER_H

#include "f32_parameter_internal.h"

/* Test-only native bridge. Slots are in canonical C1 path order. Each physical
 * buffer is aligned binary32 words with the exact extent for its slot. */
et_f32_parameter *et_mod5_v256_test_parameter_v1(
    int64_t slot, const uint32_t *words, int64_t word_count);
int64_t et_mod5_v256_test_parameter_bits_v1(
    int64_t slot, const et_f32_parameter *parameter, uint32_t *words,
    int64_t word_count);

/* Gradients are already normalized by the decoder. Tied mode has exactly
 * eleven destinations, with the combined gradient in head.weight and no
 * token_embedding.weight entry. All destinations must be bound to a native
 * parameter identity and start with ordinal 0. */
int64_t et_mod5_v256_test_gradient_transfer_v1(
    et_f32_parameter *const *parameters, int64_t parameter_count,
    const uint32_t *words, int64_t word_count, int64_t tied);

#endif
