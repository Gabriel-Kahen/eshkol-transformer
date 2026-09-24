#ifndef ET_G3C4_CONTEXT_INTERNAL_H
#define ET_G3C4_CONTEXT_INTERNAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Source-private ABI. All context calls require external serialization. */
void *et_g3c4_private_context_create_v1(void *c4_owner);
int64_t et_g3c4_private_context_close_v1(void *context);
int64_t et_g3c4_private_call_acquire_v1(
    void *context, int64_t call_kind, int64_t budget);
int64_t et_g3c4_private_call_prepare_end_v1(void *context);
int64_t et_g3c4_private_call_finish_v1(void *context);
int64_t et_g3c4_private_call_abort_v1(void *context);

#ifdef ET_G3C4_GENERATOR_PRIVATE
void *et_g3c4_private_generator_seed_v1(
    void *c4_owner, int64_t seed, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
void *et_g3c4_private_generator_rng_v1(
    void *c4_owner, void *rng, int64_t mode,
    int64_t temperature_bits, int64_t k, int64_t p_bits,
    int64_t max_new, int64_t eos);
int64_t et_g3c4_private_generator_close_v1(void *context);

void *et_g3c4_private_rng_seed_v1(int64_t seed);
void *et_g3c4_private_rng_clone_v1(void *rng);
int64_t et_g3c4_private_rng_word_v1(void *rng, int64_t index);
int64_t et_g3c4_private_rng_release_v1(void *rng);
#endif

#ifdef __cplusplus
}
#endif

#endif
