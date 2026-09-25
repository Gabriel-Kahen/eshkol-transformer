#ifndef ET_G3T_TRANSPORT_H
#define ET_G3T_TRANSPORT_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Private G3-T source ABI subset. All calls require external serialization. */
void *et_g3t_private_generator_seed_v1(
    void *model_owner, int64_t seed, int64_t mode, int64_t temperature_bits,
    int64_t k, int64_t p_bits, int64_t max_new, int64_t eos);
int64_t et_g3t_private_generator_close_v1(void *context);
int64_t et_g3t_private_call_acquire_v1(void *context, int64_t call_kind);
int64_t et_g3t_private_call_abort_v1(void *context);
int64_t et_g3t_private_last_error_domain_v1(void);
int64_t et_g3t_private_last_error_category_v1(void);
int64_t et_g3t_private_last_error_code_v1(void);
#ifdef ET_G3T_TESTING
int64_t et_g3t_test_fail_alloc_after_v1(uint64_t successful_allocations);
int64_t et_g3t_test_fail_a2_after_v1(uint64_t successful_allocations);
uint64_t et_g3t_test_live_contexts_v1(void);
int64_t et_g3t_test_pin_count_v1(void *context);
int64_t et_g3t_test_cache_empty_v1(void *context);
#endif
#ifdef __cplusplus
}
#endif
#endif
