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
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
void *et_g3t_private_input_from_token_v1(int64_t token);
int64_t et_g3t_private_tensor_release_v1(void *input);
void *et_g3t_private_output_reserve_v1(void *context, int64_t prompt_length);
int64_t et_g3t_private_output_release_v1(void *output);
int64_t et_g3t_private_output_prepare_v1(void *context, void *output);
int64_t et_g3t_private_frame_begin_v1(
    void *context, void *input, int64_t length);
int64_t et_g3t_private_role_step_v1(void *context, int64_t ordinal);
int64_t et_g3t_private_frame_prepare_v1(
    void *context, void *staged_result);
int64_t et_g3t_private_frame_commit_v1(void *context);
int64_t et_g3t_private_sample_v1(void *context);
#endif
int64_t et_g3t_private_last_error_domain_v1(void);
int64_t et_g3t_private_last_error_category_v1(void);
int64_t et_g3t_private_last_error_code_v1(void);
#ifdef ET_G3T_TESTING
int64_t et_g3t_test_fail_alloc_after_v1(uint64_t successful_allocations);
int64_t et_g3t_test_fail_a2_after_v1(uint64_t successful_allocations);
#ifdef ET_I64_TENSOR_TESTING
int64_t et_g3t_test_fail_i1_after_v1(uint64_t successful_allocations);
#endif
uint64_t et_g3t_test_live_contexts_v1(void);
int64_t et_g3t_test_pin_count_v1(void *context);
int64_t et_g3t_test_cache_empty_v1(void *context);
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
int64_t et_g3t_test_logit_bits_v1(void *context, int64_t index);
int64_t et_g3t_test_sample_token_v1(void *context);
int64_t et_g3t_test_rng_word_v1(void *context, int64_t index);
int64_t et_g3t_test_successor_word_v1(void *context, int64_t index);
int64_t et_g3t_test_greedy_argmax_v1(void *context);
int64_t et_g3t_test_frame_logit_bits_v1(void *context, int64_t index);
int64_t et_g3t_test_cache_length_v1(void *context);
int64_t et_g3t_test_output_word_v1(void *output, int64_t field);
void *et_g3t_test_output_borrow_begin_v1(void *output);
int64_t et_g3t_test_output_borrow_end_v1(void *borrow);
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif
