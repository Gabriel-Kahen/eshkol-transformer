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
#ifdef ET_G3T_GENERATOR_RNG_PRIVATE
void *et_g3t_private_generator_rng_v1(
    void *model_owner, void *rng, int64_t mode, int64_t temperature_bits,
    int64_t k, int64_t p_bits, int64_t max_new, int64_t eos);
#endif
int64_t et_g3t_private_generator_close_v1(void *context);
int64_t et_g3t_private_call_acquire_v1(void *context, int64_t call_kind);
int64_t et_g3t_private_call_abort_v1(void *context);
#ifdef ET_G3T_PREFILL_SAMPLE_PRIVATE
void *et_g3t_private_input_from_token_v1(int64_t token);
#ifdef ET_G3T_INPUT_FROM_T1_PRIVATE
void *et_g3t_private_input_from_t1_v1(void *sealed_t1);
#endif
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
void *et_g3t_private_input_from_pair_v1(int64_t first, int64_t second);
int64_t et_g3t_private_prompt_preflight_v1(
    void *context, void *input);
#endif
#ifdef ET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
int64_t et_g3t_private_full_request_preflight_v1(
    void *context, void *input);
#endif
int64_t et_g3t_private_tensor_release_v1(void *owner);
#ifdef ET_G3T_MANUAL_LOGITS_PRIVATE
void *et_g3t_private_logits_reserve_v1(void *context);
#endif
void *et_g3t_private_output_reserve_v1(void *context, int64_t prompt_length);
int64_t et_g3t_private_output_release_v1(void *output);
#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
void *et_g3t_private_output_ids_clone_v1(void *output);
#endif
#ifdef ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
void *et_g3t_private_output_lengths_clone_v1(void *output);
#endif
#ifdef ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
void *et_g3t_private_output_cache_lengths_clone_v1(void *output);
#endif
#ifdef ET_G3T_OUTPUT_RNG_CLONE_PRIVATE
void *et_g3t_private_output_rng_clone_v1(void *output);
int64_t et_g3t_private_rng_release_v1(void *rng);
#endif
int64_t et_g3t_private_output_prepare_v1(void *context, void *output);
int64_t et_g3t_private_output_copy_decode_ids_v1(
    void *context, void *output, void *staging_header);
int64_t et_g3t_private_output_accept_text_v1(
    void *context, void *output, void *raw_header);
int64_t et_g3t_private_frame_begin_v1(
    void *context, void *input, int64_t length);
int64_t et_g3t_private_role_step_v1(void *context, int64_t ordinal);
int64_t et_g3t_private_frame_prepare_v1(
    void *context, void *staged_result);
int64_t et_g3t_private_frame_commit_v1(void *context);
int64_t et_g3t_private_sample_v1(void *context);
int64_t et_g3t_private_call_prepare_end_v1(void *context);
int64_t et_g3t_private_call_finish_v1(void *context);
#endif
int64_t et_g3t_private_last_error_domain_v1(void);
int64_t et_g3t_private_last_error_category_v1(void);
int64_t et_g3t_private_last_error_code_v1(void);
#ifdef ET_G3T_TESTING
#ifdef ET_G3T_MANUAL_LOGITS_PRIVATE
int64_t et_g3t_test_logits_state_v1(void *logits);
int64_t et_g3t_test_logits_shape_v1(void *logits);
void *et_g3t_test_logits_borrow_begin_v1(void *logits);
int64_t et_g3t_test_logits_borrow_end_v1(void *borrow);
int64_t et_g3t_test_pending_logits_v1(void);
#ifdef ET_F32_TENSOR_TESTING
int64_t et_g3t_test_fail_f32_after_v1(uint64_t successful_allocations);
int64_t et_g3t_test_f32_tensors_v1(void);
#endif
#endif
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
int64_t et_g3t_test_output_id_set_v1(void *output, int64_t token);
int64_t et_g3t_test_binding_flip_v1(void *context);
int64_t et_g3t_test_output_state_v1(void *output);
#ifdef ET_G3T_INPUT_FROM_T1_PRIVATE
int64_t et_g3t_test_input_length_v1(void *input);
int64_t et_g3t_test_input_word_v1(void *input, int64_t index);
void *et_g3t_test_input_borrow_begin_v1(void *input);
int64_t et_g3t_test_input_borrow_end_v1(void *borrow);
int64_t et_g3t_test_live_t1_inputs_v1(void);
#endif
#ifdef ET_G3T_OUTPUT_IDS_CLONE_PRIVATE
int64_t et_g3t_test_ids_clone_state_v1(void *clone);
int64_t et_g3t_test_ids_clone_length_v1(void *clone);
int64_t et_g3t_test_ids_clone_word_v1(void *clone);
void *et_g3t_test_ids_clone_borrow_begin_v1(void *clone);
int64_t et_g3t_test_ids_clone_borrow_end_v1(void *borrow);
int64_t et_g3t_test_live_ids_clones_v1(void);
#endif
#ifdef ET_G3T_OUTPUT_LENGTHS_CLONE_PRIVATE
int64_t et_g3t_test_lengths_clone_state_v1(void *clone);
int64_t et_g3t_test_lengths_clone_value_v1(void *clone);
void *et_g3t_test_lengths_clone_borrow_begin_v1(void *clone);
int64_t et_g3t_test_lengths_clone_borrow_end_v1(void *borrow);
int64_t et_g3t_test_live_lengths_clones_v1(void);
#endif
#ifdef ET_G3T_OUTPUT_CACHE_LENGTHS_CLONE_PRIVATE
int64_t et_g3t_test_cache_lengths_clone_state_v1(void *clone);
int64_t et_g3t_test_cache_lengths_clone_value_v1(void *clone);
void *et_g3t_test_cache_lengths_clone_borrow_begin_v1(void *clone);
int64_t et_g3t_test_cache_lengths_clone_borrow_end_v1(void *borrow);
int64_t et_g3t_test_live_cache_lengths_clones_v1(void);
#endif
#ifdef ET_G3T_OUTPUT_RNG_CLONE_PRIVATE
int64_t et_g3t_test_rng_clone_state_v1(void *clone);
int64_t et_g3t_test_rng_clone_word_v1(void *clone, int64_t index);
int64_t et_g3t_test_live_rng_clones_v1(void);
#ifdef ET_G3T_GENERATOR_RNG_PRIVATE
int64_t et_g3t_test_rng_clone_busy_set_v1(void *clone, int64_t busy);
#endif
#endif
#ifdef ET_G3T_P2_ZERO_BUDGET_PRIVATE
int64_t et_g3t_test_input_length_set_v1(void *input, int64_t length);
#endif
#ifdef ET_G3T_FULL_REQUEST_PREFLIGHT_PRIVATE
int64_t et_g3t_test_rng_counter_set_v1(
    void *context, int64_t low, int64_t high);
#endif
#endif
#endif
#ifdef __cplusplus
}
#endif
#endif
