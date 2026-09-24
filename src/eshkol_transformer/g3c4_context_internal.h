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

#ifdef ET_G3C4_PROMPT_T1_BORROW_PRIVATE
/* Authenticates an already-sealed T1 shell through the T1 native registry,
 * copies one or two byte-token IDs into a new owned dense CPU-i64 [1,P], and
 * retains no reference to the T1 shell. */
void *et_g3c4_private_input_from_t1_v1(void *sealed_t1);
/* The current closed tensor union contains only prompt inputs. */
int64_t et_g3c4_private_tensor_release_v1(void *input);
#endif

#ifdef ET_G3C4_FULL_PREFIX_FORWARD_PRIVATE
/* Synchronous fixed-T4 numerical leaf. token_ids/logits are borrowed only for
 * the call and must address 4 i64 / 1024 f32 elements respectively. */
int64_t et_g3c4_private_full_prefix_forward_v1(
    void *context, const int64_t token_ids[4], float logits[1024]);
#endif

#ifdef ET_G3C4_SAMPLER_TRANSPORT_PRIVATE
/* Returns speculative token/RNG candidates without mutating generator state. */
int64_t et_g3c4_private_sample_last_v1(
    void *context, const float full_logits[1024],
    int64_t *token, int64_t successor_rng[4]);
#endif

#ifdef ET_G3C4_TOKEN_FRAME_PRIVATE
/* Owns one pending A2 append from sampling through joint cache/RNG/token commit.
 * K/V are exact dense CPU-f32 [1,2,1,2] outputs for speculative_token. */
int64_t et_g3c4_private_token_frame_begin_v1(
    void *context, const float full_logits[1024],
    int64_t *speculative_token);
int64_t et_g3c4_private_token_frame_stage_v1(
    void *context, int64_t speculative_token,
    const float keys[4], const float values[4]);
int64_t et_g3c4_private_token_frame_publish_v1(
    void *context, int64_t *token);
int64_t et_g3c4_private_token_frame_abort_v1(void *context);
#endif

#ifdef ET_G3C4_TOKEN_FORWARD_PRIVATE
/* Executes the exact one-token numerical schedule for the sampled candidate,
 * stages its K/V in the pending token frame, and atomically returns next logits.
 * The frame remains ready and unpublished. */
int64_t et_g3c4_private_token_forward_v1(
    void *context, int64_t speculative_token, float logits[256]);
#endif

#ifdef ET_G3C4_PREFILL3_PRIVATE
/* Executes one exact three-token numerical prefill into an unpublished cache,
 * then atomically replaces the committed cache and parameter-bit snapshot.
 * Inputs and last-position logits are borrowed only for this call. */
int64_t et_g3c4_private_prefill3_v1(
    void *context, const int64_t token_ids[3], float last_logits[256]);

#ifdef ET_G3C4_PREFILL1_PRIVATE
/* Executes one exact one-token numerical prefill into an unpublished cache,
 * then atomically replaces the committed cache and parameter-bit snapshot. */
int64_t et_g3c4_private_prefill1_v1(
    void *context, int64_t token_id, float last_logits[256]);

#ifdef ET_G3C4_PREFILL2_PRIVATE
/* Executes one exact two-token numerical prefill into an unpublished cache,
 * then atomically replaces the committed cache and parameter-bit snapshot. */
int64_t et_g3c4_private_prefill2_v1(
    void *context, const int64_t token_ids[2], float last_logits[256]);
#endif
#endif
#endif

#ifdef ET_G3C4_PROMPT_PREFILL_PRIVATE
/* Authenticates an idle generator and Step 15A input, and rejects P+G > 2
 * before call acquisition or pinning. */
int64_t et_g3c4_private_prompt_prefill_preflight_v1(
    void *context, void *input, int64_t budget);
/* Borrows the exact Step 15A I1 IDs for one synchronous matching P1/P2
 * numerical prefill. The input owner and borrowed view are never retained. */
int64_t et_g3c4_private_prompt_prefill_v1(
    void *context, void *input, float last_logits[256]);
#endif

#ifdef ET_G3C4_OUTPUT_RESERVATION_PRIVATE
/* Reserves the exact native G0/G1 output payload during an active generation
 * call, before bound prefill. No numeric result or text is published. */
void *et_g3c4_private_output_reserve_v1(
    void *context, int64_t prompt_length);
/* Releases a pending reservation. Exact dead release is idempotent. */
int64_t et_g3c4_private_output_release_v1(void *output);
#endif

#ifdef ET_G3C4_LAST_LOGIT_FRAME_PRIVATE
/* Samples directly from one exact [1,256] row, then opens the accepted pending
 * one-token frame. No fabricated full-prefix carrier is constructed. */
int64_t et_g3c4_private_token_frame_begin_last_v1(
    void *context, const float last_logits[256],
    int64_t *speculative_token);
#endif

#ifdef __cplusplus
}
#endif

#endif
