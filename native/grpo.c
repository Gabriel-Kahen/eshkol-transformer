#include "eshkol_transformer/grpo_abi.h"

#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u &&
                   sizeof(double) == 8u && sizeof(size_t) <= sizeof(uint64_t),
               "GRPO engine requires 8-bit bytes and IEEE-754 binary32/64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   DBL_MANT_DIG == 53 && FLT_EVAL_METHOD == 0,
               "GRPO engine requires IEEE-754 binary32/64 evaluation");

/* ---- FP environment, alignment, span, and finiteness helpers ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  const unsigned int mxcsr = _mm_getcsr();
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, mask all exceptions, and disable FTZ/DAZ. */
  return (x87 & 0x0c3fu) == 0x003fu && (mxcsr & 0xffc0u) == 0x1f80u &&
         fegetround() == FE_TONEAREST;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int aligned_span(const void *ptr, size_t alignment, size_t bytes) {
  if (ptr == NULL)
    return 0;
  const uintptr_t addr = (uintptr_t)ptr;
  return (addr % alignment) == 0u && addr <= UINTPTR_MAX - bytes;
}

typedef struct span_entry {
  const void *ptr;
  size_t bytes;
} span_entry;

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u)
    return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes)
    return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

/* Output spans must be pairwise disjoint and disjoint from all input spans.
 * Read-only inputs are permitted to alias one another (e.g. log_old == log_ref
 * at step 0). */
static int outputs_alias_inputs_or_each_other(const span_entry *inputs,
                                              size_t num_inputs,
                                              const span_entry *outputs,
                                              size_t num_outputs) {
  for (size_t i = 0; i < num_outputs; ++i) {
    if (outputs[i].ptr == NULL || outputs[i].bytes == 0u)
      continue;
    for (size_t j = 0; j < i; ++j) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, outputs[j].ptr,
                        outputs[j].bytes))
        return 1;
    }
    for (size_t j = 0; j < num_inputs; ++j) {
      if (spans_overlap(outputs[i].ptr, outputs[i].bytes, inputs[j].ptr,
                        inputs[j].bytes))
        return 1;
    }
  }
  return 0;
}

static et_grpo_status check_aligned_buf(const void *ptr, size_t alignment,
                                        size_t bytes) {
  if (ptr == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if (!aligned_span(ptr, alignment, bytes))
    return ET_GRPO_ERR_ALIGNMENT;
  return ET_GRPO_OK;
}

static int all_finite_f32(const float *data, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!isfinite(data[i]))
      return 0;
  }
  return 1;
}

static int validate_mask_and_counts(uint32_t batch_size, uint32_t seq_len,
                                    const uint8_t *mask,
                                    uint32_t *out_active_tokens,
                                    uint32_t *out_active_seqs) {
  uint32_t total_active = 0u;
  uint32_t active_seqs = 0u;
  for (uint32_t n = 0; n < batch_size; ++n) {
    uint32_t seq_active = 0u;
    const size_t row_base = (size_t)n * (size_t)seq_len;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const uint8_t m = mask[row_base + (size_t)t];
      if (m > 1u)
        return 0;
      seq_active += (uint32_t)m;
    }
    if (seq_active > 0u) {
      ++active_seqs;
      total_active += seq_active;
    }
  }
  *out_active_tokens = total_active;
  *out_active_seqs = active_seqs;
  return 1;
}

static et_grpo_status validate_ntv_dims(uint32_t batch_size, uint32_t seq_len,
                                        uint32_t vocab_size, int require_vocab,
                                        size_t *out_nt, size_t *out_ntv) {
  if (batch_size < 1u || batch_size > ET_GRPO_MAX_BATCH || seq_len < 1u ||
      seq_len > ET_GRPO_MAX_SEQ_LEN)
    return ET_GRPO_ERR_SHAPE;
  const size_t nt = (size_t)batch_size * (size_t)seq_len;
  size_t ntv = 0u;
  if (require_vocab) {
    if (vocab_size < 1u || vocab_size > ET_GRPO_MAX_VOCAB)
      return ET_GRPO_ERR_SHAPE;
    if (nt > (size_t)ET_GRPO_MAX_LOGITS_ELEMS / (size_t)vocab_size)
      return ET_GRPO_ERR_SHAPE;
    ntv = nt * (size_t)vocab_size;
  } else if (vocab_size > ET_GRPO_MAX_VOCAB) {
    return ET_GRPO_ERR_SHAPE;
  }
  if (out_nt != NULL)
    *out_nt = nt;
  if (out_ntv != NULL)
    *out_ntv = ntv;
  return ET_GRPO_OK;
}

static et_grpo_status validate_loss_config_internal(
    const et_grpo_loss_config *cfg, int require_vocab, size_t *out_nt,
    size_t *out_ntv) {
  if (cfg == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if (!aligned_span(cfg, _Alignof(et_grpo_loss_config), sizeof(*cfg)))
    return ET_GRPO_ERR_ALIGNMENT;
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;
  if (cfg->reserved != 0u)
    return ET_GRPO_ERR_ARGUMENT;
  const et_grpo_status dim_status = validate_ntv_dims(
      cfg->batch_size, cfg->seq_len, cfg->vocab_size, require_vocab, out_nt,
      out_ntv);
  if (dim_status != ET_GRPO_OK)
    return dim_status;
  if (cfg->mask_norm_mode != ET_GRPO_NORM_GLOBAL_TOKEN_MEAN &&
      cfg->mask_norm_mode != ET_GRPO_NORM_PER_SEQ_MEAN)
    return ET_GRPO_ERR_ARGUMENT;
  if (!isfinite(cfg->clip_eps_low) || !isfinite(cfg->clip_eps_high) ||
      !isfinite(cfg->kl_beta))
    return ET_GRPO_ERR_NONFINITE;
  if (!(cfg->clip_eps_low > 0.0f && cfg->clip_eps_low <= 1.0f) ||
      !(cfg->clip_eps_high > 0.0f) || !(cfg->kl_beta >= 0.0f))
    return ET_GRPO_ERR_ARGUMENT;
  return ET_GRPO_OK;
}

/* Numerically stable log-sum-exp and target log-probability in binary64. */
static int compute_row_lse_and_logprob(const float *row, uint32_t vocab_size,
                                       int32_t target, double *out_lse,
                                       double *out_logprob) {
  double max_val = (double)row[0];
  for (uint32_t v = 1u; v < vocab_size; ++v) {
    const double x = (double)row[v];
    if (x > max_val)
      max_val = x;
  }
  double sum_exp = 0.0;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    sum_exp += exp((double)row[v] - max_val);
  }
  if (!isfinite(sum_exp) || !(sum_exp > 0.0))
    return 0;
  const double lse = max_val + log(sum_exp);
  const double lp = (double)row[(uint32_t)target] - lse;
  if (!isfinite(lse) || !isfinite(lp))
    return 0;
  *out_lse = lse;
  *out_logprob = lp;
  return 1;
}

/* Evaluate one active token's GRPO clipped surrogate, Schulman k3 KL, and
 * exact derivative w.r.t. log_pi (unweighted). */
static et_grpo_status eval_token_grpo_scalar(
    double log_pi, double log_old, double log_ref, double adv, double c_low,
    double c_high, double kl_beta, double *out_neg_pg_obj, double *out_kl,
    double *out_dloss_dlog_pi, int *out_clipped) {
  const double delta_old = log_pi - log_old;
  const double delta_ref = log_ref - log_pi;
  if (!isfinite(delta_old) || !isfinite(delta_ref))
    return ET_GRPO_ERR_NONFINITE;
  if (delta_old > (double)ET_GRPO_MAX_LOG_DIFF ||
      delta_ref > (double)ET_GRPO_MAX_LOG_DIFF)
    return ET_GRPO_ERR_OVERFLOW;

  const double rho = exp(delta_old);
  if (!isfinite(rho))
    return ET_GRPO_ERR_OVERFLOW;

  double rho_clip = rho;
  if (rho_clip < c_low)
    rho_clip = c_low;
  if (rho_clip > c_high)
    rho_clip = c_high;

  const int clipped = (adv > 0.0 && rho > c_high) || (adv < 0.0 && rho < c_low);
  const double pg_obj = clipped ? (rho_clip * adv) : (rho * adv);
  const double dpg_dlog_pi = clipped ? 0.0 : (rho * adv);

  /* Schulman k3 non-negative KL estimator:
   *   kl = exp(delta_ref) - delta_ref - 1 = expm1(delta_ref) - delta_ref >= 0
   * Exact derivative w.r.t. log_pi:
   *   d(kl)/d(log_pi) = 1 - exp(delta_ref) = -expm1(delta_ref). */
  const double em1_ref = expm1(delta_ref);
  if (!isfinite(em1_ref))
    return ET_GRPO_ERR_OVERFLOW;
  double kl = em1_ref - delta_ref;
  if (!isfinite(kl))
    return ET_GRPO_ERR_OVERFLOW;
  if (kl < 0.0)
    kl = 0.0;
  const double dkl_dlog_pi = -em1_ref;

  const double neg_pg = -pg_obj;
  const double dloss_dlog_pi = -dpg_dlog_pi + kl_beta * dkl_dlog_pi;
  if (!isfinite(neg_pg) || !isfinite(dloss_dlog_pi))
    return ET_GRPO_ERR_OVERFLOW;

  *out_neg_pg_obj = neg_pg;
  *out_kl = kl;
  *out_dloss_dlog_pi = dloss_dlog_pi;
  *out_clipped = clipped;
  return ET_GRPO_OK;
}

static int compute_mean_reward(uint32_t batch_size, const float *rewards,
                               float *out_mean_reward) {
  if (rewards == NULL) {
    *out_mean_reward = 0.0f;
    return 1;
  }
  double sum = 0.0;
  for (uint32_t n = 0; n < batch_size; ++n)
    sum += (double)rewards[n];
  const float mean = (float)(sum / (double)batch_size);
  if (!isfinite(mean))
    return 0;
  *out_mean_reward = mean;
  return 1;
}

/* ---- Public API: Config Validation ---- */

et_grpo_status et_grpo_validate_loss_config(const et_grpo_loss_config *cfg) {
  return validate_loss_config_internal(cfg, 1, NULL, NULL);
}

/* ---- Public API: Group-Relative Advantages ---- */

et_grpo_status et_grpo_group_advantages_v1_raw(uint32_t num_groups,
                                               uint32_t group_size,
                                               const void *rewards_raw,
                                               uint32_t adv_mode, float eps,
                                               void *out_advantages_raw) {
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;
  if (num_groups < 1u || num_groups > ET_GRPO_MAX_GROUPS || group_size < 2u ||
      group_size > ET_GRPO_MAX_GROUP_SIZE)
    return ET_GRPO_ERR_SHAPE;
  const size_t total = (size_t)num_groups * (size_t)group_size;
  if (total > (size_t)ET_GRPO_MAX_BATCH)
    return ET_GRPO_ERR_SHAPE;
  if (adv_mode != ET_GRPO_ADV_STANDARD && adv_mode != ET_GRPO_ADV_DR_GRPO)
    return ET_GRPO_ERR_ARGUMENT;
  if (!isfinite(eps))
    return ET_GRPO_ERR_NONFINITE;
  if (adv_mode == ET_GRPO_ADV_STANDARD ? !(eps > 0.0f) : !(eps >= 0.0f))
    return ET_GRPO_ERR_ARGUMENT;

  const size_t bytes = total * sizeof(float);
  et_grpo_status st = check_aligned_buf(rewards_raw, _Alignof(float), bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(out_advantages_raw, _Alignof(float), bytes);
  if (st != ET_GRPO_OK)
    return st;

  const span_entry inputs[1] = {{rewards_raw, bytes}};
  const span_entry outputs[1] = {{out_advantages_raw, bytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 1u, outputs, 1u))
    return ET_GRPO_ERR_ALIAS;

  const float *rewards = (const float *)rewards_raw;
  float *out_advantages = (float *)out_advantages_raw;
  if (!all_finite_f32(rewards, total))
    return ET_GRPO_ERR_NONFINITE;

  /* Pass 1: compute and verify every output is finite before mutating output.
   * Use direct IEEE-754 binary64 division by (double)group_size rather than
   * multiplying by 1.0 / (double)group_size to avoid double rounding on
   * reciprocals such as G = 49 where 49.0 * (1.0 / 49.0) != 1.0. */
  const double g_d = (double)group_size;
  const double eps_d = (double)eps;
  for (uint32_t b = 0; b < num_groups; ++b) {
    const size_t base = (size_t)b * (size_t)group_size;
    double sum = 0.0;
    for (uint32_t g = 0; g < group_size; ++g)
      sum += (double)rewards[base + (size_t)g];
    const double mean = sum / g_d;
    if (!isfinite(mean))
      return ET_GRPO_ERR_OVERFLOW;

    double denom = 1.0;
    if (adv_mode == ET_GRPO_ADV_STANDARD) {
      double var_sum = 0.0;
      for (uint32_t g = 0; g < group_size; ++g) {
        const double diff = (double)rewards[base + (size_t)g] - mean;
        var_sum += diff * diff;
      }
      const double var = var_sum / g_d;
      denom = sqrt(var) + eps_d;
      if (!isfinite(var) || !isfinite(denom) || !(denom > 0.0))
        return ET_GRPO_ERR_OVERFLOW;
    }

    for (uint32_t g = 0; g < group_size; ++g) {
      const double diff = (double)rewards[base + (size_t)g] - mean;
      const float adv =
          (float)(adv_mode == ET_GRPO_ADV_STANDARD ? (diff / denom) : diff);
      if (!isfinite(adv))
        return ET_GRPO_ERR_OVERFLOW;
    }
  }

  /* Pass 2: write validated outputs. */
  for (uint32_t b = 0; b < num_groups; ++b) {
    const size_t base = (size_t)b * (size_t)group_size;
    double sum = 0.0;
    for (uint32_t g = 0; g < group_size; ++g)
      sum += (double)rewards[base + (size_t)g];
    const double mean = sum / g_d;

    double denom = 1.0;
    if (adv_mode == ET_GRPO_ADV_STANDARD) {
      double var_sum = 0.0;
      for (uint32_t g = 0; g < group_size; ++g) {
        const double diff = (double)rewards[base + (size_t)g] - mean;
        var_sum += diff * diff;
      }
      denom = sqrt(var_sum / g_d) + eps_d;
    }

    for (uint32_t g = 0; g < group_size; ++g) {
      const double diff = (double)rewards[base + (size_t)g] - mean;
      out_advantages[base + (size_t)g] =
          (float)(adv_mode == ET_GRPO_ADV_STANDARD ? (diff / denom) : diff);
    }
  }
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_group_advantages_v1(uint32_t num_groups,
                                           uint32_t group_size,
                                           const float *rewards,
                                           uint32_t adv_mode, float eps,
                                           float *out_advantages) {
  return et_grpo_group_advantages_v1_raw(num_groups, group_size, rewards,
                                         adv_mode, eps, out_advantages);
}

/* ---- Public API: Per-Token and Sequence Log-Probabilities ---- */

et_grpo_status et_grpo_token_logprobs_v1_raw(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const void *logits_raw, const void *targets_raw, const uint8_t *mask,
    void *out_token_logprobs_raw, void *out_seq_logprobs_raw) {
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;

  size_t nt = 0u;
  size_t ntv = 0u;
  et_grpo_status st =
      validate_ntv_dims(batch_size, seq_len, vocab_size, 1, &nt, &ntv);
  if (st != ET_GRPO_OK)
    return st;

  const size_t logits_bytes = ntv * sizeof(float);
  const size_t targets_bytes = nt * sizeof(int32_t);
  const size_t mask_bytes = nt * sizeof(uint8_t);
  const size_t tok_lp_bytes = nt * sizeof(float);
  const size_t seq_lp_bytes = (size_t)batch_size * sizeof(float);

  st = check_aligned_buf(logits_raw, _Alignof(float), logits_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(targets_raw, _Alignof(int32_t), targets_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(mask, _Alignof(uint8_t), mask_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(out_token_logprobs_raw, _Alignof(float), tok_lp_bytes);
  if (st != ET_GRPO_OK)
    return st;
  if (out_seq_logprobs_raw != NULL) {
    st = check_aligned_buf(out_seq_logprobs_raw, _Alignof(float), seq_lp_bytes);
    if (st != ET_GRPO_OK)
      return st;
  }

  const span_entry inputs[3] = {{logits_raw, logits_bytes},
                                {targets_raw, targets_bytes},
                                {mask, mask_bytes}};
  const span_entry outputs[2] = {
      {out_token_logprobs_raw, tok_lp_bytes},
      {out_seq_logprobs_raw, out_seq_logprobs_raw ? seq_lp_bytes : 0u}};
  if (outputs_alias_inputs_or_each_other(inputs, 3u, outputs, 2u))
    return ET_GRPO_ERR_ALIAS;

  const float *logits = (const float *)logits_raw;
  const int32_t *targets = (const int32_t *)targets_raw;
  float *out_tok = (float *)out_token_logprobs_raw;
  float *out_seq = (float *)out_seq_logprobs_raw;

  if (!all_finite_f32(logits, ntv))
    return ET_GRPO_ERR_NONFINITE;
  for (size_t i = 0; i < nt; ++i) {
    if (mask[i] > 1u)
      return ET_GRPO_ERR_ARGUMENT;
    if (targets[i] < 0 || (uint32_t)targets[i] >= vocab_size)
      return ET_GRPO_ERR_INDEX;
  }

  /* Pass 1: validate finiteness of every token and sequence log-prob. */
  for (uint32_t n = 0; n < batch_size; ++n) {
    double seq_sum = 0.0;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const size_t idx = (size_t)n * (size_t)seq_len + (size_t)t;
      if (mask[idx] == 0u)
        continue;
      const float *row = logits + idx * (size_t)vocab_size;
      double lse = 0.0;
      double lp = 0.0;
      if (!compute_row_lse_and_logprob(row, vocab_size, targets[idx], &lse,
                                       &lp))
        return ET_GRPO_ERR_OVERFLOW;
      const float lp_f32 = (float)lp;
      if (!isfinite(lp_f32))
        return ET_GRPO_ERR_OVERFLOW;
      seq_sum += lp;
    }
    if (!isfinite((float)seq_sum))
      return ET_GRPO_ERR_OVERFLOW;
  }

  /* Pass 2: write validated outputs. */
  for (uint32_t n = 0; n < batch_size; ++n) {
    double seq_sum = 0.0;
    for (uint32_t t = 0; t < seq_len; ++t) {
      const size_t idx = (size_t)n * (size_t)seq_len + (size_t)t;
      if (mask[idx] == 0u) {
        out_tok[idx] = 0.0f;
        continue;
      }
      const float *row = logits + idx * (size_t)vocab_size;
      double lse = 0.0;
      double lp = 0.0;
      (void)compute_row_lse_and_logprob(row, vocab_size, targets[idx], &lse,
                                        &lp);
      out_tok[idx] = (float)lp;
      seq_sum += lp;
    }
    if (out_seq != NULL)
      out_seq[n] = (float)seq_sum;
  }
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_token_logprobs_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const float *logits, const int32_t *targets, const uint8_t *mask,
    float *out_token_logprobs, float *out_seq_logprobs) {
  return et_grpo_token_logprobs_v1_raw(batch_size, seq_len, vocab_size, logits,
                                       targets, mask, out_token_logprobs,
                                       out_seq_logprobs);
}

et_grpo_status et_grpo_token_logprobs_backward_v1_raw(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const void *logits_raw, const void *targets_raw, const uint8_t *mask,
    const void *grad_token_logprobs_raw, void *out_grad_logits_raw) {
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;

  size_t nt = 0u;
  size_t ntv = 0u;
  et_grpo_status st =
      validate_ntv_dims(batch_size, seq_len, vocab_size, 1, &nt, &ntv);
  if (st != ET_GRPO_OK)
    return st;

  const size_t logits_bytes = ntv * sizeof(float);
  const size_t targets_bytes = nt * sizeof(int32_t);
  const size_t mask_bytes = nt * sizeof(uint8_t);
  const size_t grad_tok_bytes = nt * sizeof(float);

  st = check_aligned_buf(logits_raw, _Alignof(float), logits_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(targets_raw, _Alignof(int32_t), targets_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(mask, _Alignof(uint8_t), mask_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(grad_token_logprobs_raw, _Alignof(float),
                         grad_tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(out_grad_logits_raw, _Alignof(float), logits_bytes);
  if (st != ET_GRPO_OK)
    return st;

  const span_entry inputs[4] = {{logits_raw, logits_bytes},
                                {targets_raw, targets_bytes},
                                {mask, mask_bytes},
                                {grad_token_logprobs_raw, grad_tok_bytes}};
  const span_entry outputs[1] = {{out_grad_logits_raw, logits_bytes}};
  if (outputs_alias_inputs_or_each_other(inputs, 4u, outputs, 1u))
    return ET_GRPO_ERR_ALIAS;

  const float *logits = (const float *)logits_raw;
  const int32_t *targets = (const int32_t *)targets_raw;
  const float *grad_tok = (const float *)grad_token_logprobs_raw;
  float *out_grad = (float *)out_grad_logits_raw;

  if (!all_finite_f32(logits, ntv) || !all_finite_f32(grad_tok, nt))
    return ET_GRPO_ERR_NONFINITE;
  for (size_t i = 0; i < nt; ++i) {
    if (mask[i] > 1u)
      return ET_GRPO_ERR_ARGUMENT;
    if (targets[i] < 0 || (uint32_t)targets[i] >= vocab_size)
      return ET_GRPO_ERR_INDEX;
  }

  /* Pass 1: validate finiteness of every output gradient element. */
  for (size_t i = 0; i < nt; ++i) {
    if (mask[i] == 0u)
      continue;
    const float *row = logits + i * (size_t)vocab_size;
    double lse = 0.0;
    double lp = 0.0;
    if (!compute_row_lse_and_logprob(row, vocab_size, targets[i], &lse, &lp))
      return ET_GRPO_ERR_OVERFLOW;
    const double g = (double)grad_tok[i];
    const uint32_t tgt = (uint32_t)targets[i];
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      const double p = exp((double)row[v] - lse);
      const float gv = (float)(g * ((v == tgt ? 1.0 : 0.0) - p));
      if (!isfinite(gv))
        return ET_GRPO_ERR_OVERFLOW;
    }
  }

  /* Pass 2: write validated gradients. */
  for (size_t i = 0; i < nt; ++i) {
    float *grow = out_grad + i * (size_t)vocab_size;
    if (mask[i] == 0u) {
      for (uint32_t v = 0u; v < vocab_size; ++v)
        grow[v] = 0.0f;
      continue;
    }
    const float *row = logits + i * (size_t)vocab_size;
    double lse = 0.0;
    double lp = 0.0;
    (void)compute_row_lse_and_logprob(row, vocab_size, targets[i], &lse, &lp);
    const double g = (double)grad_tok[i];
    const uint32_t tgt = (uint32_t)targets[i];
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      const double p = exp((double)row[v] - lse);
      grow[v] = (float)(g * ((v == tgt ? 1.0 : 0.0) - p));
    }
  }
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_token_logprobs_backward_v1(
    uint32_t batch_size, uint32_t seq_len, uint32_t vocab_size,
    const float *logits, const int32_t *targets, const uint8_t *mask,
    const float *grad_token_logprobs, float *out_grad_logits) {
  return et_grpo_token_logprobs_backward_v1_raw(
      batch_size, seq_len, vocab_size, logits, targets, mask,
      grad_token_logprobs, out_grad_logits);
}

/* ---- Public API: Log-Probability-Level GRPO Forward & Backward ---- */

static et_grpo_status grpo_logprob_loss_core(
    const et_grpo_loss_config *cfg, const float *log_pi, const float *log_old,
    const float *log_ref, const uint8_t *mask, const float *advantages,
    const float *rewards, float *out_grad_log_pi,
    et_grpo_loss_metrics *out_metrics, int need_grad) {
  size_t nt = 0u;
  et_grpo_status st = validate_loss_config_internal(cfg, 0, &nt, NULL);
  if (st != ET_GRPO_OK)
    return st;

  const size_t tok_bytes = nt * sizeof(float);
  const size_t mask_bytes = nt * sizeof(uint8_t);
  const size_t batch_bytes = (size_t)cfg->batch_size * sizeof(float);

  st = check_aligned_buf(log_pi, _Alignof(float), tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(log_old, _Alignof(float), tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(log_ref, _Alignof(float), tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(mask, _Alignof(uint8_t), mask_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(advantages, _Alignof(float), batch_bytes);
  if (st != ET_GRPO_OK)
    return st;
  if (rewards != NULL) {
    st = check_aligned_buf(rewards, _Alignof(float), batch_bytes);
    if (st != ET_GRPO_OK)
      return st;
  }
  if (need_grad) {
    st = check_aligned_buf(out_grad_log_pi, _Alignof(float), tok_bytes);
    if (st != ET_GRPO_OK)
      return st;
  }
  if (out_metrics != NULL) {
    st = check_aligned_buf(out_metrics, _Alignof(et_grpo_loss_metrics),
                           sizeof(*out_metrics));
    if (st != ET_GRPO_OK)
      return st;
  } else if (!need_grad) {
    return ET_GRPO_ERR_ARGUMENT;
  }

  const span_entry inputs[7] = {{cfg, sizeof(*cfg)},
                                {log_pi, tok_bytes},
                                {log_old, tok_bytes},
                                {log_ref, tok_bytes},
                                {mask, mask_bytes},
                                {advantages, batch_bytes},
                                {rewards, rewards ? batch_bytes : 0u}};
  const span_entry outputs[2] = {
      {out_grad_log_pi, need_grad ? tok_bytes : 0u},
      {out_metrics, out_metrics ? sizeof(*out_metrics) : 0u}};
  if (outputs_alias_inputs_or_each_other(inputs, 7u, outputs, 2u))
    return ET_GRPO_ERR_ALIAS;

  if (!all_finite_f32(log_pi, nt) || !all_finite_f32(log_old, nt) ||
      !all_finite_f32(log_ref, nt) ||
      !all_finite_f32(advantages, (size_t)cfg->batch_size) ||
      (rewards != NULL && !all_finite_f32(rewards, (size_t)cfg->batch_size)))
    return ET_GRPO_ERR_NONFINITE;

  uint32_t active_tokens = 0u;
  uint32_t active_seqs = 0u;
  if (!validate_mask_and_counts(cfg->batch_size, cfg->seq_len, mask,
                                &active_tokens, &active_seqs))
    return ET_GRPO_ERR_ARGUMENT;
  if (active_tokens == 0u)
    return ET_GRPO_ERR_EMPTY_MASK;

  float mean_reward = 0.0f;
  if (!compute_mean_reward(cfg->batch_size, rewards, &mean_reward))
    return ET_GRPO_ERR_OVERFLOW;

  const double c_low = 1.0 - (double)cfg->clip_eps_low;
  const double c_high = 1.0 + (double)cfg->clip_eps_high;
  const double kl_beta = (double)cfg->kl_beta;

  double pg_loss_acc = 0.0;
  double kl_acc = 0.0;
  uint32_t clipped_tokens = 0u;

  /* Pass 1: validate all scalars, overflow guards, and gradients. */
  for (uint32_t n = 0; n < cfg->batch_size; ++n) {
    uint32_t seq_active = 0u;
    const size_t row_base = (size_t)n * (size_t)cfg->seq_len;
    for (uint32_t t = 0; t < cfg->seq_len; ++t)
      seq_active += (uint32_t)mask[row_base + (size_t)t];
    if (seq_active == 0u)
      continue;

    const double token_weight =
        (cfg->mask_norm_mode == ET_GRPO_NORM_GLOBAL_TOKEN_MEAN)
            ? (1.0 / (double)active_tokens)
            : (1.0 / ((double)active_seqs * (double)seq_active));
    const double adv = (double)advantages[n];

    for (uint32_t t = 0; t < cfg->seq_len; ++t) {
      const size_t idx = row_base + (size_t)t;
      if (mask[idx] == 0u)
        continue;
      double neg_pg = 0.0;
      double kl = 0.0;
      double dloss = 0.0;
      int clipped = 0;
      st = eval_token_grpo_scalar((double)log_pi[idx], (double)log_old[idx],
                                  (double)log_ref[idx], adv, c_low, c_high,
                                  kl_beta, &neg_pg, &kl, &dloss, &clipped);
      if (st != ET_GRPO_OK)
        return st;
      pg_loss_acc += token_weight * neg_pg;
      kl_acc += token_weight * kl;
      clipped_tokens += (uint32_t)clipped;
      if (need_grad) {
        const float g = (float)(token_weight * dloss);
        if (!isfinite(g))
          return ET_GRPO_ERR_OVERFLOW;
      }
    }
  }

  const double total_loss_d = pg_loss_acc + kl_beta * kl_acc;
  const float loss_f32 = (float)total_loss_d;
  const float pg_loss_f32 = (float)pg_loss_acc;
  const float mean_kl_f32 = (float)kl_acc;
  const float clip_frac_f32 = (float)((double)clipped_tokens /
                                      (double)active_tokens);
  if (!isfinite(loss_f32) || !isfinite(pg_loss_f32) || !isfinite(mean_kl_f32) ||
      !isfinite(clip_frac_f32))
    return ET_GRPO_ERR_OVERFLOW;

  /* Pass 2: commit validated outputs. */
  if (need_grad) {
    for (uint32_t n = 0; n < cfg->batch_size; ++n) {
      uint32_t seq_active = 0u;
      const size_t row_base = (size_t)n * (size_t)cfg->seq_len;
      for (uint32_t t = 0; t < cfg->seq_len; ++t)
        seq_active += (uint32_t)mask[row_base + (size_t)t];
      const double token_weight =
          seq_active == 0u
              ? 0.0
              : ((cfg->mask_norm_mode == ET_GRPO_NORM_GLOBAL_TOKEN_MEAN)
                     ? (1.0 / (double)active_tokens)
                     : (1.0 / ((double)active_seqs * (double)seq_active)));
      const double adv = (double)advantages[n];
      for (uint32_t t = 0; t < cfg->seq_len; ++t) {
        const size_t idx = row_base + (size_t)t;
        if (mask[idx] == 0u) {
          out_grad_log_pi[idx] = 0.0f;
          continue;
        }
        double neg_pg = 0.0;
        double kl = 0.0;
        double dloss = 0.0;
        int clipped = 0;
        (void)eval_token_grpo_scalar((double)log_pi[idx], (double)log_old[idx],
                                     (double)log_ref[idx], adv, c_low, c_high,
                                     kl_beta, &neg_pg, &kl, &dloss, &clipped);
        out_grad_log_pi[idx] = (float)(token_weight * dloss);
      }
    }
  }

  if (out_metrics != NULL) {
    out_metrics->loss = loss_f32;
    out_metrics->pg_loss = pg_loss_f32;
    out_metrics->mean_kl = mean_kl_f32;
    out_metrics->clip_fraction = clip_frac_f32;
    out_metrics->mean_reward = mean_reward;
    out_metrics->active_tokens = active_tokens;
    out_metrics->clipped_tokens = clipped_tokens;
    out_metrics->active_seqs = active_seqs;
  }
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_logprob_loss_forward_v1(
    const et_grpo_loss_config *cfg, const float *log_pi, const float *log_old,
    const float *log_ref, const uint8_t *mask, const float *advantages,
    const float *rewards, et_grpo_loss_metrics *out_metrics) {
  return grpo_logprob_loss_core(cfg, log_pi, log_old, log_ref, mask, advantages,
                                rewards, NULL, out_metrics, 0);
}

et_grpo_status et_grpo_logprob_loss_backward_v1(
    const et_grpo_loss_config *cfg, const float *log_pi, const float *log_old,
    const float *log_ref, const uint8_t *mask, const float *advantages,
    const float *rewards, float *out_grad_log_pi,
    et_grpo_loss_metrics *out_metrics) {
  return grpo_logprob_loss_core(cfg, log_pi, log_old, log_ref, mask, advantages,
                                rewards, out_grad_log_pi, out_metrics, 1);
}

/* ---- Public API: Fused End-to-End Logits GRPO Forward & Backward ---- */

static et_grpo_status grpo_fused_loss_core(
    const et_grpo_loss_config *cfg, const void *pi_logits_raw,
    const void *targets_raw, const void *log_old_raw, const void *log_ref_raw,
    const uint8_t *mask, const void *advantages_raw, const void *rewards_raw,
    void *out_grad_logits_raw, void *out_grad_log_pi_raw, void *out_metrics_raw,
    int need_grad) {
  size_t nt = 0u;
  size_t ntv = 0u;
  et_grpo_status st = validate_loss_config_internal(cfg, 1, &nt, &ntv);
  if (st != ET_GRPO_OK)
    return st;

  const size_t logits_bytes = ntv * sizeof(float);
  const size_t targets_bytes = nt * sizeof(int32_t);
  const size_t tok_bytes = nt * sizeof(float);
  const size_t mask_bytes = nt * sizeof(uint8_t);
  const size_t batch_bytes = (size_t)cfg->batch_size * sizeof(float);

  st = check_aligned_buf(pi_logits_raw, _Alignof(float), logits_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(targets_raw, _Alignof(int32_t), targets_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(log_old_raw, _Alignof(float), tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(log_ref_raw, _Alignof(float), tok_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(mask, _Alignof(uint8_t), mask_bytes);
  if (st != ET_GRPO_OK)
    return st;
  st = check_aligned_buf(advantages_raw, _Alignof(float), batch_bytes);
  if (st != ET_GRPO_OK)
    return st;
  if (rewards_raw != NULL) {
    st = check_aligned_buf(rewards_raw, _Alignof(float), batch_bytes);
    if (st != ET_GRPO_OK)
      return st;
  }
  if (need_grad) {
    st = check_aligned_buf(out_grad_logits_raw, _Alignof(float), logits_bytes);
    if (st != ET_GRPO_OK)
      return st;
    if (out_grad_log_pi_raw != NULL) {
      st = check_aligned_buf(out_grad_log_pi_raw, _Alignof(float), tok_bytes);
      if (st != ET_GRPO_OK)
        return st;
    }
  }
  if (out_metrics_raw != NULL) {
    st = check_aligned_buf(out_metrics_raw, _Alignof(et_grpo_loss_metrics),
                           sizeof(et_grpo_loss_metrics));
    if (st != ET_GRPO_OK)
      return st;
  } else if (!need_grad) {
    return ET_GRPO_ERR_ARGUMENT;
  }

  const span_entry inputs[8] = {
      {cfg, sizeof(*cfg)},
      {pi_logits_raw, logits_bytes},
      {targets_raw, targets_bytes},
      {log_old_raw, tok_bytes},
      {log_ref_raw, tok_bytes},
      {mask, mask_bytes},
      {advantages_raw, batch_bytes},
      {rewards_raw, rewards_raw ? batch_bytes : 0u}};
  const span_entry outputs[3] = {
      {out_grad_logits_raw, need_grad ? logits_bytes : 0u},
      {out_grad_log_pi_raw,
       (need_grad && out_grad_log_pi_raw) ? tok_bytes : 0u},
      {out_metrics_raw, out_metrics_raw ? sizeof(et_grpo_loss_metrics) : 0u}};
  if (outputs_alias_inputs_or_each_other(inputs, 8u, outputs, 3u))
    return ET_GRPO_ERR_ALIAS;

  const float *pi_logits = (const float *)pi_logits_raw;
  const int32_t *targets = (const int32_t *)targets_raw;
  const float *log_old = (const float *)log_old_raw;
  const float *log_ref = (const float *)log_ref_raw;
  const float *advantages = (const float *)advantages_raw;
  const float *rewards = (const float *)rewards_raw;
  float *out_grad_logits = (float *)out_grad_logits_raw;
  float *out_grad_log_pi = (float *)out_grad_log_pi_raw;
  et_grpo_loss_metrics *out_metrics = (et_grpo_loss_metrics *)out_metrics_raw;

  if (!all_finite_f32(pi_logits, ntv) || !all_finite_f32(log_old, nt) ||
      !all_finite_f32(log_ref, nt) ||
      !all_finite_f32(advantages, (size_t)cfg->batch_size) ||
      (rewards != NULL && !all_finite_f32(rewards, (size_t)cfg->batch_size)))
    return ET_GRPO_ERR_NONFINITE;

  for (size_t i = 0; i < nt; ++i) {
    if (targets[i] < 0 || (uint32_t)targets[i] >= cfg->vocab_size)
      return ET_GRPO_ERR_INDEX;
  }

  uint32_t active_tokens = 0u;
  uint32_t active_seqs = 0u;
  if (!validate_mask_and_counts(cfg->batch_size, cfg->seq_len, mask,
                                &active_tokens, &active_seqs))
    return ET_GRPO_ERR_ARGUMENT;
  if (active_tokens == 0u)
    return ET_GRPO_ERR_EMPTY_MASK;

  float mean_reward = 0.0f;
  if (!compute_mean_reward(cfg->batch_size, rewards, &mean_reward))
    return ET_GRPO_ERR_OVERFLOW;

  const double c_low = 1.0 - (double)cfg->clip_eps_low;
  const double c_high = 1.0 + (double)cfg->clip_eps_high;
  const double kl_beta = (double)cfg->kl_beta;

  double pg_loss_acc = 0.0;
  double kl_acc = 0.0;
  uint32_t clipped_tokens = 0u;

  /* Pass 1: validate every token's log-softmax, ratio, KL, and gradients
   * before writing any output buffer. */
  for (uint32_t n = 0; n < cfg->batch_size; ++n) {
    uint32_t seq_active = 0u;
    const size_t row_base = (size_t)n * (size_t)cfg->seq_len;
    for (uint32_t t = 0; t < cfg->seq_len; ++t)
      seq_active += (uint32_t)mask[row_base + (size_t)t];
    if (seq_active == 0u)
      continue;

    const double token_weight =
        (cfg->mask_norm_mode == ET_GRPO_NORM_GLOBAL_TOKEN_MEAN)
            ? (1.0 / (double)active_tokens)
            : (1.0 / ((double)active_seqs * (double)seq_active));
    const double adv = (double)advantages[n];

    for (uint32_t t = 0; t < cfg->seq_len; ++t) {
      const size_t idx = row_base + (size_t)t;
      if (mask[idx] == 0u)
        continue;
      const float *row = pi_logits + idx * (size_t)cfg->vocab_size;
      double lse = 0.0;
      double log_pi = 0.0;
      if (!compute_row_lse_and_logprob(row, cfg->vocab_size, targets[idx], &lse,
                                       &log_pi))
        return ET_GRPO_ERR_OVERFLOW;

      double neg_pg = 0.0;
      double kl = 0.0;
      double dloss = 0.0;
      int clipped = 0;
      st = eval_token_grpo_scalar(log_pi, (double)log_old[idx],
                                  (double)log_ref[idx], adv, c_low, c_high,
                                  kl_beta, &neg_pg, &kl, &dloss, &clipped);
      if (st != ET_GRPO_OK)
        return st;
      pg_loss_acc += token_weight * neg_pg;
      kl_acc += token_weight * kl;
      clipped_tokens += (uint32_t)clipped;

      if (need_grad) {
        const double g_lp = token_weight * dloss;
        if (!isfinite((float)g_lp))
          return ET_GRPO_ERR_OVERFLOW;
        const uint32_t tgt = (uint32_t)targets[idx];
        for (uint32_t v = 0u; v < cfg->vocab_size; ++v) {
          const double p = exp((double)row[v] - lse);
          const float gv = (float)(g_lp * ((v == tgt ? 1.0 : 0.0) - p));
          if (!isfinite(gv))
            return ET_GRPO_ERR_OVERFLOW;
        }
      }
    }
  }

  const double total_loss_d = pg_loss_acc + kl_beta * kl_acc;
  const float loss_f32 = (float)total_loss_d;
  const float pg_loss_f32 = (float)pg_loss_acc;
  const float mean_kl_f32 = (float)kl_acc;
  const float clip_frac_f32 = (float)((double)clipped_tokens /
                                      (double)active_tokens);
  if (!isfinite(loss_f32) || !isfinite(pg_loss_f32) || !isfinite(mean_kl_f32) ||
      !isfinite(clip_frac_f32))
    return ET_GRPO_ERR_OVERFLOW;

  /* Pass 2: commit validated gradients and metrics. */
  if (need_grad) {
    for (uint32_t n = 0; n < cfg->batch_size; ++n) {
      uint32_t seq_active = 0u;
      const size_t row_base = (size_t)n * (size_t)cfg->seq_len;
      for (uint32_t t = 0; t < cfg->seq_len; ++t)
        seq_active += (uint32_t)mask[row_base + (size_t)t];
      const double token_weight =
          seq_active == 0u
              ? 0.0
              : ((cfg->mask_norm_mode == ET_GRPO_NORM_GLOBAL_TOKEN_MEAN)
                     ? (1.0 / (double)active_tokens)
                     : (1.0 / ((double)active_seqs * (double)seq_active)));
      const double adv = (double)advantages[n];

      for (uint32_t t = 0; t < cfg->seq_len; ++t) {
        const size_t idx = row_base + (size_t)t;
        float *grow = out_grad_logits + idx * (size_t)cfg->vocab_size;
        if (mask[idx] == 0u) {
          for (uint32_t v = 0u; v < cfg->vocab_size; ++v)
            grow[v] = 0.0f;
          if (out_grad_log_pi != NULL)
            out_grad_log_pi[idx] = 0.0f;
          continue;
        }
        const float *row = pi_logits + idx * (size_t)cfg->vocab_size;
        double lse = 0.0;
        double log_pi = 0.0;
        (void)compute_row_lse_and_logprob(row, cfg->vocab_size, targets[idx],
                                          &lse, &log_pi);
        double neg_pg = 0.0;
        double kl = 0.0;
        double dloss = 0.0;
        int clipped = 0;
        (void)eval_token_grpo_scalar(log_pi, (double)log_old[idx],
                                     (double)log_ref[idx], adv, c_low, c_high,
                                     kl_beta, &neg_pg, &kl, &dloss, &clipped);
        const double g_lp = token_weight * dloss;
        if (out_grad_log_pi != NULL)
          out_grad_log_pi[idx] = (float)g_lp;
        const uint32_t tgt = (uint32_t)targets[idx];
        for (uint32_t v = 0u; v < cfg->vocab_size; ++v) {
          const double p = exp((double)row[v] - lse);
          grow[v] = (float)(g_lp * ((v == tgt ? 1.0 : 0.0) - p));
        }
      }
    }
  }

  if (out_metrics != NULL) {
    out_metrics->loss = loss_f32;
    out_metrics->pg_loss = pg_loss_f32;
    out_metrics->mean_kl = mean_kl_f32;
    out_metrics->clip_fraction = clip_frac_f32;
    out_metrics->mean_reward = mean_reward;
    out_metrics->active_tokens = active_tokens;
    out_metrics->clipped_tokens = clipped_tokens;
    out_metrics->active_seqs = active_seqs;
  }
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_loss_forward_v1_raw(
    const et_grpo_loss_config *cfg, const void *pi_logits, const void *targets,
    const void *log_old, const void *log_ref, const uint8_t *mask,
    const void *advantages, const void *rewards, void *out_metrics) {
  return grpo_fused_loss_core(cfg, pi_logits, targets, log_old, log_ref, mask,
                              advantages, rewards, NULL, NULL, out_metrics, 0);
}

et_grpo_status et_grpo_loss_forward_v1(
    const et_grpo_loss_config *cfg, const float *pi_logits,
    const int32_t *targets, const float *log_old, const float *log_ref,
    const uint8_t *mask, const float *advantages, const float *rewards,
    et_grpo_loss_metrics *out_metrics) {
  return et_grpo_loss_forward_v1_raw(cfg, pi_logits, targets, log_old, log_ref,
                                     mask, advantages, rewards, out_metrics);
}

et_grpo_status et_grpo_loss_backward_v1_raw(
    const et_grpo_loss_config *cfg, const void *pi_logits, const void *targets,
    const void *log_old, const void *log_ref, const uint8_t *mask,
    const void *advantages, const void *rewards, void *out_grad_logits,
    void *out_grad_log_pi, void *out_metrics) {
  return grpo_fused_loss_core(cfg, pi_logits, targets, log_old, log_ref, mask,
                              advantages, rewards, out_grad_logits,
                              out_grad_log_pi, out_metrics, 1);
}

et_grpo_status et_grpo_loss_backward_v1(
    const et_grpo_loss_config *cfg, const float *pi_logits,
    const int32_t *targets, const float *log_old, const float *log_ref,
    const uint8_t *mask, const float *advantages, const float *rewards,
    float *out_grad_logits, float *out_grad_log_pi,
    et_grpo_loss_metrics *out_metrics) {
  return et_grpo_loss_backward_v1_raw(
      cfg, pi_logits, targets, log_old, log_ref, mask, advantages, rewards,
      out_grad_logits, out_grad_log_pi, out_metrics);
}

/* ---- Public API: Deterministic RLVR Verifier Harness ---- */

static int is_ascii_ws(char c) {
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static void trim_ascii_slice(const char *buf, size_t *io_start,
                             size_t *io_end) {
  size_t s = *io_start;
  size_t e = *io_end;
  while (s < e && is_ascii_ws(buf[s]))
    ++s;
  while (e > s && is_ascii_ws(buf[e - 1u]))
    --e;
  *io_start = s;
  *io_end = e;
}

static int slice_starts_with(const char *buf, size_t pos, size_t end,
                             const char *lit, size_t lit_len) {
  if (end - pos < lit_len)
    return 0;
  return memcmp(buf + pos, lit, lit_len) == 0;
}

static int find_literal(const char *buf, size_t start, size_t end,
                        const char *lit, size_t lit_len, size_t *out_pos) {
  if (lit_len == 0u || end < start || end - start < lit_len)
    return 0;
  const size_t limit = end - lit_len;
  for (size_t p = start; p <= limit; ++p) {
    if (memcmp(buf + p, lit, lit_len) == 0) {
      if (out_pos != NULL)
        *out_pos = p;
      return 1;
    }
  }
  return 0;
}

et_grpo_status et_grpo_verify_exact_match_v1(const char *completion,
                                             size_t completion_len,
                                             const char *expected_answer,
                                             size_t expected_len,
                                             uint32_t flags,
                                             float *out_reward) {
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;
  if (completion == NULL || expected_answer == NULL || out_reward == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if ((flags & ~ET_GRPO_VERIFY_TRIM_WS) != 0u)
    return ET_GRPO_ERR_ARGUMENT;
  if (expected_len == 0u || completion_len > ET_GRPO_MAX_VERIFY_BYTES ||
      expected_len > ET_GRPO_MAX_VERIFY_BYTES)
    return ET_GRPO_ERR_SHAPE;
  if (!aligned_span(completion, 1u, completion_len) ||
      !aligned_span(expected_answer, 1u, expected_len) ||
      !aligned_span(out_reward, _Alignof(float), sizeof(float)))
    return ET_GRPO_ERR_ALIGNMENT;

  const span_entry inputs[2] = {{completion, completion_len},
                                {expected_answer, expected_len}};
  const span_entry outputs[1] = {{out_reward, sizeof(float)}};
  if (outputs_alias_inputs_or_each_other(inputs, 2u, outputs, 1u))
    return ET_GRPO_ERR_ALIAS;

  size_t c_start = 0u;
  size_t c_end = completion_len;
  size_t e_start = 0u;
  size_t e_end = expected_len;
  if ((flags & ET_GRPO_VERIFY_TRIM_WS) != 0u) {
    trim_ascii_slice(completion, &c_start, &c_end);
    trim_ascii_slice(expected_answer, &e_start, &e_end);
    if (e_start == e_end)
      return ET_GRPO_ERR_ARGUMENT;
  }

  const size_t c_len = c_end - c_start;
  const size_t e_len = e_end - e_start;
  const int matched =
      (c_len == e_len) &&
      (memcmp(completion + c_start, expected_answer + e_start, e_len) == 0);
  *out_reward = matched ? 1.0f : 0.0f;
  return ET_GRPO_OK;
}

et_grpo_status et_grpo_verify_format_and_answer_v1(
    const char *completion, size_t completion_len, const char *expected_answer,
    size_t expected_len, const et_grpo_verify_format_config *cfg,
    float *out_reward, et_grpo_verify_detail *out_detail) {
  if (!fp_environment_supported())
    return ET_GRPO_ERR_FP_ENV;
  if (completion == NULL || expected_answer == NULL || cfg == NULL ||
      out_reward == NULL)
    return ET_GRPO_ERR_ARGUMENT;
  if (!aligned_span(cfg, _Alignof(et_grpo_verify_format_config),
                    sizeof(*cfg)) ||
      !aligned_span(out_reward, _Alignof(float), sizeof(float)) ||
      (out_detail != NULL &&
       !aligned_span(out_detail, _Alignof(et_grpo_verify_detail),
                     sizeof(*out_detail))) ||
      !aligned_span(completion, 1u, completion_len) ||
      !aligned_span(expected_answer, 1u, expected_len))
    return ET_GRPO_ERR_ALIGNMENT;
  if (cfg->reserved != 0u || (cfg->flags & ~ET_GRPO_VERIFY_TRIM_WS) != 0u)
    return ET_GRPO_ERR_ARGUMENT;
  if (expected_len == 0u || completion_len > ET_GRPO_MAX_VERIFY_BYTES ||
      expected_len > ET_GRPO_MAX_VERIFY_BYTES)
    return ET_GRPO_ERR_SHAPE;
  if (!isfinite(cfg->format_weight) || !isfinite(cfg->answer_weight))
    return ET_GRPO_ERR_NONFINITE;
  if (!(cfg->format_weight >= 0.0f) || !(cfg->answer_weight >= 0.0f))
    return ET_GRPO_ERR_ARGUMENT;
  const double max_reward_d =
      (double)cfg->format_weight + (double)cfg->answer_weight;
  if (!isfinite((float)max_reward_d) || !(max_reward_d > 0.0))
    return ET_GRPO_ERR_ARGUMENT;

  const span_entry inputs[3] = {{completion, completion_len},
                                {expected_answer, expected_len},
                                {cfg, sizeof(*cfg)}};
  const span_entry outputs[2] = {
      {out_reward, sizeof(float)},
      {out_detail, out_detail ? sizeof(*out_detail) : 0u}};
  if (outputs_alias_inputs_or_each_other(inputs, 3u, outputs, 2u))
    return ET_GRPO_ERR_ALIAS;

  size_t e_start = 0u;
  size_t e_end = expected_len;
  const int trim_ws = (cfg->flags & ET_GRPO_VERIFY_TRIM_WS) != 0u;
  if (trim_ws) {
    trim_ascii_slice(expected_answer, &e_start, &e_end);
    if (e_start == e_end)
      return ET_GRPO_ERR_ARGUMENT;
  }

  static const char kThinkOpen[] = "<think>";
  static const char kThinkClose[] = "</think>";
  static const char kAnsOpen[] = "<answer>";
  static const char kAnsClose[] = "</answer>";
  const size_t kThinkOpenLen = 7u;
  const size_t kThinkCloseLen = 8u;
  const size_t kAnsOpenLen = 8u;
  const size_t kAnsCloseLen = 9u;

  uint32_t format_matched = 0u;
  uint32_t answer_matched = 0u;
  uint64_t ans_off = 0u;
  uint64_t ans_len = 0u;

  size_t pos = 0u;
  size_t end = completion_len;
  if (trim_ws)
    trim_ascii_slice(completion, &pos, &end);

  if (slice_starts_with(completion, pos, end, kThinkOpen, kThinkOpenLen)) {
    const size_t think_start = pos + kThinkOpenLen;
    size_t think_end = 0u;
    if (find_literal(completion, think_start, end, kThinkClose, kThinkCloseLen,
                     &think_end)) {
      size_t t_lo = think_start;
      size_t t_hi = think_end;
      if (trim_ws)
        trim_ascii_slice(completion, &t_lo, &t_hi);
      const int think_clean =
          (t_hi > t_lo) &&
          !find_literal(completion, think_start, think_end, kThinkOpen,
                        kThinkOpenLen, NULL) &&
          !find_literal(completion, think_start, think_end, kAnsOpen,
                        kAnsOpenLen, NULL) &&
          !find_literal(completion, think_start, think_end, kAnsClose,
                        kAnsCloseLen, NULL);
      if (think_clean) {
        size_t cur = think_end + kThinkCloseLen;
        while (cur < end && is_ascii_ws(completion[cur]))
          ++cur;
        if (slice_starts_with(completion, cur, end, kAnsOpen, kAnsOpenLen)) {
          const size_t ans_start = cur + kAnsOpenLen;
          size_t ans_end = 0u;
          if (find_literal(completion, ans_start, end, kAnsClose, kAnsCloseLen,
                           &ans_end) &&
              ans_end + kAnsCloseLen == end) {
            const int ans_clean =
                !find_literal(completion, ans_start, ans_end, kThinkOpen,
                              kThinkOpenLen, NULL) &&
                !find_literal(completion, ans_start, ans_end, kThinkClose,
                              kThinkCloseLen, NULL) &&
                !find_literal(completion, ans_start, ans_end, kAnsOpen,
                              kAnsOpenLen, NULL);
            size_t a_lo = ans_start;
            size_t a_hi = ans_end;
            if (trim_ws)
              trim_ascii_slice(completion, &a_lo, &a_hi);
            if (ans_clean && a_hi > a_lo) {
              format_matched = 1u;
              ans_off = (uint64_t)a_lo;
              ans_len = (uint64_t)(a_hi - a_lo);
              const size_t e_len = e_end - e_start;
              if ((a_hi - a_lo) == e_len &&
                  memcmp(completion + a_lo, expected_answer + e_start, e_len) ==
                      0) {
                answer_matched = 1u;
              }
            }
          }
        }
      }
    }
  }

  const float reward =
      (float)((format_matched ? (double)cfg->format_weight : 0.0) +
              (answer_matched ? (double)cfg->answer_weight : 0.0));
  *out_reward = reward;
  if (out_detail != NULL) {
    out_detail->format_matched = format_matched;
    out_detail->answer_matched = answer_matched;
    out_detail->total_reward = reward;
    out_detail->reserved = 0u;
    out_detail->answer_offset = ans_off;
    out_detail->answer_length = ans_len;
  }
  return ET_GRPO_OK;
}
