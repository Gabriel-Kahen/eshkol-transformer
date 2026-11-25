#include "eshkol_transformer/spec_decode_abi.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#else
#include <fenv.h>
#endif

#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "spec_decode requires 8-bit bytes and at most 64-bit size_t");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   sizeof(float) == 4 && DBL_MANT_DIG == 53 &&
                   DBL_MAX_EXP == 1024 && sizeof(double) == sizeof(uint64_t),
               "spec_decode requires IEEE-754 binary32 and binary64");

typedef struct byte_span {
  const void *ptr;
  size_t bytes;
} byte_span;

static uint32_t f32_bits(float value) {
  uint32_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint64_t f64_bits(double value) {
  uint64_t bits = 0u;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int is_finite_f32(float value) {
  return (f32_bits(value) & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int is_finite_f64(double value) {
  return (f64_bits(value) & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

static int is_signbit_f64(double value) {
  return (f64_bits(value) & UINT64_C(0x8000000000000000)) != 0u;
}

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87 = 0u;
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  /* Round to nearest, mask all x87/SSE exceptions, FTZ=0, DAZ=0. */
  return (x87 & 0x0c3fu) == 0x003fu && (_mm_getcsr() & 0xffc0u) == 0x1f80u;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

static int is_aligned_ptr(const void *ptr, size_t alignment) {
  return (((uintptr_t)ptr) % alignment) == 0u;
}

static int range_valid(const void *ptr, size_t bytes, uintptr_t *lo,
                       uintptr_t *hi) {
  const uintptr_t start = (uintptr_t)ptr;
  if (bytes > UINTPTR_MAX - start)
    return 0;
  *lo = start;
  *hi = start + bytes;
  return 1;
}

static int spans_pairwise_disjoint(const byte_span *spans, size_t count) {
  uintptr_t lo[8];
  uintptr_t hi[8];
  if (count > 8u)
    return 0;
  for (size_t i = 0u; i < count; ++i) {
    if (!range_valid(spans[i].ptr, spans[i].bytes, &lo[i], &hi[i]))
      return 0;
  }
  for (size_t i = 0u; i < count; ++i) {
    for (size_t j = i + 1u; j < count; ++j) {
      if (lo[i] < hi[j] && lo[j] < hi[i])
        return 0;
    }
  }
  return 1;
}

static et_spec_status validate_vocab(uint32_t vocab_size) {
  if (vocab_size < ET_SPEC_MIN_VOCAB || vocab_size > ET_SPEC_MAX_VOCAB)
    return ET_SPEC_ERR_BOUNDS;
  return ET_SPEC_OK;
}

static et_spec_status validate_horizon(uint32_t horizon) {
  if (horizon < ET_SPEC_MIN_HORIZON || horizon > ET_SPEC_MAX_HORIZON)
    return ET_SPEC_ERR_BOUNDS;
  return ET_SPEC_OK;
}

static et_spec_status validate_eos(uint32_t eos_token_id, uint32_t vocab_size) {
  if (eos_token_id != ET_SPEC_NO_EOS && eos_token_id >= vocab_size)
    return ET_SPEC_ERR_TOKEN_RANGE;
  return ET_SPEC_OK;
}

static et_spec_status validate_params(const et_spec_sampling_params_v1 *params,
                                      uint32_t vocab_size) {
  if (!is_finite_f64(params->temperature) || !is_finite_f64(params->top_p))
    return ET_SPEC_ERR_NONFINITE;
  if (is_signbit_f64(params->temperature) || params->temperature < 0.0)
    return ET_SPEC_ERR_PARAM;
  if (params->top_k < 1u || params->top_k > vocab_size)
    return ET_SPEC_ERR_PARAM;
  if (is_signbit_f64(params->top_p) || params->top_p <= 0.0 ||
      params->top_p > 1.0)
    return ET_SPEC_ERR_PARAM;
  return validate_eos(params->eos_token_id, vocab_size);
}

static et_spec_status validate_logits_finite(const float *logits,
                                             size_t count) {
  for (size_t i = 0u; i < count; ++i) {
    if (!is_finite_f32(logits[i]))
      return ET_SPEC_ERR_NONFINITE;
  }
  return ET_SPEC_OK;
}

static et_spec_status validate_scaled_rows_finite(const float *rows,
                                                  uint32_t row_count,
                                                  uint32_t vocab_size,
                                                  double temperature) {
  if (temperature == 0.0)
    return ET_SPEC_OK;
  for (uint32_t r = 0u; r < row_count; ++r) {
    const float *row = rows + ((size_t)r * (size_t)vocab_size);
    double max_logit = (double)row[0];
    for (uint32_t v = 1u; v < vocab_size; ++v) {
      const double val = (double)row[v];
      if (val > max_logit)
        max_logit = val;
    }
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      const double scaled = ((double)row[v] - max_logit) / temperature;
      if (!is_finite_f64(scaled))
        return ET_SPEC_ERR_NONFINITE;
    }
  }
  return ET_SPEC_OK;
}

static uint32_t argmax_lowest_index(const float *logits, uint32_t vocab_size) {
  uint32_t best = 0u;
  double best_val = (double)logits[0];
  for (uint32_t v = 1u; v < vocab_size; ++v) {
    const double val = (double)logits[v];
    if (val > best_val) {
      best_val = val;
      best = v;
    }
  }
  return best;
}

static int prob_index_before(const double *probs, uint32_t a, uint32_t b) {
  const double pa = probs[a];
  const double pb = probs[b];
  if (pa > pb)
    return 1;
  if (pa < pb)
    return 0;
  return a < b;
}

static void sort_indices_desc(const double *probs, uint32_t vocab_size,
                              uint32_t *order, uint32_t *scratch) {
  for (uint32_t i = 0u; i < vocab_size; ++i)
    order[i] = i;

  uint32_t *src = order;
  uint32_t *dst = scratch;
  for (uint32_t width = 1u; width < vocab_size; width <<= 1u) {
    for (uint32_t left = 0u; left < vocab_size; left += (width << 1u)) {
      uint32_t mid = left + width;
      if (mid > vocab_size)
        mid = vocab_size;
      uint32_t right = left + (width << 1u);
      if (right > vocab_size)
        right = vocab_size;
      uint32_t i = left;
      uint32_t j = mid;
      uint32_t k = left;
      while (i < mid && j < right) {
        if (prob_index_before(probs, src[i], src[j]))
          dst[k++] = src[i++];
        else
          dst[k++] = src[j++];
      }
      while (i < mid)
        dst[k++] = src[i++];
      while (j < right)
        dst[k++] = src[j++];
    }
    uint32_t *swap = src;
    src = dst;
    dst = swap;
  }

  if (src != order)
    memcpy(order, src, (size_t)vocab_size * sizeof(uint32_t));
}

static et_spec_status compute_probs_internal(
    const float *logits, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params, double *probs, uint32_t *order,
    uint32_t *sort_tmp) {
  if (params->temperature == 0.0) {
    const uint32_t best = argmax_lowest_index(logits, vocab_size);
    memset(probs, 0, (size_t)vocab_size * sizeof(double));
    probs[best] = 1.0;
    return ET_SPEC_OK;
  }

  double max_logit = (double)logits[0];
  for (uint32_t v = 1u; v < vocab_size; ++v) {
    const double val = (double)logits[v];
    if (val > max_logit)
      max_logit = val;
  }

  double sum_exp = 0.0;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    const double scaled = ((double)logits[v] - max_logit) / params->temperature;
    if (!is_finite_f64(scaled))
      return ET_SPEC_ERR_NONFINITE;
    probs[v] = scaled;
    sum_exp += exp(scaled);
  }
  if (!is_finite_f64(sum_exp) || sum_exp <= 0.0)
    return ET_SPEC_ERR_NONFINITE;

  const double log_z = log(sum_exp);
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    const double p = exp(probs[v] - log_z);
    if (!is_finite_f64(p))
      return ET_SPEC_ERR_NONFINITE;
    probs[v] = p;
  }

  /* Sort indices by descending probability with lowest-index tie-breaking. */
  sort_indices_desc(probs, vocab_size, order, sort_tmp);

  /* Top-k truncation and renormalization. */
  if (params->top_k < vocab_size) {
    for (uint32_t r = params->top_k; r < vocab_size; ++r)
      probs[order[r]] = 0.0;
  }
  double topk_sum = 0.0;
  for (uint32_t v = 0u; v < vocab_size; ++v)
    topk_sum += probs[v];
  if (!is_finite_f64(topk_sum) || topk_sum <= 0.0)
    return ET_SPEC_ERR_NONFINITE;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    if (probs[v] > 0.0)
      probs[v] /= topk_sum;
  }

  /* Top-p (nucleus) shortest prefix keeping at least the top-1 token. */
  if (params->top_p < 1.0) {
    double cum = 0.0;
    uint32_t keep = params->top_k;
    for (uint32_t r = 0u; r < params->top_k; ++r) {
      cum += probs[order[r]];
      if (cum >= params->top_p) {
        keep = r + 1u;
        break;
      }
    }
    for (uint32_t r = keep; r < params->top_k; ++r)
      probs[order[r]] = 0.0;

    double final_sum = 0.0;
    for (uint32_t v = 0u; v < vocab_size; ++v)
      final_sum += probs[v];
    if (!is_finite_f64(final_sum) || final_sum <= 0.0)
      return ET_SPEC_ERR_NONFINITE;
    for (uint32_t v = 0u; v < vocab_size; ++v) {
      if (probs[v] > 0.0)
        probs[v] /= final_sum;
    }
  }

  return ET_SPEC_OK;
}

static uint64_t splitmix64_mix(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27u)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31u);
}

static double splitmix64_uniform01(uint64_t seed, uint64_t counter) {
  const uint64_t input = seed + (counter * UINT64_C(0x9e3779b97f4a7c15));
  const uint64_t bits = splitmix64_mix(input);
  return (double)(bits >> 11u) * 0x1.0p-53;
}

static uint32_t sample_categorical(const double *probs, uint32_t vocab_size,
                                   double u) {
  double cdf = 0.0;
  uint32_t last_nonzero = 0u;
  int found_nonzero = 0;
  for (uint32_t v = 0u; v < vocab_size; ++v) {
    const double pv = probs[v];
    if (pv > 0.0) {
      last_nonzero = v;
      found_nonzero = 1;
      cdf += pv;
      if (u < cdf)
        return v;
    }
  }
  return found_nonzero ? last_nonzero : 0u;
}

et_spec_status et_spec_logits_to_probs_v1(
    const float *logits, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params, double *probs_out) {
  if (!fp_environment_supported())
    return ET_SPEC_ERR_FP_ENV;
  if (logits == NULL || params == NULL || probs_out == NULL)
    return ET_SPEC_ERR_NULL;
  if (!is_aligned_ptr(logits, _Alignof(float)) ||
      !is_aligned_ptr(params, _Alignof(et_spec_sampling_params_v1)) ||
      !is_aligned_ptr(probs_out, _Alignof(double)))
    return ET_SPEC_ERR_ALIGNMENT;

  et_spec_status st = validate_vocab(vocab_size);
  if (st != ET_SPEC_OK)
    return st;

  const byte_span spans[] = {
      {logits, (size_t)vocab_size * sizeof(float)},
      {params, sizeof(*params)},
      {probs_out, (size_t)vocab_size * sizeof(double)},
  };
  if (!spans_pairwise_disjoint(spans, 3u))
    return ET_SPEC_ERR_ALIAS;

  st = validate_params(params, vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_logits_finite(logits, (size_t)vocab_size);
  if (st != ET_SPEC_OK)
    return st;

  double local_probs[ET_SPEC_MAX_VOCAB];
  uint32_t order[ET_SPEC_MAX_VOCAB];
  uint32_t sort_tmp[ET_SPEC_MAX_VOCAB];
  st = compute_probs_internal(logits, vocab_size, params, local_probs, order,
                              sort_tmp);
  if (st != ET_SPEC_OK)
    return st;

  memcpy(probs_out, local_probs, (size_t)vocab_size * sizeof(double));
  return ET_SPEC_OK;
}

et_spec_status et_spec_sample_token_v1(
    const float *logits, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params,
    const et_spec_rng_state_v1 *rng_in, uint32_t *out_token,
    et_spec_rng_state_v1 *out_rng) {
  if (!fp_environment_supported())
    return ET_SPEC_ERR_FP_ENV;
  if (logits == NULL || params == NULL || rng_in == NULL || out_token == NULL ||
      out_rng == NULL)
    return ET_SPEC_ERR_NULL;
  if (!is_aligned_ptr(logits, _Alignof(float)) ||
      !is_aligned_ptr(params, _Alignof(et_spec_sampling_params_v1)) ||
      !is_aligned_ptr(rng_in, _Alignof(et_spec_rng_state_v1)) ||
      !is_aligned_ptr(out_token, _Alignof(uint32_t)) ||
      !is_aligned_ptr(out_rng, _Alignof(et_spec_rng_state_v1)))
    return ET_SPEC_ERR_ALIGNMENT;

  et_spec_status st = validate_vocab(vocab_size);
  if (st != ET_SPEC_OK)
    return st;

  const byte_span spans[] = {
      {logits, (size_t)vocab_size * sizeof(float)},
      {params, sizeof(*params)},
      {rng_in, sizeof(*rng_in)},
      {out_token, sizeof(*out_token)},
      {out_rng, sizeof(*out_rng)},
  };
  if (!spans_pairwise_disjoint(spans, 5u))
    return ET_SPEC_ERR_ALIAS;

  st = validate_params(params, vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  if (params->temperature > 0.0 && rng_in->counter == UINT64_MAX)
    return ET_SPEC_ERR_BOUNDS;
  st = validate_logits_finite(logits, (size_t)vocab_size);
  if (st != ET_SPEC_OK)
    return st;

  uint32_t token = 0u;
  et_spec_rng_state_v1 next_rng = *rng_in;
  if (params->temperature == 0.0) {
    token = argmax_lowest_index(logits, vocab_size);
  } else {
    double local_probs[ET_SPEC_MAX_VOCAB];
    uint32_t order[ET_SPEC_MAX_VOCAB];
    uint32_t sort_tmp[ET_SPEC_MAX_VOCAB];
    st = compute_probs_internal(logits, vocab_size, params, local_probs, order,
                                sort_tmp);
    if (st != ET_SPEC_OK)
      return st;
    const double u = splitmix64_uniform01(rng_in->seed, rng_in->counter);
    token = sample_categorical(local_probs, vocab_size, u);
    next_rng.counter = rng_in->counter + 1u;
  }

  *out_token = token;
  *out_rng = next_rng;
  return ET_SPEC_OK;
}

et_spec_status et_spec_verify_greedy_v1(
    const uint32_t *draft_tokens, const float *target_logits, uint32_t horizon,
    uint32_t vocab_size, uint32_t eos_token_id, uint32_t *out_tokens,
    uint32_t out_capacity, et_spec_verify_result_v1 *out_result) {
  if (!fp_environment_supported())
    return ET_SPEC_ERR_FP_ENV;
  if (draft_tokens == NULL || target_logits == NULL || out_tokens == NULL ||
      out_result == NULL)
    return ET_SPEC_ERR_NULL;
  if (!is_aligned_ptr(draft_tokens, _Alignof(uint32_t)) ||
      !is_aligned_ptr(target_logits, _Alignof(float)) ||
      !is_aligned_ptr(out_tokens, _Alignof(uint32_t)) ||
      !is_aligned_ptr(out_result, _Alignof(et_spec_verify_result_v1)))
    return ET_SPEC_ERR_ALIGNMENT;

  et_spec_status st = validate_horizon(horizon);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_vocab(vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  if (out_capacity < horizon + 1u ||
      (uint64_t)out_capacity > (uint64_t)SIZE_MAX / sizeof(uint32_t))
    return ET_SPEC_ERR_BOUNDS;

  const size_t target_rows = (size_t)horizon + 1u;
  const size_t target_count = target_rows * (size_t)vocab_size;
  const byte_span spans[] = {
      {draft_tokens, (size_t)horizon * sizeof(uint32_t)},
      {target_logits, target_count * sizeof(float)},
      {out_tokens, (size_t)out_capacity * sizeof(uint32_t)},
      {out_result, sizeof(*out_result)},
  };
  if (!spans_pairwise_disjoint(spans, 4u))
    return ET_SPEC_ERR_ALIAS;

  st = validate_eos(eos_token_id, vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  for (uint32_t i = 0u; i < horizon; ++i) {
    if (draft_tokens[i] >= vocab_size)
      return ET_SPEC_ERR_TOKEN_RANGE;
  }
  st = validate_logits_finite(target_logits, target_count);
  if (st != ET_SPEC_OK)
    return st;

  uint32_t local_tokens[ET_SPEC_MAX_HORIZON + 1u];
  et_spec_verify_result_v1 local_result;
  memset(&local_result, 0, sizeof(local_result));

  for (uint32_t i = 0u; i < horizon; ++i) {
    const float *row = target_logits + ((size_t)i * (size_t)vocab_size);
    const uint32_t target_tok = argmax_lowest_index(row, vocab_size);
    local_tokens[local_result.emitted_count++] = target_tok;
    if (draft_tokens[i] == target_tok) {
      local_result.accepted_count++;
      if (eos_token_id != ET_SPEC_NO_EOS && target_tok == eos_token_id) {
        local_result.eos_hit = 1u;
        break;
      }
    } else {
      local_result.rejected = 1u;
      if (eos_token_id != ET_SPEC_NO_EOS && target_tok == eos_token_id)
        local_result.eos_hit = 1u;
      break;
    }
  }

  if (!local_result.rejected && !local_result.eos_hit) {
    const float *bonus_row =
        target_logits + ((size_t)horizon * (size_t)vocab_size);
    const uint32_t bonus_tok = argmax_lowest_index(bonus_row, vocab_size);
    local_tokens[local_result.emitted_count++] = bonus_tok;
    if (eos_token_id != ET_SPEC_NO_EOS && bonus_tok == eos_token_id)
      local_result.eos_hit = 1u;
  }

  memcpy(out_tokens, local_tokens,
         (size_t)local_result.emitted_count * sizeof(uint32_t));
  *out_result = local_result;
  return ET_SPEC_OK;
}

et_spec_status et_spec_verify_sample_v1(
    const uint32_t *draft_tokens, const float *draft_logits,
    const float *target_logits, uint32_t horizon, uint32_t vocab_size,
    const et_spec_sampling_params_v1 *params,
    const et_spec_rng_state_v1 *rng_in, uint32_t *out_tokens,
    uint32_t out_capacity, et_spec_verify_result_v1 *out_result) {
  if (!fp_environment_supported())
    return ET_SPEC_ERR_FP_ENV;
  if (draft_tokens == NULL || draft_logits == NULL || target_logits == NULL ||
      params == NULL || rng_in == NULL || out_tokens == NULL ||
      out_result == NULL)
    return ET_SPEC_ERR_NULL;
  if (!is_aligned_ptr(draft_tokens, _Alignof(uint32_t)) ||
      !is_aligned_ptr(draft_logits, _Alignof(float)) ||
      !is_aligned_ptr(target_logits, _Alignof(float)) ||
      !is_aligned_ptr(params, _Alignof(et_spec_sampling_params_v1)) ||
      !is_aligned_ptr(rng_in, _Alignof(et_spec_rng_state_v1)) ||
      !is_aligned_ptr(out_tokens, _Alignof(uint32_t)) ||
      !is_aligned_ptr(out_result, _Alignof(et_spec_verify_result_v1)))
    return ET_SPEC_ERR_ALIGNMENT;

  et_spec_status st = validate_horizon(horizon);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_vocab(vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  if (out_capacity < horizon + 1u ||
      (uint64_t)out_capacity > (uint64_t)SIZE_MAX / sizeof(uint32_t))
    return ET_SPEC_ERR_BOUNDS;
  if (rng_in->counter > UINT64_MAX - (uint64_t)(horizon + 1u))
    return ET_SPEC_ERR_BOUNDS;

  const size_t draft_count = (size_t)horizon * (size_t)vocab_size;
  const size_t target_count = ((size_t)horizon + 1u) * (size_t)vocab_size;
  const byte_span spans[] = {
      {draft_tokens, (size_t)horizon * sizeof(uint32_t)},
      {draft_logits, draft_count * sizeof(float)},
      {target_logits, target_count * sizeof(float)},
      {params, sizeof(*params)},
      {rng_in, sizeof(*rng_in)},
      {out_tokens, (size_t)out_capacity * sizeof(uint32_t)},
      {out_result, sizeof(*out_result)},
  };
  if (!spans_pairwise_disjoint(spans, 7u))
    return ET_SPEC_ERR_ALIAS;

  st = validate_params(params, vocab_size);
  if (st != ET_SPEC_OK)
    return st;
  for (uint32_t i = 0u; i < horizon; ++i) {
    if (draft_tokens[i] >= vocab_size)
      return ET_SPEC_ERR_TOKEN_RANGE;
  }
  st = validate_logits_finite(draft_logits, draft_count);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_logits_finite(target_logits, target_count);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_scaled_rows_finite(draft_logits, horizon, vocab_size,
                                   params->temperature);
  if (st != ET_SPEC_OK)
    return st;
  st = validate_scaled_rows_finite(target_logits, horizon + 1u, vocab_size,
                                   params->temperature);
  if (st != ET_SPEC_OK)
    return st;

  double q_probs[ET_SPEC_MAX_VOCAB];
  double p_probs[ET_SPEC_MAX_VOCAB];
  uint32_t order[ET_SPEC_MAX_VOCAB];
  uint32_t sort_tmp[ET_SPEC_MAX_VOCAB];

  uint32_t local_tokens[ET_SPEC_MAX_HORIZON + 1u];
  et_spec_verify_result_v1 local_result;
  memset(&local_result, 0, sizeof(local_result));
  local_result.next_rng = *rng_in;

  for (uint32_t i = 0u; i < horizon; ++i) {
    const float *q_row = draft_logits + ((size_t)i * (size_t)vocab_size);
    const float *p_row = target_logits + ((size_t)i * (size_t)vocab_size);
    st = compute_probs_internal(q_row, vocab_size, params, q_probs, order,
                                sort_tmp);
    if (st != ET_SPEC_OK)
      return st;
    st = compute_probs_internal(p_row, vocab_size, params, p_probs, order,
                                sort_tmp);
    if (st != ET_SPEC_OK)
      return st;

    const uint32_t x_i = draft_tokens[i];
    const double q_x = q_probs[x_i];
    const double p_x = p_probs[x_i];

    const double u_accept =
        splitmix64_uniform01(rng_in->seed, local_result.next_rng.counter);
    local_result.next_rng.counter++;
    local_result.rng_draws++;

    /* Exact acceptance criterion for half-open u in [0, 1):
     * - Never accept a zero-probability target token (p_x == 0.0), even if
     *   u_accept == 0.0.
     * - Always accept when q_x <= p_x (including q_x == 0.0 && p_x > 0.0,
     *   with zero division, and p == q -> 100% acceptance).
     * - Otherwise accept iff u_accept * q_x < p_x (exact half-open [0, 1)
     *   equivalence to u < p_x / q_x without division).
     */
    int accept = 0;
    if (p_x > 0.0) {
      if (q_x <= p_x)
        accept = 1;
      else
        accept = (u_accept * q_x < p_x) ? 1 : 0;
    }

    if (accept) {
      local_tokens[local_result.emitted_count++] = x_i;
      local_result.accepted_count++;
      if (params->eos_token_id != ET_SPEC_NO_EOS &&
          x_i == params->eos_token_id) {
        local_result.eos_hit = 1u;
        break;
      }
    } else {
      /* Compute exact normalized residual r(v) = max(0, p(v) - q(v)) / Z.
       * Reuses q_probs buffer for residual storage.
       */
      double residual_sum = 0.0;
      for (uint32_t v = 0u; v < vocab_size; ++v) {
        const double diff = p_probs[v] - q_probs[v];
        const double r = (diff > 0.0) ? diff : 0.0;
        q_probs[v] = r;
        residual_sum += r;
      }
      if (!is_finite_f64(residual_sum))
        return ET_SPEC_ERR_NONFINITE;

      const double u_res =
          splitmix64_uniform01(rng_in->seed, local_result.next_rng.counter);
      local_result.next_rng.counter++;
      local_result.rng_draws++;

      uint32_t corr_tok = 0u;
      if (residual_sum > 0.0) {
        for (uint32_t v = 0u; v < vocab_size; ++v) {
          if (q_probs[v] > 0.0)
            q_probs[v] /= residual_sum;
        }
        corr_tok = sample_categorical(q_probs, vocab_size, u_res);
      } else {
        /* Measure-zero boundary: p == q everywhere (e.g., adversarial caller
         * supplied x_i with q(x_i) == p(x_i) == 0). Target p_i is exact.
         */
        corr_tok = sample_categorical(p_probs, vocab_size, u_res);
      }

      local_tokens[local_result.emitted_count++] = corr_tok;
      local_result.rejected = 1u;
      if (params->eos_token_id != ET_SPEC_NO_EOS &&
          corr_tok == params->eos_token_id)
        local_result.eos_hit = 1u;
      break;
    }
  }

  if (!local_result.rejected && !local_result.eos_hit) {
    const float *bonus_row =
        target_logits + ((size_t)horizon * (size_t)vocab_size);
    st = compute_probs_internal(bonus_row, vocab_size, params, p_probs, order,
                                sort_tmp);
    if (st != ET_SPEC_OK)
      return st;
    const double u_bonus =
        splitmix64_uniform01(rng_in->seed, local_result.next_rng.counter);
    local_result.next_rng.counter++;
    local_result.rng_draws++;
    const uint32_t bonus_tok = sample_categorical(p_probs, vocab_size, u_bonus);
    local_tokens[local_result.emitted_count++] = bonus_tok;
    if (params->eos_token_id != ET_SPEC_NO_EOS &&
        bonus_tok == params->eos_token_id)
      local_result.eos_hit = 1u;
  }

  memcpy(out_tokens, local_tokens,
         (size_t)local_result.emitted_count * sizeof(uint32_t));
  *out_result = local_result;
  return ET_SPEC_OK;
}
