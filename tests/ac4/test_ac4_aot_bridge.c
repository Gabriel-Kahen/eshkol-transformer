#include "ac4_eshkol_bridge.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma STDC FP_CONTRACT OFF

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static int forbid_allocations = 0;
static int fail_calloc_countdown = 0;

void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
void __real_free(void *);

void *__wrap_malloc(size_t bytes) {
  CHECK(!forbid_allocations);
  return __real_malloc(bytes);
}

void *__wrap_calloc(size_t count, size_t bytes) {
  CHECK(!forbid_allocations);
  if (fail_calloc_countdown > 0) {
    --fail_calloc_countdown;
    if (fail_calloc_countdown == 0)
      return NULL;
  }
  return __real_calloc(count, bytes);
}

void *__wrap_realloc(void *ptr, size_t bytes) {
  CHECK(!forbid_allocations);
  return __real_realloc(ptr, bytes);
}

void __wrap_free(void *ptr) {
  CHECK(!forbid_allocations);
  __real_free(ptr);
}

static float sample(uint32_t *state) {
  *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
  return (float)(*state >> 16) * 0x1p-15f - 1.0f;
}

static void populate_context(et_ac4_aot_context *ctx, int64_t layers) {
  const int64_t x_count = et_ac4_aot_activation_floats(ctx);
  const int64_t p_count = et_ac4_aot_parameter_floats(ctx);
  const int64_t per_layer = et_ac4_aot_parameter_floats_per_layer(ctx);
  const int64_t width = 8;
  const int64_t kv_width = 4;
  const int64_t gamma2 = width + 2 * width * width + 2 * kv_width * width;
  CHECK(x_count == 48);
  CHECK(per_layer == 400);
  CHECK(p_count == layers * per_layer);

  uint32_t state = 41u;
  for (int64_t i = 0; i < x_count; ++i) {
    const double x = 0.65 * (double)sample(&state);
    const double t = 0.62 * x + 0.27 * (double)sample(&state);
    CHECK(et_ac4_aot_set_input(ctx, i, x) == ET_AC4_OK);
    CHECK(et_ac4_aot_set_target(ctx, i, t) == ET_AC4_OK);
  }
  for (int64_t i = 0; i < p_count; ++i) {
    CHECK(et_ac4_aot_set_parameter(ctx, i, 0.28 * (double)sample(&state)) ==
          ET_AC4_OK);
  }
  for (int64_t l = 0; l < layers; ++l) {
    const int64_t base = l * per_layer;
    for (int64_t d = 0; d < width; ++d) {
      CHECK(et_ac4_aot_set_parameter(ctx, base + d,
                                     0.92 + 0.16 * (double)sample(&state)) ==
            ET_AC4_OK);
      CHECK(et_ac4_aot_set_parameter(ctx, base + gamma2 + d,
                                     0.87 + 0.16 * (double)sample(&state)) ==
            ET_AC4_OK);
    }
  }
}

static void check_policy_parity_and_memory(void) {
  const int64_t depths[] = {1, 4, 8, 16};
  const double rates[] = {0.0, 0.2};
  const int64_t policies[] = {ET_AC4_POLICY_NONE, ET_AC4_POLICY_EVERY_LAYER,
                              ET_AC4_POLICY_SQRT_SEGMENTS,
                              ET_AC4_POLICY_EVERY_K};

  for (size_t di = 0; di < COUNT(depths); ++di) {
    const int64_t layers = depths[di];
    for (size_t ri = 0; ri < COUNT(rates); ++ri) {
      const double rate = rates[ri];
      et_ac4_aot_context *ctxs[4] = {NULL, NULL, NULL, NULL};
      int64_t peaks[4] = {0, 0, 0, 0};

      for (size_t p = 0; p < 4; ++p) {
        const int64_t k = (p == 3) ? layers : 0;
        CHECK(et_ac4_aot_create(layers, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                                policies[p], k, 0.125, rate,
                                INT64_C(0x570efdb000311fb0), 0,
                                &ctxs[p]) == ET_AC4_OK);
        CHECK(ctxs[p] != NULL);
        peaks[p] = et_ac4_aot_planned_peak_bytes(ctxs[p]);
        CHECK(peaks[p] > 0);
        CHECK(et_ac4_aot_set_activation_limit(ctxs[p], peaks[p]) == ET_AC4_OK);
        populate_context(ctxs[p], layers);
      }

      if (layers >= 4) {
        CHECK(peaks[0] > peaks[1]);
        CHECK(peaks[1] > peaks[2]);
        CHECK(peaks[2] > peaks[3]);
      }

      const int64_t x_count = et_ac4_aot_activation_floats(ctxs[0]);
      const int64_t p_count = et_ac4_aot_parameter_floats(ctxs[0]);

      /* Run 4 SGD steps with zero heap allocations allowed during execution. */
      forbid_allocations = 1;
      for (int64_t step = 0; step < 4; ++step) {
        for (size_t p = 0; p < 4; ++p) {
          CHECK(et_ac4_aot_run(ctxs[p], step) == ET_AC4_OK);
          CHECK(et_ac4_aot_peak_activation_bytes(ctxs[p]) == peaks[p]);
          CHECK(et_ac4_aot_recomputation_count(ctxs[p]) ==
                et_ac4_aot_planned_rematerialized_layers(ctxs[p]));
          CHECK(et_ac4_aot_retained_checkpoints(ctxs[p]) ==
                et_ac4_aot_planned_retained_checkpoints(ctxs[p]));
          CHECK(et_ac4_aot_retained_tapes(ctxs[p]) ==
                et_ac4_aot_planned_retained_tapes(ctxs[p]));
          CHECK(et_ac4_aot_workspace_bytes(ctxs[p]) ==
                et_ac4_aot_planned_workspace_bytes(ctxs[p]));
          CHECK(et_ac4_aot_gradient_staging_bytes(ctxs[p]) ==
                et_ac4_aot_planned_gradient_staging_bytes(ctxs[p]));
          if (p > 0) {
            CHECK(et_ac4_aot_loss_bits_lo(ctxs[p]) ==
                  et_ac4_aot_loss_bits_lo(ctxs[0]));
            CHECK(et_ac4_aot_loss_bits_hi(ctxs[p]) ==
                  et_ac4_aot_loss_bits_hi(ctxs[0]));
            for (int64_t i = 0; i < x_count; ++i) {
              CHECK(et_ac4_aot_output_bits(ctxs[p], i) ==
                    et_ac4_aot_output_bits(ctxs[0], i));
              CHECK(et_ac4_aot_d_input_bits(ctxs[p], i) ==
                    et_ac4_aot_d_input_bits(ctxs[0], i));
            }
            for (int64_t i = 0; i < p_count; ++i) {
              CHECK(et_ac4_aot_d_parameter_bits(ctxs[p], i) ==
                    et_ac4_aot_d_parameter_bits(ctxs[0], i));
            }
          }
          CHECK(et_ac4_aot_sgd_step(ctxs[p], 0.035) == ET_AC4_OK);
          if (p > 0) {
            for (int64_t i = 0; i < p_count; ++i) {
              CHECK(et_ac4_aot_parameter_bits(ctxs[p], i) ==
                    et_ac4_aot_parameter_bits(ctxs[0], i));
            }
          }
        }
      }
      forbid_allocations = 0;

      for (size_t p = 0; p < 4; ++p)
        et_ac4_aot_destroy(ctxs[p]);
    }
  }
}

static void check_rng_replay_and_forward_only(void) {
  et_ac4_aot_context *a = NULL, *b = NULL, *fwd = NULL;
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_SQRT_SEGMENTS, 0, 0.125, 0.25, 777, 0,
                          &a) == ET_AC4_OK);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_EVERY_K, 2, 0.125, 0.25, 778, 0,
                          &b) == ET_AC4_OK);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_ONLY,
                          ET_AC4_POLICY_EVERY_LAYER, 0, 0.125, 0.25, 777, 0,
                          &fwd) == ET_AC4_OK);
  populate_context(a, 4);
  populate_context(b, 4);
  populate_context(fwd, 4);

  forbid_allocations = 1;
  CHECK(et_ac4_aot_run(a, 3) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(b, 3) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(fwd, 3) == ET_AC4_OK);
  /* Forward-only matches forward+backward loss and output bitwise. */
  CHECK(et_ac4_aot_loss_bits_lo(a) == et_ac4_aot_loss_bits_lo(fwd));
  CHECK(et_ac4_aot_loss_bits_hi(a) == et_ac4_aot_loss_bits_hi(fwd));
  CHECK(et_ac4_aot_recomputation_count(fwd) == 0);
  CHECK(et_ac4_aot_gradient_staging_bytes(fwd) == 0);
  /* Forward-only rejects gradient reads and SGD step. */
  CHECK(isnan(et_ac4_aot_get_d_input(fwd, 0)));
  CHECK(isnan(et_ac4_aot_get_d_parameter(fwd, 0)));
  CHECK(et_ac4_aot_d_input_bits(fwd, 0) == -1);
  CHECK(et_ac4_aot_d_parameter_bits(fwd, 0) == -1);
  CHECK(et_ac4_aot_sgd_step(fwd, 0.01) == ET_AC4_INVALID_ARGUMENT);
  /* Different seed produces a different dropout mask and loss. */
  CHECK(et_ac4_aot_loss_bits_lo(a) != et_ac4_aot_loss_bits_lo(b) ||
        et_ac4_aot_loss_bits_hi(a) != et_ac4_aot_loss_bits_hi(b));
  /* Different step on same seed produces a different loss. */
  const int64_t step3_lo = et_ac4_aot_loss_bits_lo(a);
  const int64_t step3_hi = et_ac4_aot_loss_bits_hi(a);
  CHECK(et_ac4_aot_run(a, 4) == ET_AC4_OK);
  CHECK(et_ac4_aot_loss_bits_lo(a) != step3_lo ||
        et_ac4_aot_loss_bits_hi(a) != step3_hi);
  /* Replaying step 3 restores bitwise-identical loss. */
  CHECK(et_ac4_aot_run(a, 3) == ET_AC4_OK);
  CHECK(et_ac4_aot_loss_bits_lo(a) == step3_lo &&
        et_ac4_aot_loss_bits_hi(a) == step3_hi);
  forbid_allocations = 0;

  et_ac4_aot_destroy(fwd);
  et_ac4_aot_destroy(b);
  et_ac4_aot_destroy(a);
}

static void check_negative_and_oom_cases(void) {
  et_ac4_aot_context *sentinel = (et_ac4_aot_context *)(uintptr_t)0xdeadbeefu;
  et_ac4_aot_context *ctx = sentinel;

  /* Null out_ctx. */
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          NULL) == ET_AC4_INVALID_ARGUMENT);
  /* Layer counts out of range. */
  CHECK(et_ac4_aot_create(0, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(-1, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(17, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  /* Invalid mode, policy, k, epsilon, dropout, activation_limit. */
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, 2, ET_AC4_POLICY_NONE, 0, 0.125,
                          0.0, 0, 0, &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, -1, ET_AC4_POLICY_NONE, 0, 0.125,
                          0.0, 0, 0, &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD, 4, 0,
                          0.125, 0.0, 0, 0, &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD, -1, 0,
                          0.125, 0.0, 0, 0, &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_EVERY_K, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_EVERY_K, 5, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 2, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.0, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, -0.1, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, (double)NAN, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 1.0e-50, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 1.0e300, 0.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, -0.01, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 1.0, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.999999999, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, (double)NAN, 0, 0,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, -1,
                          &ctx) == ET_AC4_INVALID_ARGUMENT);
  /* Invalid / unsupported shapes. */
  CHECK(et_ac4_aot_create(4, 0, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_SHAPE_MISMATCH);
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 3, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_SHAPE_MISMATCH);
  CHECK(et_ac4_aot_create(4, 2, 4, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                          &ctx) == ET_AC4_UNSUPPORTED);
  CHECK(et_ac4_aot_create(4, INT64_C(0x100000000), 3, 4, 2, 2,
                          ET_AC4_FORWARD_BACKWARD, ET_AC4_POLICY_NONE, 0, 0.125,
                          0.0, 0, 0, &ctx) == ET_AC4_UNSUPPORTED);
  CHECK(ctx == sentinel);

  /* OOM injection for every calloc site in et_ac4_aot_create (1..8). */
  for (int fail_at = 1; fail_at <= 8; ++fail_at) {
    fail_calloc_countdown = fail_at;
    ctx = sentinel;
    CHECK(et_ac4_aot_create(2, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                            ET_AC4_POLICY_NONE, 0, 0.125, 0.0, 0, 0,
                            &ctx) == ET_AC4_AOT_OUT_OF_MEMORY);
    CHECK(ctx == sentinel);
  }
  fail_calloc_countdown = 0;

  /* Null handle accessors. */
  et_ac4_aot_destroy(NULL);
  CHECK(et_ac4_aot_set_activation_limit(NULL, 0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_activation_floats(NULL) == -1);
  CHECK(et_ac4_aot_parameter_floats(NULL) == -1);
  CHECK(et_ac4_aot_parameter_floats_per_layer(NULL) == -1);
  CHECK(et_ac4_aot_planned_peak_bytes(NULL) == -1);
  CHECK(et_ac4_aot_planned_workspace_bytes(NULL) == -1);
  CHECK(et_ac4_aot_planned_gradient_staging_bytes(NULL) == -1);
  CHECK(et_ac4_aot_planned_retained_checkpoints(NULL) == -1);
  CHECK(et_ac4_aot_planned_retained_tapes(NULL) == -1);
  CHECK(et_ac4_aot_planned_rematerialized_layers(NULL) == -1);
  CHECK(et_ac4_aot_set_input(NULL, 0, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_target(NULL, 0, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_parameter(NULL, 0, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_run(NULL, 0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_sgd_step(NULL, 0.01) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_peak_activation_bytes(NULL) == -1);
  CHECK(et_ac4_aot_recomputation_count(NULL) == -1);
  CHECK(et_ac4_aot_retained_checkpoints(NULL) == -1);
  CHECK(et_ac4_aot_retained_tapes(NULL) == -1);
  CHECK(et_ac4_aot_workspace_bytes(NULL) == -1);
  CHECK(et_ac4_aot_gradient_staging_bytes(NULL) == -1);
  CHECK(isnan(et_ac4_aot_get_loss(NULL)));
  CHECK(isnan(et_ac4_aot_get_output(NULL, 0)));
  CHECK(isnan(et_ac4_aot_get_parameter(NULL, 0)));
  CHECK(isnan(et_ac4_aot_get_d_input(NULL, 0)));
  CHECK(isnan(et_ac4_aot_get_d_parameter(NULL, 0)));
  CHECK(et_ac4_aot_loss_bits_lo(NULL) == -1);
  CHECK(et_ac4_aot_loss_bits_hi(NULL) == -1);
  CHECK(et_ac4_aot_output_bits(NULL, 0) == -1);
  CHECK(et_ac4_aot_parameter_bits(NULL, 0) == -1);
  CHECK(et_ac4_aot_d_input_bits(NULL, 0) == -1);
  CHECK(et_ac4_aot_d_parameter_bits(NULL, 0) == -1);

  /* Context-level negative cases and failure atomicity. */
  ctx = NULL;
  CHECK(et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                          ET_AC4_POLICY_EVERY_LAYER, 0, 0.125, 0.0, 99, 0,
                          &ctx) == ET_AC4_OK);
  /* Before first run, result accessors report missing result. */
  CHECK(et_ac4_aot_peak_activation_bytes(ctx) == -1);
  CHECK(et_ac4_aot_recomputation_count(ctx) == -1);
  CHECK(isnan(et_ac4_aot_get_loss(ctx)));
  CHECK(et_ac4_aot_loss_bits_lo(ctx) == -1);
  CHECK(et_ac4_aot_output_bits(ctx, 0) == -1);
  CHECK(et_ac4_aot_sgd_step(ctx, 0.01) == ET_AC4_INVALID_ARGUMENT);

  populate_context(ctx, 4);
  const int64_t x_count = et_ac4_aot_activation_floats(ctx);
  const int64_t p_count = et_ac4_aot_parameter_floats(ctx);
  CHECK(et_ac4_aot_set_input(ctx, -1, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_input(ctx, x_count, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_target(ctx, -1, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_target(ctx, x_count, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_parameter(ctx, -1, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_parameter(ctx, p_count, 0.0) == ET_AC4_INVALID_ARGUMENT);

  CHECK(et_ac4_aot_run(ctx, 0) == ET_AC4_OK);
  const int64_t good_loss_lo = et_ac4_aot_loss_bits_lo(ctx);
  const int64_t good_loss_hi = et_ac4_aot_loss_bits_hi(ctx);
  const int64_t good_out0 = et_ac4_aot_output_bits(ctx, 0);
  const int64_t good_dp0 = et_ac4_aot_d_parameter_bits(ctx, 0);
  const int64_t good_p0 = et_ac4_aot_parameter_bits(ctx, 0);

  /* Negative step rejected. */
  CHECK(et_ac4_aot_run(ctx, -1) == ET_AC4_INVALID_ARGUMENT);
  /* Activation limit below planned peak rejected without mutating results. */
  const int64_t peak = et_ac4_aot_planned_peak_bytes(ctx);
  CHECK(et_ac4_aot_set_activation_limit(ctx, -1) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_set_activation_limit(ctx, peak - 1) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(ctx, 1) == ET_AC4_ACTIVATION_LIMIT);
  CHECK(et_ac4_aot_set_activation_limit(ctx, peak) == ET_AC4_OK);

  /* Non-finite input (both NAN and f32 overflow via 1e300), target, param. */
  const double saved_x0 = et_ac4_aot_get_parameter(ctx, 0);
  (void)saved_x0;
  CHECK(et_ac4_aot_set_input(ctx, 0, 1.0e300) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(ctx, 1) == ET_AC4_NUMERIC_ERROR);
  CHECK(et_ac4_aot_set_input(ctx, 0, (double)NAN) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(ctx, 1) == ET_AC4_NUMERIC_ERROR);
  CHECK(et_ac4_aot_set_input(ctx, 0, 0.25) == ET_AC4_OK);

  CHECK(et_ac4_aot_set_target(ctx, 0, -(double)INFINITY) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(ctx, 1) == ET_AC4_NUMERIC_ERROR);
  CHECK(et_ac4_aot_set_target(ctx, 0, 0.25) == ET_AC4_OK);

  const double saved_p0 = et_ac4_aot_get_parameter(ctx, 0);
  CHECK(et_ac4_aot_set_parameter(ctx, 0, (double)INFINITY) == ET_AC4_OK);
  CHECK(et_ac4_aot_run(ctx, 1) == ET_AC4_NUMERIC_ERROR);
  CHECK(et_ac4_aot_set_parameter(ctx, 0, saved_p0) == ET_AC4_OK);

  /* Previous good results remained intact across all rejected runs. */
  CHECK(et_ac4_aot_loss_bits_lo(ctx) == good_loss_lo);
  CHECK(et_ac4_aot_loss_bits_hi(ctx) == good_loss_hi);
  CHECK(et_ac4_aot_output_bits(ctx, 0) == good_out0);
  CHECK(et_ac4_aot_d_parameter_bits(ctx, 0) == good_dp0);

  /* Invalid learning rates and overflow-atomic SGD rejection. */
  CHECK(et_ac4_aot_sgd_step(ctx, 0.0) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_sgd_step(ctx, -0.01) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_sgd_step(ctx, (double)NAN) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_sgd_step(ctx, 1.0e-50) == ET_AC4_INVALID_ARGUMENT);
  CHECK(et_ac4_aot_sgd_step(ctx, 1.0e300) == ET_AC4_INVALID_ARGUMENT);
  int64_t overflow_idx = -1;
  for (int64_t i = 1; i < p_count; ++i) {
    if (fabs(et_ac4_aot_get_d_parameter(ctx, i)) > 1.0e-6) {
      overflow_idx = i;
      break;
    }
  }
  CHECK(overflow_idx > 0);
  const double saved_pk = et_ac4_aot_get_parameter(ctx, overflow_idx);
  const double dp_k = et_ac4_aot_get_d_parameter(ctx, overflow_idx);
  const double extreme_p = dp_k > 0.0 ? -(double)FLT_MAX : (double)FLT_MAX;
  CHECK(et_ac4_aot_set_parameter(ctx, overflow_idx, extreme_p) == ET_AC4_OK);
  const int64_t extreme_pk_bits = et_ac4_aot_parameter_bits(ctx, overflow_idx);
  CHECK(et_ac4_aot_sgd_step(ctx, (double)FLT_MAX) == ET_AC4_NUMERIC_ERROR);
  CHECK(et_ac4_aot_parameter_bits(ctx, 0) == good_p0);
  CHECK(et_ac4_aot_parameter_bits(ctx, overflow_idx) == extreme_pk_bits);
  CHECK(et_ac4_aot_set_parameter(ctx, overflow_idx, saved_pk) == ET_AC4_OK);

  /* OOB reads return NAN / -1. */
  CHECK(isnan(et_ac4_aot_get_output(ctx, -1)));
  CHECK(isnan(et_ac4_aot_get_output(ctx, x_count)));
  CHECK(isnan(et_ac4_aot_get_parameter(ctx, -1)));
  CHECK(isnan(et_ac4_aot_get_parameter(ctx, p_count)));
  CHECK(isnan(et_ac4_aot_get_d_input(ctx, -1)));
  CHECK(isnan(et_ac4_aot_get_d_input(ctx, x_count)));
  CHECK(isnan(et_ac4_aot_get_d_parameter(ctx, -1)));
  CHECK(isnan(et_ac4_aot_get_d_parameter(ctx, p_count)));
  CHECK(et_ac4_aot_output_bits(ctx, x_count) == -1);
  CHECK(et_ac4_aot_parameter_bits(ctx, p_count) == -1);
  CHECK(et_ac4_aot_d_input_bits(ctx, x_count) == -1);
  CHECK(et_ac4_aot_d_parameter_bits(ctx, p_count) == -1);

  et_ac4_aot_destroy(ctx);
}

int main(void) {
  check_policy_parity_and_memory();
  check_rng_replay_and_forward_only();
  check_negative_and_oom_cases();
  puts("AC4 AOT bridge native tests (parity, memory, RNG, OOM, negative): PASS");
  return EXIT_SUCCESS;
}
