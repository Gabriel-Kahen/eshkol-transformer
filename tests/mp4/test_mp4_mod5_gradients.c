#define main mod5_training_smoke_main
#include "../mod5/test_mod5_training_smoke.c"
#undef main

#include "eshkol_transformer/mp4_mixed_precision_abi.h"

#pragma STDC FP_CONTRACT OFF

enum { TILE = 8, ACCEPTED_STEPS = 4 };
static const uint64_t tile_shape[] = {TILE};
static const float learning_rate = 0.25f;

typedef struct format {
  const char *dtype, *cast, *unscale;
  float scale, fault_target, atol, rtol;
} format;

static const format formats[] = {
    {"f16", "cast.f32-to-f16", "grad.unscale-f16-to-f32", 32.0f, 1e8f,
     2e-5f, 0.005f},
    {"bf16", "cast.f32-to-bf16", "grad.unscale-bf16-to-f32", 65536.0f,
     1e37f, 1e-4f, 0.02f},
};

static const et_kernel_provider_v1 *resolve_mp4(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mp4_kernel_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 mp_view(void *data, const char *dtype,
                                        size_t rank, const uint64_t *shape) {
  const size_t width = strcmp(dtype, "f32") == 0
                           ? sizeof(float)
                           : strcmp(dtype, "bool") == 0 ? 1u : sizeof(uint16_t);
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1),
                                     data,
                                     rank == 0 ? width : TILE * width,
                                     dtype,
                                     "cpu",
                                     ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
                                     0u,
                                     rank,
                                     shape};
}

static int cast_tile(const et_kernel_runtime *runtime, const format *fmt,
                     float *scaled, uint16_t *narrow) {
  const et_kernel_request_v1 request = {sizeof(request), fmt->cast, "f32", "cpu",
                                        1u, tile_shape, 1u, {0}};
  const et_kernel_tensor_view_v1 in =
      mp_view(scaled, "f32", 1u, tile_shape);
  et_kernel_tensor_view_v1 out =
      mp_view(narrow, fmt->dtype, 1u, tile_shape);
  const et_kernel_call_v1 call =
      test_call(strcmp(fmt->dtype, "f16") == 0 ? "kernel.cast-f16"
                                                 : "kernel.cast-bf16",
                &request, 1u, &in, 1u, &out);
  et_kernel_error error;
  const int status = et_kernel_runtime_dispatch(runtime, &call, &error);
  if (status != 0)
    CHECK(error.code == ET_KERNEL_CODE_PROVIDER_REJECTED);
  return status;
}

static uint8_t unscale_tile(const et_kernel_runtime *runtime, const format *fmt,
                            uint16_t *narrow, float inverse, float *gradient) {
  const et_kernel_request_v1 request = {sizeof(request), fmt->unscale, "f32",
                                        "cpu", 1u, tile_shape, 1u, {0}};
  const et_kernel_tensor_view_v1 in[] = {
      mp_view(narrow, fmt->dtype, 1u, tile_shape),
      mp_view(&inverse, "f32", 0u, NULL)};
  uint8_t overflow = UINT8_MAX;
  et_kernel_tensor_view_v1 out[] = {
      mp_view(gradient, "f32", 1u, tile_shape),
      mp_view(&overflow, "bool", 0u, NULL)};
  const et_kernel_call_v1 call =
      test_call("kernel.grad-unscale", &request, 2u, in, 2u, out);
  dispatch(runtime, &call);
  CHECK(overflow <= 1u);
  return overflow;
}

static uint8_t store_and_unscale(const et_kernel_runtime *runtime,
                                 const format *fmt,
                                 const et_mp4_loss_scaler_v1 *scaler,
                                 const weights *raw, weights *unscaled) {
  const float *source = &raw->data[0][0];
  float *destination = &unscaled->data[0][0];
  double activity = 0.0;
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i) {
      CHECK(isfinite(raw->data[p][i]));
      activity += fabs(raw->data[p][i]);
    }
  CHECK(activity > 0.001);
  for (size_t offset = 0; offset < sizeof(*raw) / sizeof(float); offset += TILE) {
    float scaled[TILE];
    uint16_t narrow[TILE];
    for (size_t i = 0; i < TILE; ++i) {
      CHECK(isfinite(source[offset + i]));
      scaled[i] = source[offset + i] * scaler->scale;
    }
    if (cast_tile(runtime, fmt, scaled, narrow) != 0)
      return 1u;
    if (unscale_tile(runtime, fmt, narrow, 1.0f / scaler->scale,
                     destination + offset) != 0)
      return 2u;
  }
  return 0u;
}

static void apply_sgd(weights *model, const weights *gradient) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i)
      model->data[p][i] -= learning_rate * gradient->data[p][i];
}

static void check_parity(const weights *actual, const weights *baseline,
                         const format *fmt) {
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i)
      check_close(actual->data[p][i], baseline->data[p][i], fmt->atol,
                  fmt->rtol);
}

static void initialize(weights *w, float *x, float *target) {
  memset(w, 0, sizeof(*w));
  for (size_t p = 0; p < PARAMETERS; ++p)
    for (size_t i = 0; i < param_size(p, 2u); ++i) {
      const int seed = (int)((p * 29u + i * 11u + i / 5u) % 37u) - 18;
      w->data[p][i] = (p >= GAMMA1 ? 1.0f : 0.0f) + (float)seed * 0x1p-7f;
    }
  for (size_t i = 0; i < ELEMENTS; ++i) {
    x[i] = (float)((int)((i * 7u + i / 5u) % 31u) - 15) * 0x1p-5f;
    target[i] = 0.65f * x[i] + (float)((int)((i * 3u) % 13u) - 6) * 0x1p-6f;
  }
}

static void run_format(const et_kernel_runtime *mod5,
                       const et_kernel_runtime *mp4, const format *fmt) {
  weights baseline, master;
  float x[ELEMENTS], target[ELEMENTS], fault_target[ELEMENTS], dx[ELEMENTS];
  initialize(&baseline, x, target);
  master = baseline;
  for (size_t i = 0; i < ELEMENTS; ++i)
    fault_target[i] = fmt->fault_target;
  et_mp4_loss_scaler_v1 scaler;
  CHECK(et_mp4_loss_scaler_init(&scaler, fmt->scale, 2.0f, 0.5f, 100u, 1.0f,
                                fmt->scale) == ET_MP4_LOSS_SCALER_STEP_ACCEPTED);

  unsigned accepted = 0;
  for (unsigned attempt = 0; attempt < ACCEPTED_STEPS + 1u; ++attempt) {
    const int fault = attempt == 2u;
    weights raw, unscaled;
    const weights before = master;
    const double loss = block_loss(mod5, 2u, &master, x,
                                   fault ? fault_target : target, &raw, dx);
    CHECK(isfinite(loss));
    const uint8_t overflow = store_and_unscale(mp4, fmt, &scaler, &raw,
                                                &unscaled);
    const float prior_scale = scaler.scale;
    const et_mp4_loss_scaler_status outcome =
        et_mp4_loss_scaler_update(&scaler, overflow != 0u);
    if (fault) {
      CHECK(overflow == 1u &&
            outcome == ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF);
      CHECK(memcmp(&master, &before, sizeof(master)) == 0);
      CHECK(scaler.scale == prior_scale * 0.5f && scaler.clean_steps == 0u);
      continue;
    }
    CHECK(overflow == 0u && outcome == ET_MP4_LOSS_SCALER_STEP_ACCEPTED);
    weights reference_gradient;
    CHECK(isfinite(block_loss(mod5, 2u, &baseline, x, target,
                              &reference_gradient, dx)));
    for (size_t p = 0; p < PARAMETERS; ++p)
      for (size_t i = 0; i < param_size(p, 2u); ++i)
        check_close(unscaled.data[p][i], reference_gradient.data[p][i],
                    fmt->atol, fmt->rtol);
    apply_sgd(&baseline, &reference_gradient);
    apply_sgd(&master, &unscaled);
    check_parity(&master, &baseline, fmt);
    ++accepted;
  }
  CHECK(accepted == ACCEPTED_STEPS);
  printf("MP4/MOD5 %s real-gradient storage: PASS (%u updates, overflow "
         "backoff, recovery)\n", fmt->dtype, accepted);
}

int main(void) {
  et_kernel_runtime *mod5 = discover_mod5();
  et_kernel_runtime *mp4 = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_mp4, NULL, &mp4, &error) == 0);
  for (size_t i = 0; i < ARRAY_COUNT(formats); ++i)
    run_format(mod5, mp4, &formats[i]);
  et_kernel_runtime_destroy(mp4);
  et_kernel_runtime_destroy(mod5);
  return 0;
}
