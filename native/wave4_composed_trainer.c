#include "eshkol_transformer/wave4_composed_trainer_abi.h"

#include <math.h>
#include <string.h>

#if !defined(__x86_64__)
#error "Wave 4/5 composition requires the reviewed x86-64 floating-point lane"
#endif
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define SPAN(ptr) {ptr, sizeof(*(ptr)), __alignof__(*(ptr))}
#define ERROR(name) ET_WAVE4_COMPOSED_##name

enum phase {
  BOOTSTRAP_PREFLIGHT = 1,
  BOOTSTRAP_WEIGHTS,
  BOOTSTRAP_COMMIT,
  STEP_OVERFLOW,
  STEP_GRADIENTS,
  STEP_COMMIT
};

typedef struct {
  const void *data;
  size_t bytes, alignment;
} span;

typedef struct {
  float *weights, *gradients, *output, *d_input, *ac4;
  uint16_t *narrow;
} buffers;

enum cast_kind { NARROW, WIDEN, UNSCALE };

typedef struct {
  const char *dtype, *cast_capability, *operations[3];
} format;

static const format formats[] = {
    {"f16",
     "kernel.cast-f16",
     {"cast.f32-to-f16", "cast.f16-to-f32", "grad.unscale-f16-to-f32"}},
    {"bf16",
     "kernel.cast-bf16",
     {"cast.f32-to-bf16", "cast.bf16-to-f32", "grad.unscale-bf16-to-f32"}}};

static int valid_span(span range) {
  const uintptr_t address = (uintptr_t)range.data;
  return address != 0 && address % range.alignment == 0 &&
         range.bytes <= UINTPTR_MAX - address;
}

static int overlaps(span a, span b) {
  if (a.bytes == 0 || b.bytes == 0)
    return 0;
  const uintptr_t x = (uintptr_t)a.data, y = (uintptr_t)b.data;
  return x <= y ? y - x < a.bytes : x - y < b.bytes;
}

static et_wave4_composed_status check_spans(const span *ranges, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (!valid_span(ranges[i]))
      return ERROR(INVALID_ARGUMENT);
    for (size_t j = 0; j < i; ++j)
      if (overlaps(ranges[i], ranges[j]))
        return ERROR(ALIASING);
  }
  return ERROR(OK);
}

static int supported_fenv(void) {
  uint16_t x87;
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  return (x87 & 0x0c3fu) == 0x003fu && (_mm_getcsr() & 0xffc0u) == 0x1f80u;
}

static et_wave4_composed_status ac4_status(et_ac4_status status) {
  switch (status) {
  case ET_AC4_OK:
    return ERROR(OK);
  case ET_AC4_UNSUPPORTED:
  case ET_AC4_INVALID_ARGUMENT:
  case ET_AC4_SHAPE_MISMATCH:
    return ERROR(INVALID_ARGUMENT);
  case ET_AC4_ALIASING:
    return ERROR(ALIASING);
  case ET_AC4_ACTIVATION_LIMIT:
    return ERROR(ACTIVATION_LIMIT);
  case ET_AC4_WORKSPACE_TOO_SMALL:
    return ERROR(WORKSPACE_TOO_SMALL);
  case ET_AC4_NUMERIC_ERROR:
    return ERROR(NUMERIC_ERROR);
  default:
    return ERROR(PROVIDER_ERROR);
  }
}

static et_wave4_composed_status
make_plan(const et_wave4_composed_config_v1 *config,
          et_wave4_composed_memory_v1 *out) {
  if (!supported_fenv())
    return ERROR(UNSUPPORTED_ENVIRONMENT);
  if ((config->ac4.shape.layers != 4 && config->ac4.shape.layers != 8) ||
      config->ac4.mode != ET_AC4_FORWARD_BACKWARD ||
      (config->dtype != ET_WAVE4_COMPOSED_F16 &&
       config->dtype != ET_WAVE4_COMPOSED_BF16) ||
      !isfinite(config->learning_rate) || config->learning_rate <= 0.0f ||
      !isfinite(config->momentum) || config->momentum < 0.0f ||
      config->momentum >= 1.0f)
    return ERROR(INVALID_ARGUMENT);

  et_ac4_memory_v1 ac4;
  const et_wave4_composed_status status =
      ac4_status(et_ac4_plan_v1(&config->ac4, &ac4));
  if (status != ERROR(OK))
    return status;

  /* Every supported AC4 row is small and has an eight-aligned parameter count.
   */
  const size_t narrow = ac4.parameter_floats * sizeof(uint16_t);
  const size_t staging = 2 * (ac4.parameter_floats + ac4.activation_floats);
  *out = (et_wave4_composed_memory_v1){
      ac4, narrow, ac4.workspace_bytes + staging * sizeof(float) + narrow};
  return ERROR(OK);
}

et_wave4_composed_status
et_wave4_composed_plan_v1(const et_wave4_composed_config_v1 *config,
                          et_wave4_composed_memory_v1 *out) {
  const span ranges[] = {SPAN(config), SPAN(out)};
  const et_wave4_composed_status status = check_spans(ranges, COUNT(ranges));
  return status == ERROR(OK) ? make_plan(config, out) : status;
}

static buffers bind_buffers(et_wave4_composed_workspace_v1 workspace,
                            const et_wave4_composed_memory_v1 *plan) {
  const size_t p = plan->ac4.parameter_floats, x = plan->ac4.activation_floats;
  float *weights = workspace.data;
  const buffers bound = {weights,
                         weights + p,
                         weights + 2 * p,
                         weights + 2 * p + x,
                         weights + 2 * (p + x),
                         (uint16_t *)((char *)workspace.data +
                                      plan->workspace_bytes -
                                      plan->narrow_parameter_bytes)};
  return bound;
}

static et_wave4_composed_status
check_io(const et_wave4_composed_state_v1 *state,
         const et_wave4_composed_batch_v1 *batch,
         et_wave4_composed_workspace_v1 workspace,
         const et_wave4_composed_memory_v1 *plan, span *ranges, size_t count) {
  const size_t p = plan->ac4.parameter_floats, x = plan->ac4.activation_floats;
  if (state->parameter_count != p ||
      (batch && (batch->input.count != x || batch->target.count != x)))
    return ERROR(INVALID_ARGUMENT);

  ranges[count++] = (span){state->master, p * sizeof(float), _Alignof(float)};
  ranges[count++] = (span){state->momentum, p * sizeof(float), _Alignof(float)};
  ranges[count++] = (span){workspace.data, workspace.bytes, _Alignof(float)};
  if (batch) {
    ranges[count++] =
        (span){batch->input.data, x * sizeof(float), _Alignof(float)};
    ranges[count++] =
        (span){batch->target.data, x * sizeof(float), _Alignof(float)};
  }
  const et_wave4_composed_status status = check_spans(ranges, count);
  if (status != ERROR(OK))
    return status;
  return workspace.bytes >= plan->workspace_bytes ? ERROR(OK)
                                                  : ERROR(WORKSPACE_TOO_SMALL);
}

static et_wave4_composed_status dist_status(et_dist5_status_t status) {
  return status == ET_DIST5_OK ? ERROR(OK) : ERROR(DIST5_ERROR_BASE) + status;
}

static et_wave4_composed_status vote(et_dist5_comm_t *comm, int timeout_ms,
                                     enum phase tag,
                                     et_wave4_composed_status local_status,
                                     int64_t *overflow) {
  int64_t flags[] = {local_status, *overflow};
  const et_dist5_status_t status =
      et_dist5_allreduce(comm, flags, flags, COUNT(flags), ET_DIST5_I64,
                         ET_DIST5_MAX, tag, timeout_ms);
  if (status != ET_DIST5_OK)
    return dist_status(status);
  *overflow = flags[1];
  return (et_wave4_composed_status)flags[0];
}

static int invertible_scale(float scale) {
  const float inverse = 1.0f / scale;
  return isfinite(inverse) && inverse > 0.0f;
}

et_wave4_composed_status
et_wave4_composed_bootstrap_v1(const et_wave4_composed_config_v1 *config,
                               et_dist5_comm_t *comm, int timeout_ms,
                               const et_wave4_composed_scaler_config_v1 *scaler,
                               et_wave4_composed_state_v1 *state,
                               et_wave4_composed_workspace_v1 workspace) {
  span ranges[6] = {SPAN(config), SPAN(scaler), SPAN(state)};
  et_wave4_composed_memory_v1 plan;
  et_mp4_loss_scaler_v1 initial;
  et_wave4_composed_status status = check_spans(ranges, 3);
  if (status == ERROR(OK))
    status = make_plan(config, &plan);
  if (status == ERROR(OK))
    status = check_io(state, NULL, workspace, &plan, ranges, 3);
  if (status == ERROR(OK) &&
      (et_mp4_loss_scaler_init(&initial, scaler->scale, scaler->growth_factor,
                               scaler->backoff_factor, scaler->growth_interval,
                               scaler->min_scale, scaler->max_scale) !=
           ET_MP4_LOSS_SCALER_STEP_ACCEPTED ||
       !invertible_scale(initial.min_scale)))
    status = ERROR(INVALID_ARGUMENT);

  int64_t overflow = 0;
  status = vote(comm, timeout_ms, BOOTSTRAP_PREFLIGHT, status, &overflow);
  if (status != ERROR(OK))
    return status;

  const size_t p = plan.ac4.parameter_floats, bytes = p * sizeof(float);
  float *weights = workspace.data;
  memcpy(weights, state->master, bytes);
  status = dist_status(et_dist5_broadcast(comm, weights, p, ET_DIST5_F32, 0,
                                          BOOTSTRAP_WEIGHTS, timeout_ms));
  if (status != ERROR(OK))
    return status;
  for (size_t i = 0; i < p; ++i)
    if (!isfinite(weights[i]))
      status = ERROR(NUMERIC_ERROR);

  status = vote(comm, timeout_ms, BOOTSTRAP_COMMIT, status, &overflow);
  if (status == ERROR(OK)) {
    memcpy(state->master, weights, bytes);
    memset(state->momentum, 0, bytes);
    state->scaler = initial;
  }
  return status;
}

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                     const char *dtype, const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){
      sizeof(et_kernel_tensor_view_v1), data, bytes,         dtype, "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0,    shape != NULL, shape};
}

static const et_kernel_provider_v1 *resolve_mp4(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_mp4_kernel_provider_v1()
             : NULL;
}

static et_wave4_composed_status convert(const et_kernel_runtime *runtime,
                                        const format *storage,
                                        enum cast_kind kind, const void *input,
                                        void *output, size_t count,
                                        float inverse, int64_t *overflow) {
  const size_t in_width = kind == NARROW ? sizeof(float) : sizeof(uint16_t);
  const size_t out_width = kind == NARROW ? sizeof(uint16_t) : sizeof(float);
  const char *in_dtype = kind == NARROW ? "f32" : storage->dtype;
  const char *out_dtype = kind == NARROW ? storage->dtype : "f32";
  const size_t arity = kind == UNSCALE ? 2 : 1;
  const char *capability =
      kind == UNSCALE ? "kernel.grad-unscale" : storage->cast_capability;

  for (size_t offset = 0; offset < count;) {
    const size_t left = count - offset;
    const uint64_t tile = left >= 256  ? 256
                          : left >= 64 ? 64
                          : left >= 16 ? 16
                                       : 8;
    uint8_t tile_overflow = 0;
    const et_kernel_tensor_view_v1 in[] = {
        view((unsigned char *)input + offset * in_width, tile * in_width,
             in_dtype, &tile),
        view(&inverse, sizeof(inverse), "f32", NULL)};
    et_kernel_tensor_view_v1 out[] = {
        view((unsigned char *)output + offset * out_width, tile * out_width,
             out_dtype, &tile),
        view(&tile_overflow, sizeof(tile_overflow), "bool", NULL)};
    const et_kernel_request_v1 request = {sizeof(request),
                                          storage->operations[kind],
                                          "f32",
                                          "cpu",
                                          1,
                                          &tile,
                                          1,
                                          {0}};
    const et_kernel_call_v1 call = {
        sizeof(call), capability,           &request, arity,
        sizeof(*in),  arity * sizeof(*in),  in,       arity,
        sizeof(*out), arity * sizeof(*out), out};
    et_kernel_error error;
    if (et_kernel_runtime_dispatch(runtime, &call, &error) != 0) {
      if (error.category == ET_KERNEL_ERROR_INVALID_ARGUMENT &&
          error.code == ET_KERNEL_CODE_PROVIDER_REJECTED)
        return ERROR(NUMERIC_ERROR);
      return ERROR(PROVIDER_ERROR);
    }
    *overflow |= tile_overflow;
    offset += tile;
  }
  return ERROR(OK);
}

static et_wave4_composed_status
run_model(const et_wave4_composed_config_v1 *config,
          const et_wave4_composed_state_v1 *state,
          const et_wave4_composed_batch_v1 *batch,
          const et_wave4_composed_memory_v1 *plan, buffers work,
          et_ac4_result_v1 *result, int64_t *overflow, int training) {
  const format *storage = &formats[config->dtype - 1];
  const size_t p = plan->ac4.parameter_floats, x = plan->ac4.activation_floats;
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  if (et_kernel_runtime_discover(resolve_mp4, NULL, &runtime, &error) != 0)
    return ERROR(PROVIDER_ERROR);

  et_wave4_composed_status status = convert(
      runtime, storage, NARROW, state->master, work.narrow, p, 0, overflow);
  if (status == ERROR(OK))
    status = convert(runtime, storage, WIDEN, work.narrow, work.weights, p, 0,
                     overflow);

  if (status == ERROR(OK)) {
    et_ac4_config_v1 ac4 = config->ac4;
    if (!training) {
      ac4.mode = ET_AC4_FORWARD_ONLY;
      ac4.dropout_rate = 0;
    }
    const et_ac4_io_v1 io = {
        batch->input,
        batch->target,
        {work.weights, p},
        {work.output, x},
        {training ? work.d_input : NULL, training ? x : 0},
        {training ? work.gradients : NULL, training ? p : 0},
        work.ac4,
        plan->ac4.workspace_bytes};
    /* AC4 dispatches the pinned MOD5 provider for its layer primitives. */
    status = ac4_status(et_ac4_run_v1(&ac4, &io, result));
  }

  if (training && status == ERROR(OK)) {
    for (size_t i = 0; i < p; ++i)
      work.gradients[i] *= state->scaler.scale;
    status = convert(runtime, storage, NARROW, work.gradients, work.narrow, p,
                     0, overflow);
    if (status == ERROR(NUMERIC_ERROR)) {
      *overflow = 1;
      status = ERROR(OK);
    } else if (status == ERROR(OK)) {
      const float inverse = 1.0f / state->scaler.scale;
      status = convert(runtime, storage, UNSCALE, work.narrow, work.gradients,
                       p, inverse, overflow);
    }
  }
  et_kernel_runtime_destroy(runtime);
  return status;
}

static int valid_scaler(const et_mp4_loss_scaler_v1 *scaler) {
  et_mp4_loss_scaler_v1 checked;
  return et_mp4_loss_scaler_init(
             &checked, scaler->scale, scaler->growth_factor,
             scaler->backoff_factor, scaler->growth_interval, scaler->min_scale,
             scaler->max_scale) == ET_MP4_LOSS_SCALER_STEP_ACCEPTED &&
         scaler->clean_steps < scaler->growth_interval &&
         invertible_scale(scaler->min_scale);
}

static et_wave4_composed_status
stage_optimizer(const et_wave4_composed_config_v1 *config,
                const et_wave4_composed_state_v1 *state, buffers work) {
  for (size_t i = 0; i < state->parameter_count; ++i) {
    const float momentum =
        config->momentum * state->momentum[i] + work.gradients[i];
    const float weight = state->master[i] - config->learning_rate * momentum;
    if (!isfinite(momentum) || !isfinite(weight))
      return ERROR(NUMERIC_ERROR);
    work.gradients[i] = momentum;
    work.weights[i] = weight;
  }
  return ERROR(OK);
}

et_wave4_composed_status
et_wave4_composed_step_v1(const et_wave4_composed_config_v1 *config,
                          et_dist5_comm_t *comm, int timeout_ms,
                          et_wave4_composed_state_v1 *state,
                          const et_wave4_composed_batch_v1 *batch,
                          et_wave4_composed_workspace_v1 workspace,
                          et_wave4_composed_result_v1 *result) {
  span ranges[9] = {SPAN(config), SPAN(state), SPAN(batch), SPAN(result)};
  et_wave4_composed_memory_v1 plan;
  et_wave4_composed_status status = check_spans(ranges, 4);
  if (status == ERROR(OK))
    status = make_plan(config, &plan);
  if (status == ERROR(OK))
    status = check_io(state, batch, workspace, &plan, ranges, 4);
  if (status == ERROR(OK) && !valid_scaler(&state->scaler))
    status = ERROR(INVALID_ARGUMENT);

  int64_t overflow = 0;
  buffers work = {0};
  et_ac4_result_v1 local;
  if (status == ERROR(OK)) {
    work = bind_buffers(workspace, &plan);
    status = run_model(config, state, batch, &plan, work, &local, &overflow, 1);
  }

  const uint32_t local_overflow = (uint32_t)overflow;
  status = vote(comm, timeout_ms, STEP_OVERFLOW, status, &overflow);
  if (status != ERROR(OK))
    return status;

  const size_t p = plan.ac4.parameter_floats;
  if (!overflow) {
    status = dist_status(et_dist5_sync_gradients_f32(
        comm, work.gradients, work.gradients, p, STEP_GRADIENTS, timeout_ms));
    if (status != ERROR(OK))
      return status;
  }

  et_mp4_loss_scaler_v1 next = state->scaler;
  switch (et_mp4_loss_scaler_update(&next, (uint8_t)overflow)) {
  case ET_MP4_LOSS_SCALER_STEP_ACCEPTED:
    status = stage_optimizer(config, state, work);
    break;
  case ET_MP4_LOSS_SCALER_OVERFLOW_BACKOFF:
    status = ERROR(OVERFLOW_BACKOFF);
    break;
  case ET_MP4_LOSS_SCALER_UNDERFLOW_FAULT:
    status = ERROR(UNDERFLOW_FAULT);
    break;
  case ET_MP4_LOSS_SCALER_UNSUPPORTED_ENVIRONMENT:
    status = ERROR(UNSUPPORTED_ENVIRONMENT);
    break;
  default:
    status = ERROR(INVALID_ARGUMENT);
  }

  status = vote(comm, timeout_ms, STEP_COMMIT, status, &overflow);
  if (status != ERROR(OK) && status != ERROR(OVERFLOW_BACKOFF))
    return status;
  if (status == ERROR(OK)) {
    memcpy(state->master, work.weights, p * sizeof(float));
    memcpy(state->momentum, work.gradients, p * sizeof(float));
  }
  state->scaler = next;
  plan.ac4 = local.memory;
  *result = (et_wave4_composed_result_v1){local.loss, plan, local_overflow};
  return status;
}

static et_eval5_status write_and_verify(const char *path,
                                        const et_eval5_manifest *manifest,
                                        et_eval5_manifest *reloaded) {
  char expected[ET_EVAL5_MANIFEST_MAX_BYTES + 1], actual[sizeof(expected)];
  size_t expected_size, actual_size;
  et_eval5_status status = et_eval5_manifest_format(
      manifest, expected, sizeof(expected), &expected_size);
  if (status == ET_EVAL5_OK)
    status = et_eval5_manifest_write_atomic(path, manifest);
  if (status == ET_EVAL5_OK)
    status = et_eval5_manifest_read_file(path, reloaded);
  if (status == ET_EVAL5_OK)
    status = et_eval5_manifest_format(reloaded, actual, sizeof(actual),
                                      &actual_size);
  if (status == ET_EVAL5_OK && (actual_size != expected_size ||
                                memcmp(expected, actual, actual_size) != 0))
    return ET_EVAL5_ERR_CHECKSUM;
  return status;
}

et_wave4_composed_status
et_wave4_composed_evaluate_v1(const et_wave4_composed_config_v1 *config,
                              const et_wave4_composed_state_v1 *state,
                              const et_wave4_composed_eval_v1 *eval,
                              et_wave4_composed_workspace_v1 workspace,
                              et_eval5_manifest *out) {
  span ranges[14] = {SPAN(config), SPAN(state), SPAN(eval), SPAN(out)};
  et_wave4_composed_status status = check_spans(ranges, 4);
  if (status != ERROR(OK))
    return status;
  et_wave4_composed_memory_v1 plan;
  status = make_plan(config, &plan);
  if (status != ERROR(OK))
    return status;

  const size_t rows = config->ac4.shape.batch * config->ac4.shape.tokens;
  const et_eval5_document *train = &eval->train_document,
                          *held_out = &eval->eval_document;
  if (eval->path == NULL || eval->target_count != rows ||
      train->byte_count == 0 || held_out->byte_count == 0 ||
      train->tokens != NULL || train->token_count != 0 ||
      held_out->tokens != NULL || held_out->token_count != 0)
    return ERROR(INVALID_ARGUMENT);
  ranges[4] = (span)SPAN(eval->identity);
  ranges[5] = (span){eval->path, strlen(eval->path) + 1, 1};
  ranges[6] =
      (span){eval->target_ids, rows * sizeof(uint32_t), _Alignof(uint32_t)};
  ranges[7] = (span){train->bytes, train->byte_count, 1};
  ranges[8] = (span){held_out->bytes, held_out->byte_count, 1};
  status = check_io(state, &eval->batch, workspace, &plan, ranges, 9);
  if (status != ERROR(OK))
    return status;
  if (eval->identity->contamination.ngram_mode != ET_EVAL5_NGRAM_BYTES)
    return ERROR(INVALID_ARGUMENT);

  buffers work = bind_buffers(workspace, &plan);
  et_ac4_result_v1 forward;
  int64_t overflow = 0;
  status = run_model(config, state, &eval->batch, &plan, work, &forward,
                     &overflow, 0);
  if (status != ERROR(OK))
    return status;

  const size_t x = plan.ac4.activation_floats;
  const et_eval5_sequence sequence = {work.output, eval->target_ids, rows,
                                      x / rows, x};
  et_eval5_manifest manifest = *eval->identity;
  manifest.context_length = config->ac4.shape.tokens;
  manifest.metric_mask = ET_EVAL5_METRIC_LM;
  manifest.mc = (et_eval5_mc_metrics){0};
  manifest.mc_normalize_by_length = 0;

  uint64_t contaminated;
  et_eval5_status scored = et_eval5_audit_contamination(
      train, 1, held_out, 1, &manifest.contamination, &contaminated, 1,
      &manifest.audit);
  if (scored == ET_EVAL5_OK)
    scored = et_eval5_score_lm(&sequence, held_out->byte_count, &manifest.lm);
  et_eval5_manifest verified;
  if (scored == ET_EVAL5_OK)
    scored = write_and_verify(eval->path, &manifest, &verified);
  if (scored != ET_EVAL5_OK)
    return ERROR(EVAL5_ERROR_BASE) - scored;
  *out = verified;
  return ERROR(OK);
}
