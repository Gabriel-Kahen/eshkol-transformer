#define main ac4_checkpointing_main
#include "test_ac4_checkpointing.c"
#undef main

#include "o2_optimizer_internal.h"

enum { LAYERS = 3, WIDTH = 8, KV_WIDTH = 4, UPDATES = 4 };
enum { TENSORS = 9 * LAYERS, WEIGHT_WORDS = LAYERS * (2 * WIDTH +
                                5 * WIDTH * WIDTH + 2 * KV_WIDTH * WIDTH) };

typedef struct span {
  uint64_t dims[2];
  size_t rank, words, offset;
} span;

typedef struct model {
  fixture data;
  et_f32_parameter *parameters[TENSORS];
  et_o2_optimizer *optimizer;
} model;

typedef struct image {
  uint32_t values[WEIGHT_WORDS], first[WEIGHT_WORDS], second[WEIGHT_WORDS];
  uint64_t updates;
} image;

static void describe(const fixture *data, span spans[TENSORS]) {
  /* AC4's packed order, with the actual rank for each gamma or projection. */
  static const uint64_t dims[][2] = {
      {WIDTH, 0}, {WIDTH, WIDTH}, {KV_WIDTH, WIDTH}, {KV_WIDTH, WIDTH},
      {WIDTH, WIDTH}, {WIDTH, 0}, {WIDTH, WIDTH}, {WIDTH, WIDTH}, {WIDTH, WIDTH}};
  size_t offset = 0;
  for (size_t i = 0; i < TENSORS; ++i) {
    const uint64_t *shape = dims[i % COUNT(dims)];
    const size_t rank = shape[1] ? 2u : 1u;
    const size_t words = (size_t)(shape[0] * (rank == 2 ? shape[1] : 1u));
    spans[i] = (span){{shape[0], shape[1]}, rank, words, offset};
    offset += words;
    if ((i + 1u) % COUNT(dims) == 0)
      CHECK(offset == ((i + 1u) / COUNT(dims)) * data->per_layer);
  }
  CHECK(offset == data->p && offset == WEIGHT_WORDS);
}

static et_f32_tensor *tensor_from(const span *entry, const float *data) {
  et_f32_tensor *tensor = NULL;
  uint32_t bits[WIDTH * WIDTH];
  memcpy(bits, data + entry->offset, entry->words * sizeof(*bits));
  CHECK(et_f32_tensor_create_v1(entry->rank, entry->dims, &tensor, NULL) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, entry->words, NULL) == 0);
  return tensor;
}

static void allocate_parameters(model *live, const span spans[TENSORS]) {
  for (size_t i = 0; i < TENSORS; ++i) {
    et_f32_tensor *value = tensor_from(&spans[i], live->data.parameters);
    CHECK(et_f32_parameter_create_v1(value, &live->parameters[i], NULL) == 0);
    CHECK(et_f32_tensor_destroy_v1(&value, NULL) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(live->parameters[i],
                                            live->parameters[i], NULL) == 0);
  }
}

static void enroll_optimizer(model *live) {
  et_o2_optimizer_builder *builder = NULL;
  CHECK(et_o2_optimizer_builder_create_v1(TENSORS, ET_O2_CLIP_NONE, 0,
        ET_O2_SCHEDULE_CONSTANT, 0, 0, 0x3f800000u, &builder, NULL) == 0);
  for (size_t i = 0; i < TENSORS; ++i)
    CHECK(et_o2_optimizer_builder_set_v1(builder, i, live->parameters[i],
          live->parameters[i], 0x3c800000u, 0x3f600000u, 0x3f780000u,
          0x35800000u, 0x3c800000u, NULL) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &live->optimizer, NULL) == 0);
}

static void check_gradient_count(const model *live, uint64_t count) {
  for (size_t i = 0; i < TENSORS; ++i) {
    et_f32_gradient_metadata_v1 meta = {.struct_size = sizeof(meta)};
    CHECK(et_f32_parameter_gradient_metadata_v1(live->parameters[i], &meta, NULL) == 0);
    CHECK(meta.contribution_count == count);
    CHECK(meta.normalization_weight_bits == (count ? 0x3f800000u : 0u));
  }
}

static void transfer_gradients(const model *live, const span spans[TENSORS],
                               const float *gradient) {
  et_f32_tensor *numerators[TENSORS];
  et_f32_gradient_contribution_v1 rows[TENSORS];
  for (size_t i = 0; i < TENSORS; ++i) {
    numerators[i] = tensor_from(&spans[i], gradient);
    rows[i] = (et_f32_gradient_contribution_v1){
        .struct_size = sizeof(rows[i]), .destination = live->parameters[i],
        .weighted_numerator = numerators[i], .expected_ordinal = 0};
  }
  et_f32_gradient_plan *plan = NULL;
  CHECK(et_f32_gradient_plan_prepare_v1(TENSORS, rows, 0x3f800000u, &plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, NULL) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, NULL) == 0);
  for (size_t i = 0; i < TENSORS; ++i)
    CHECK(et_f32_tensor_destroy_v1(&numerators[i], NULL) == 0);
}

static void copy_values(const model *live, const span spans[TENSORS],
                        uint32_t bits[WEIGHT_WORDS]) {
  for (size_t i = 0; i < TENSORS; ++i) {
    const et_f32_tensor *value = NULL;
    CHECK(et_f32_parameter_value_tensor_v1(live->parameters[i], &value, NULL) == 0);
    CHECK(et_f32_tensor_copy_bits_to_v1(value, bits + spans[i].offset,
                                      spans[i].words, NULL) == 0);
  }
}

static et_ac4_status step(model *live, const span spans[TENSORS], run *trial) {
  const et_ac4_status status = et_ac4_run_v1(&trial->config, &trial->io, &trial->result);
  if (status != ET_AC4_OK)
    return status;
  if (live->optimizer == NULL)
    enroll_optimizer(live);
  check_gradient_count(live, 0);
  transfer_gradients(live, spans, trial->grad);
  check_gradient_count(live, 1);
  CHECK(et_o2_optimizer_step_v1(live->optimizer, NULL) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(live->optimizer, NULL) == 0);
  check_gradient_count(live, 0);
  return ET_AC4_OK;
}

static void capture(const model *live, const span spans[TENSORS], image *saved) {
  et_o2_optimizer_state *state = NULL;
  size_t count = 0;
  copy_values(live, spans, saved->values);
  CHECK(et_o2_optimizer_state_snapshot_v1(live->optimizer, &state, NULL) == 0);
  CHECK(et_o2_optimizer_state_completed_updates_v1(state, &saved->updates, NULL) == 0);
  CHECK(et_o2_optimizer_state_entry_count_v1(state, &count, NULL) == 0 &&
        count == TENSORS);
  for (size_t i = 0; i < TENSORS; ++i) {
    const span *entry = &spans[i];
    uint32_t *first = saved->first + entry->offset, *second = saved->second + entry->offset;
    CHECK(et_o2_optimizer_state_copy_moment_bits_v1(
          state, i, ET_O2_MOMENT_EXP_AVG, first, entry->words, NULL) == 0);
    CHECK(et_o2_optimizer_state_copy_moment_bits_v1(
          state, i, ET_O2_MOMENT_EXP_AVG_SQ, second, entry->words, NULL) == 0);
    uint32_t used_first = 0, used_second = 0;
    for (size_t j = 0; j < entry->words; ++j) {
      CHECK((first[j] & 0x7f800000u) != 0x7f800000u);
      CHECK((second[j] & 0xff800000u) < 0x7f800000u);
      used_first |= first[j] & 0x7fffffffu;
      used_second |= second[j];
    }
    CHECK(used_first != 0 && used_second != 0);
  }
  CHECK(et_o2_optimizer_state_release_v1(state, NULL) == 0);
}

static void cap_rejection(model *live, const span spans[TENSORS], run *trial) {
  uint32_t values[WEIGHT_WORDS];
  et_o2_test_live_counts_v1 before = {.struct_size = sizeof(before)};
  et_o2_test_live_counts_v1 after = {.struct_size = sizeof(after)};
  const fixture original = live->data;
  const size_t cap = trial->config.activation_limit_bytes;
  et_o2_test_live_counts_snapshot_v1(&before);
  memset(trial->output, 0xa1, sizeof(trial->output));
  memset(trial->dx, 0xa1, sizeof(trial->dx));
  memset(trial->grad, 0xa1, sizeof(trial->grad));
  memset(&trial->result, 0xa1, sizeof(trial->result));
  memset(trial->workspace, 0xa1, trial->plan.workspace_bytes);
  trial->config.activation_limit_bytes = cap - 1u;
  CHECK(step(live, spans, trial) == ET_AC4_ACTIVATION_LIMIT);
  trial->config.activation_limit_bytes = cap;

  const void *buffers[] = {trial->output, trial->dx, trial->grad,
                          &trial->result, trial->workspace};
  const size_t sizes[] = {sizeof(trial->output), sizeof(trial->dx), sizeof(trial->grad),
                          sizeof(trial->result), trial->plan.workspace_bytes};
  for (size_t i = 0; i < COUNT(buffers); ++i)
    for (size_t j = 0; j < sizes[i]; ++j)
      CHECK(((const unsigned char *)buffers[i])[j] == 0xa1);
  et_o2_test_live_counts_snapshot_v1(&after);
  CHECK(live->optimizer == NULL && memcmp(&before, &after, sizeof(before)) == 0);
  CHECK(memcmp(&original, &live->data, sizeof(original)) == 0);
  copy_values(live, spans, values);
  CHECK(memcmp(values, original.parameters, sizeof(values)) == 0);
  check_gradient_count(live, 0);
}

static void write_step(FILE *trace, const run *trial, const image *saved) {
  CHECK(fwrite(&trial->result.loss, sizeof(double), 1, trace) == 1);
  CHECK(fwrite(trial->output, sizeof(float), trial->io.output.count, trace) ==
        trial->io.output.count);
  CHECK(fwrite(trial->dx, sizeof(float), trial->io.d_input.count, trace) ==
        trial->io.d_input.count);
  CHECK(fwrite(trial->grad, sizeof(float), WEIGHT_WORDS, trace) == WEIGHT_WORDS);
  CHECK(fwrite(saved->values, sizeof(uint32_t), WEIGHT_WORDS, trace) == WEIGHT_WORDS);
  CHECK(fwrite(saved->first, sizeof(uint32_t), WEIGHT_WORDS, trace) == WEIGHT_WORDS);
  CHECK(fwrite(saved->second, sizeof(uint32_t), WEIGHT_WORDS, trace) == WEIGHT_WORDS);
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  FILE *trace = fopen(argv[1], "wb");
  CHECK(trace != NULL);
  const et_ac4_shape_v1 shape = shapes[2];
  CHECK(shape.layers == LAYERS && shape.batch == 2 && shape.tokens == 3 &&
        shape.query_heads == 4 && shape.kv_heads == 2 && shape.head_dim == 2 &&
        shape.query_heads * shape.head_dim == WIDTH &&
        shape.kv_heads * shape.head_dim == KV_WIDTH);
  model retained = {0}, replay = {0};
  fill_fixture(&retained.data, shape);
  replay.data = retained.data;
  span spans[TENSORS];
  describe(&retained.data, spans);
  allocate_parameters(&retained, spans);
  allocate_parameters(&replay, spans);
  for (size_t i = 0; i < TENSORS; ++i)
    CHECK(retained.parameters[i] != replay.parameters[i]);

  run a = {0}, b = {0};
  init_run(&a, config_for(shape, ET_AC4_POLICY_NONE, 0.2f, 0), &retained.data);
  init_run(&b, config_for(shape, ET_AC4_POLICY_EVERY_K_REPLAY_GRADIENTS, 0.2f, 2), &replay.data);
  a.config.activation_limit_bytes = a.plan.activation_peak_bytes;
  b.config.activation_limit_bytes = b.plan.activation_peak_bytes;
  CHECK(a.workspace != b.workspace && b.plan.workspace_bytes < a.plan.workspace_bytes);
  CHECK(a.plan.gradient_staging_bytes == WEIGHT_WORDS * sizeof(float) &&
        b.plan.gradient_staging_bytes == replay.data.per_layer * sizeof(float));
  cap_rejection(&replay, spans, &b);

  for (uint64_t i = 0; i < UPDATES; ++i) {
    a.config.step = b.config.step = i;
    CHECK(step(&replay, spans, &b) == ET_AC4_OK);
    CHECK(step(&retained, spans, &a) == ET_AC4_OK);
    CHECK(replay.optimizer != retained.optimizer);
    check_stats(&a);
    check_stats(&b);
    CHECK(a.result.memory.rematerialized_layers == 0 &&
          b.result.memory.rematerialized_layers != 0 &&
          isfinite(a.result.loss));
    check_parity(&a, &b);
    image left, right;
    capture(&retained, spans, &left);
    capture(&replay, spans, &right);
    CHECK(left.updates == i + 1u && right.updates == left.updates);
    CHECK(memcmp(left.values, right.values, sizeof(left.values)) == 0);
    CHECK(memcmp(left.first, right.first, sizeof(left.first)) == 0);
    CHECK(memcmp(left.second, right.second, sizeof(left.second)) == 0);
    CHECK(memcmp(left.values, retained.data.parameters, sizeof(left.values)) != 0);
    memcpy(retained.data.parameters, left.values, sizeof(left.values));
    memcpy(replay.data.parameters, right.values, sizeof(right.values));
    write_step(trace, &b, &right);
  }

  CHECK(fclose(trace) == 0);
  close_run(&a);
  close_run(&b);
  printf("AC4/O2 replay PASS: %d GQA layers, %d tensor destinations, %d AdamW steps, "
         "exact predictions/loss/gradients/weights/moments, cap rejection and retry\n",
         LAYERS, TENSORS, UPDATES);
  printf("AC4/O2 retained/replay workspace bytes %zu/%zu, replay forwards %u\n",
         a.result.memory.workspace_bytes, b.result.memory.workspace_bytes,
         b.result.memory.rematerialized_layers);
  return EXIT_SUCCESS;
}
