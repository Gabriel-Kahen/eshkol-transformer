#include "o2_optimizer_internal.h"
#include "tr3_o2_step_clear_internal.h"

#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define N14 14u
#define MAX_ELEMENTS 1024u

static size_t checks;
static size_t failures;
static size_t outer_retained_bytes_per_update;
static size_t i2_retained_bytes_per_update;
static uint64_t repeated_state_digest;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
      failures++;                                                              \
    }                                                                          \
  } while (0)

typedef struct model_fixture {
  size_t count;
  size_t elements[N14];
  et_f32_parameter *parameters[N14];
  et_o2_optimizer *optimizer;
} model_fixture;

typedef struct optimizer_config {
  uint32_t clip_kind;
  uint32_t clip_max_bits;
  uint32_t schedule_kind;
  uint64_t warmup_updates;
  uint64_t total_updates;
  uint32_t minimum_ratio_bits;
} optimizer_config;

static void expect_o2_error(int32_t result, const et_o2_error_v1 *error,
                            uint32_t category, uint32_t code) {
  if (result != (int32_t)category || error->category != category ||
      error->code != code) {
    (void)fprintf(stderr,
                  "error mismatch: result=%d category=%u code=%u op=%s "
                  "expected=%u/%u\n",
                  result, error->category, error->code, error->operation,
                  category, code);
  }
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
}

static void expect_i2_error(int32_t result, const et_f32_tensor_error *error,
                            uint32_t category, uint32_t code) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
}

static et_f32_tensor *make_tensor(size_t rank, const uint64_t *shape,
                                  const uint32_t *bits, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, count, &error) == 0);
  return tensor;
}

static void destroy_tensor(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static uint64_t completed_updates(const et_o2_optimizer *optimizer) {
  et_o2_error_v1 error;
  uint64_t result = UINT64_MAX;
  CHECK(et_o2_optimizer_completed_updates_v1(optimizer, &result, &error) == 0);
  return result;
}

static et_f32_gradient_metadata_v1
gradient_metadata(et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  et_f32_gradient_metadata_v1 result = {.struct_size = sizeof(result)};
  CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &result, &error) == 0);
  return result;
}

static void read_parameter_bits(et_f32_parameter *parameter, uint32_t *bits,
                                size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor *snapshot = NULL;
  CHECK(et_f32_parameter_value_snapshot_v1(parameter, &snapshot, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(snapshot, bits, count, &error) == 0);
  destroy_tensor(&snapshot);
}

static const et_f32_tensor *value_storage(et_f32_parameter *parameter) {
  et_f32_tensor_error error;
  const et_f32_tensor *result = NULL;
  CHECK(et_f32_parameter_value_tensor_v1(parameter, &result, &error) == 0);
  CHECK(result != NULL);
  return result;
}

static const void *read_gradient_storage(et_f32_parameter *parameter,
                                         uint32_t *bits, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const void *result = NULL;
  CHECK(et_f32_parameter_gradient_borrow_begin_v1(parameter, &borrow, &error) ==
        0);
  CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL);
  if (view != NULL) {
    CHECK(view->byte_length == count * sizeof(*bits));
    if (view->byte_length == count * sizeof(*bits)) {
      memcpy(bits, view->data, count * sizeof(*bits));
    }
    result = view->data;
  }
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
  return result;
}

static const void *gradient_storage(et_f32_parameter *parameter,
                                    const uint32_t *expected, size_t count) {
  uint32_t actual[MAX_ELEMENTS] = {0u};
  const void *result = read_gradient_storage(parameter, actual, count);
  CHECK(memcmp(actual, expected, count * sizeof(*expected)) == 0);
  return result;
}

static void set_gradient(et_f32_parameter *parameter, const uint32_t *bits,
                         size_t count, uint64_t contributions,
                         uint32_t weight_bits) {
  et_f32_parameter_test_set_gradient_bits_v1(parameter, bits, count);
  et_f32_parameter_test_set_metadata_v1(parameter, ET_F32_GRADIENT_PRESENT,
                                        contributions, weight_bits);
}

static void accumulate_gradient(et_f32_parameter *parameter,
                                const uint32_t *numerator_bits, size_t count,
                                uint64_t ordinal, uint32_t weight_bits) {
  const uint64_t shape[1] = {(uint64_t)count};
  et_f32_tensor_error error;
  et_f32_tensor *numerator = make_tensor(1u, shape, numerator_bits, count);
  et_f32_gradient_plan *plan = NULL;
  et_f32_gradient_contribution_v1 contribution = {
      .struct_size = sizeof(contribution),
      .destination = parameter,
      .weighted_numerator = numerator,
      .expected_ordinal = ordinal};
  CHECK(et_f32_gradient_plan_prepare_v1(1u, &contribution, weight_bits, &plan,
                                        &error) == 0);
  CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
  CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
  destroy_tensor(&numerator);
}

static void model_create(model_fixture *model, size_t count,
                         const optimizer_config *config) {
  static const uint64_t shapes[N14][2] = {
      {4u, 4u}, {4u, 4u}, {4u, 4u}, {4u, 4u},   {4u, 8u}, {8u, 4u}, {4u, 0u},
      {4u, 0u}, {4u, 0u}, {4u, 0u}, {256u, 4u}, {4u, 0u}, {4u, 0u}, {2u, 4u}};
  static const size_t ranks[N14] = {2u, 2u, 2u, 2u, 2u, 2u, 1u,
                                    1u, 1u, 1u, 2u, 1u, 1u, 2u};
  const uint32_t one = UINT32_C(0x3f800000);
  et_o2_optimizer_builder *builder = NULL;
  et_o2_error_v1 o2_error;
  size_t index;

  memset(model, 0, sizeof(*model));
  model->count = count;
  CHECK(count == 1u || count == N14);
  CHECK(et_o2_optimizer_builder_create_v1(
            count, config->clip_kind, config->clip_max_bits,
            config->schedule_kind, config->warmup_updates,
            config->total_updates, config->minimum_ratio_bits, &builder,
            &o2_error) == 0);
  for (index = 0u; index < count; index++) {
    const size_t rank = count == 1u ? 1u : ranks[index];
    const uint64_t one_shape[1] = {1u};
    const uint64_t *shape = count == 1u ? one_shape : shapes[index];
    const size_t elements =
        count == 1u
            ? 1u
            : (rank == 1u ? (size_t)shape[0] : (size_t)(shape[0] * shape[1]));
    uint32_t bits[MAX_ELEMENTS];
    et_f32_tensor_error tensor_error;
    et_f32_tensor *initial;
    size_t element;
    CHECK(elements <= MAX_ELEMENTS);
    model->elements[index] = elements;
    for (element = 0u; element < elements; element++) {
      bits[element] = ((element + index) & 1u) == 0u ? UINT32_C(0x3f800000)
                                                     : UINT32_C(0xbf000000);
    }
    initial = make_tensor(rank, shape, bits, elements);
    CHECK(et_f32_parameter_create_v1(initial, &model->parameters[index],
                                     &tensor_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(model->parameters[index],
                                            model->parameters[index],
                                            &tensor_error) == 0);
    CHECK(et_o2_optimizer_builder_set_v1(
              builder, index, model->parameters[index],
              model->parameters[index],
              (index & 1u) == 0u ? UINT32_C(0x3c23d70a) : UINT32_C(0x3ba3d70a),
              UINT32_C(0x3f666666), UINT32_C(0x3f7fbe77), UINT32_C(0x322bcc77),
              (index % 3u) == 0u ? UINT32_C(0x3dcccccd) : 0u, &o2_error) == 0);
    destroy_tensor(&initial);
  }
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &model->optimizer,
                                          &o2_error) == 0);
  CHECK(builder == NULL);
  CHECK(model->optimizer != NULL);
  (void)one;
}

static void model_create_n1_options(model_fixture *model,
                                    const optimizer_config *config,
                                    uint32_t initial_bits,
                                    uint32_t learning_rate_bits,
                                    uint32_t weight_decay_bits) {
  const uint64_t shape[1] = {1u};
  et_f32_tensor_error tensor_error;
  et_o2_error_v1 o2_error;
  et_o2_optimizer_builder *builder = NULL;
  et_f32_tensor *initial;
  memset(model, 0, sizeof(*model));
  model->count = 1u;
  model->elements[0] = 1u;
  initial = make_tensor(1u, shape, &initial_bits, 1u);
  CHECK(et_f32_parameter_create_v1(initial, &model->parameters[0],
                                   &tensor_error) == 0);
  CHECK(et_f32_parameter_bind_identity_v1(
            model->parameters[0], model->parameters[0], &tensor_error) == 0);
  CHECK(et_o2_optimizer_builder_create_v1(
            1u, config->clip_kind, config->clip_max_bits, config->schedule_kind,
            config->warmup_updates, config->total_updates,
            config->minimum_ratio_bits, &builder, &o2_error) == 0);
  CHECK(et_o2_optimizer_builder_set_v1(
            builder, 0u, model->parameters[0], model->parameters[0],
            learning_rate_bits, UINT32_C(0x3f666666), UINT32_C(0x3f7fbe77),
            UINT32_C(0x322bcc77), weight_decay_bits, &o2_error) == 0);
  CHECK(et_o2_optimizer_builder_finish_v1(&builder, &model->optimizer,
                                          &o2_error) == 0);
  destroy_tensor(&initial);
}

static void model_set_gradients(model_fixture *model, uint64_t contributions,
                                uint32_t weight_bits, uint32_t magnitude_bits) {
  size_t index;
  for (index = 0u; index < model->count; index++) {
    uint32_t bits[MAX_ELEMENTS];
    size_t element;
    for (element = 0u; element < model->elements[index]; element++) {
      bits[element] = ((element + index) & 1u) == 0u
                          ? magnitude_bits
                          : (magnitude_bits ^ UINT32_C(0x80000000));
    }
    set_gradient(model->parameters[index], bits, model->elements[index],
                 contributions, weight_bits);
  }
}

static void model_compare(const model_fixture *left,
                          const model_fixture *right) {
  et_o2_error_v1 error;
  size_t index;
  CHECK(left->count == right->count);
  CHECK(completed_updates(left->optimizer) ==
        completed_updates(right->optimizer));
  for (index = 0u; index < left->count; index++) {
    uint32_t left_bits[MAX_ELEMENTS];
    uint32_t right_bits[MAX_ELEMENTS];
    uint32_t left_moment[MAX_ELEMENTS];
    uint32_t right_moment[MAX_ELEMENTS];
    uint32_t kind;
    CHECK(left->elements[index] == right->elements[index]);
    read_parameter_bits(left->parameters[index], left_bits,
                        left->elements[index]);
    read_parameter_bits(right->parameters[index], right_bits,
                        right->elements[index]);
    CHECK(memcmp(left_bits, right_bits,
                 left->elements[index] * sizeof(*left_bits)) == 0);
    for (kind = ET_O2_MOMENT_EXP_AVG; kind <= ET_O2_MOMENT_EXP_AVG_SQ; kind++) {
      CHECK(et_o2_test_optimizer_moment_bits_v1(
                left->optimizer, index, kind, left_moment,
                left->elements[index], &error) == 0);
      CHECK(et_o2_test_optimizer_moment_bits_v1(
                right->optimizer, index, kind, right_moment,
                right->elements[index], &error) == 0);
      CHECK(memcmp(left_moment, right_moment,
                   left->elements[index] * sizeof(*left_moment)) == 0);
    }
    {
      et_f32_gradient_metadata_v1 left_metadata =
          gradient_metadata(left->parameters[index]);
      et_f32_gradient_metadata_v1 right_metadata =
          gradient_metadata(right->parameters[index]);
      CHECK(left_metadata.state == ET_F32_GRADIENT_ABSENT);
      CHECK(left_metadata.contribution_count == 0u);
      CHECK(left_metadata.normalization_weight_bits == 0u);
      CHECK(memcmp(&left_metadata, &right_metadata, sizeof(left_metadata)) ==
            0);
    }
  }
}

static void private_update_clear(model_fixture *model, uint64_t current,
                                 uint64_t contributions, uint32_t weight_bits) {
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  CHECK(et_o2_trainer_step_clear_prepare_v1(model->optimizer, current,
                                            contributions, weight_bits, &plan,
                                            &error) == 0);
  CHECK(plan != NULL);
  CHECK(et_o2_trainer_step_clear_commit_v1(plan, &error) == 0);
}

static void public_update_clear(model_fixture *model) {
  et_o2_error_v1 error;
  CHECK(et_o2_optimizer_step_v1(model->optimizer, &error) == 0);
  CHECK(et_o2_optimizer_zero_grad_v1(model->optimizer, &error) == 0);
}

static optimizer_config constant_config(uint32_t clip_kind,
                                        uint32_t clip_max_bits) {
  optimizer_config result = {
      clip_kind, clip_max_bits,       ET_O2_SCHEDULE_CONSTANT, 0u,
      0u,        UINT32_C(0x3f800000)};
  return result;
}

static et_f32_test_live_counts_v1 i2_live_counts(void) {
  et_f32_test_live_counts_v1 result = {.struct_size = sizeof(result)};
  et_f32_test_live_counts_snapshot_v1(&result);
  return result;
}

static et_o2_trainer_step_clear_test_counts_v1 transaction_counts(void) {
  et_o2_trainer_step_clear_test_counts_v1 result = {.struct_size =
                                                        sizeof(result)};
  et_o2_trainer_step_clear_test_counts_snapshot_v1(&result);
  return result;
}

static et_f32_test_retired_counts_v1 i2_retired_counts(void) {
  et_f32_test_retired_counts_v1 result = {.struct_size = sizeof(result)};
  et_f32_test_retired_counts_snapshot_v1(&result);
  return result;
}

static void
check_i2_live_counts_equal(const et_f32_test_live_counts_v1 *left,
                           const et_f32_test_live_counts_v1 *right) {
  CHECK(left->tensors == right->tensors);
  CHECK(left->parameters == right->parameters);
  CHECK(left->borrows == right->borrows);
  CHECK(left->copy_plans == right->copy_plans);
  CHECK(left->gradient_plans == right->gradient_plans);
  CHECK(left->reset_plans == right->reset_plans);
  CHECK(left->owned_clones == right->owned_clones);
}

static void check_transaction_counts_equal(
    const et_o2_trainer_step_clear_test_counts_v1 *left,
    const et_o2_trainer_step_clear_test_counts_v1 *right) {
  CHECK(left->live_plans == right->live_plans);
  CHECK(left->live_stages == right->live_stages);
  CHECK(left->committed_plans == right->committed_plans);
  CHECK(left->aborted_plans == right->aborted_plans);
  CHECK(left->retained_control_bytes == right->retained_control_bytes);
}

typedef struct model_state_snapshot {
  uint64_t updates;
  uint32_t parameters[N14][MAX_ELEMENTS];
  uint32_t gradients[N14][MAX_ELEMENTS];
  uint32_t moments[N14][2][MAX_ELEMENTS];
  et_f32_gradient_metadata_v1 metadata[N14];
  const et_f32_tensor *value_identities[N14];
  const et_f32_tensor *moment_identities[N14][2];
  const void *gradient_identities[N14];
  et_f32_test_live_counts_v1 i2_counts;
  et_o2_trainer_step_clear_test_counts_v1 transaction_counts;
} model_state_snapshot;

static void capture_model_state(model_fixture *model,
                                model_state_snapshot *snapshot) {
  et_o2_error_v1 error;
  size_t index;
  memset(snapshot, 0, sizeof(*snapshot));
  snapshot->updates = completed_updates(model->optimizer);
  snapshot->i2_counts = i2_live_counts();
  snapshot->transaction_counts = transaction_counts();
  for (index = 0u; index < model->count; index++) {
    read_parameter_bits(model->parameters[index], snapshot->parameters[index],
                        model->elements[index]);
    snapshot->metadata[index] = gradient_metadata(model->parameters[index]);
    snapshot->value_identities[index] = value_storage(model->parameters[index]);
    snapshot->gradient_identities[index] = read_gradient_storage(
        model->parameters[index], snapshot->gradients[index],
        model->elements[index]);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG,
              snapshot->moments[index][0], model->elements[index],
              &error) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ,
              snapshot->moments[index][1], model->elements[index],
              &error) == 0);
    snapshot->moment_identities[index][0] =
        et_o2_trainer_step_clear_test_moment_storage_v1(model->optimizer, index,
                                                        ET_O2_MOMENT_EXP_AVG);
    snapshot->moment_identities[index][1] =
        et_o2_trainer_step_clear_test_moment_storage_v1(
            model->optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ);
  }
}

static void check_model_state_matches(model_fixture *model,
                                      const model_state_snapshot *snapshot,
                                      size_t skipped_gradient_index) {
  et_o2_error_v1 error;
  et_f32_test_live_counts_v1 i2_after = i2_live_counts();
  et_o2_trainer_step_clear_test_counts_v1 tx_after = transaction_counts();
  size_t index;
  CHECK(completed_updates(model->optimizer) == snapshot->updates);
  check_i2_live_counts_equal(&i2_after, &snapshot->i2_counts);
  check_transaction_counts_equal(&tx_after, &snapshot->transaction_counts);
  for (index = 0u; index < model->count; index++) {
    uint32_t actual[MAX_ELEMENTS];
    uint32_t moment[MAX_ELEMENTS];
    read_parameter_bits(model->parameters[index], actual,
                        model->elements[index]);
    CHECK(memcmp(actual, snapshot->parameters[index],
                 model->elements[index] * sizeof(*actual)) == 0);
    CHECK(value_storage(model->parameters[index]) ==
          snapshot->value_identities[index]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG) ==
          snapshot->moment_identities[index][0]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ) ==
          snapshot->moment_identities[index][1]);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG, moment,
              model->elements[index], &error) == 0);
    CHECK(memcmp(moment, snapshot->moments[index][0],
                 model->elements[index] * sizeof(*moment)) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ, moment,
              model->elements[index], &error) == 0);
    CHECK(memcmp(moment, snapshot->moments[index][1],
                 model->elements[index] * sizeof(*moment)) == 0);
    if (index != skipped_gradient_index) {
      et_f32_gradient_metadata_v1 metadata =
          gradient_metadata(model->parameters[index]);
      CHECK(memcmp(&metadata, &snapshot->metadata[index], sizeof(metadata)) ==
            0);
      CHECK(gradient_storage(model->parameters[index],
                             snapshot->gradients[index],
                             model->elements[index]) ==
            snapshot->gradient_identities[index]);
    }
  }
}

static void check_n1_state_unchanged(model_fixture *model,
                                     uint32_t parameter_bits,
                                     uint32_t gradient_bits,
                                     uint64_t contributions,
                                     uint32_t weight_bits, uint64_t updates) {
  et_o2_error_v1 error;
  et_f32_gradient_metadata_v1 metadata =
      gradient_metadata(model->parameters[0]);
  uint32_t actual = UINT32_MAX;
  uint32_t moment = UINT32_MAX;
  read_parameter_bits(model->parameters[0], &actual, 1u);
  CHECK(actual == parameter_bits);
  CHECK(metadata.state == ET_F32_GRADIENT_PRESENT);
  CHECK(metadata.contribution_count == contributions);
  CHECK(metadata.normalization_weight_bits == weight_bits);
  CHECK(gradient_storage(model->parameters[0], &gradient_bits, 1u) != NULL);
  CHECK(et_o2_test_optimizer_moment_bits_v1(model->optimizer, 0u,
                                            ET_O2_MOMENT_EXP_AVG, &moment, 1u,
                                            &error) == 0);
  CHECK(moment == 0u);
  CHECK(et_o2_test_optimizer_moment_bits_v1(model->optimizer, 0u,
                                            ET_O2_MOMENT_EXP_AVG_SQ, &moment,
                                            1u, &error) == 0);
  CHECK(moment == 0u);
  CHECK(completed_updates(model->optimizer) == updates);
}

static void test_public_parity_cases(void) {
  static const struct {
    uint32_t clip_kind;
    uint32_t clip_max_bits;
    uint32_t gradient_bits;
  } cases[] = {
      {ET_O2_CLIP_NONE, 0u, 0u},
      {ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x40a00000), UINT32_C(0x40000000)},
      {ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x40a00000), UINT32_C(0x40a00000)},
      {ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x40a00000), UINT32_C(0x41200000)}};
  size_t case_index;
  for (case_index = 0u; case_index < sizeof(cases) / sizeof(cases[0]);
       case_index++) {
    optimizer_config config = constant_config(cases[case_index].clip_kind,
                                              cases[case_index].clip_max_bits);
    model_fixture private_model;
    model_fixture public_model;
    model_create(&private_model, 1u, &config);
    model_create(&public_model, 1u, &config);
    model_set_gradients(&private_model, 1u, UINT32_C(0x3f800000),
                        cases[case_index].gradient_bits);
    model_set_gradients(&public_model, 1u, UINT32_C(0x3f800000),
                        cases[case_index].gradient_bits);
    private_update_clear(&private_model, 0u, 1u, UINT32_C(0x3f800000));
    public_update_clear(&public_model);
    model_compare(&private_model, &public_model);
  }

  {
    optimizer_config config = {ET_O2_CLIP_GLOBAL_L2,
                               UINT32_C(0x3f000000),
                               ET_O2_SCHEDULE_LINEAR,
                               2u,
                               6u,
                               UINT32_C(0x3dcccccd)};
    model_fixture private_model;
    model_fixture public_model;
    uint32_t first = UINT32_C(0x3f800000);
    uint32_t second = UINT32_C(0x40800000);
    model_create(&private_model, 1u, &config);
    model_create(&public_model, 1u, &config);
    CHECK(et_o2_test_optimizer_set_completed_updates_v1(private_model.optimizer,
                                                        2u) == 0);
    CHECK(et_o2_test_optimizer_set_completed_updates_v1(public_model.optimizer,
                                                        2u) == 0);
    accumulate_gradient(private_model.parameters[0], &first, 1u, 0u,
                        UINT32_C(0x3f800000));
    accumulate_gradient(private_model.parameters[0], &second, 1u, 1u,
                        UINT32_C(0x40000000));
    accumulate_gradient(public_model.parameters[0], &first, 1u, 0u,
                        UINT32_C(0x3f800000));
    accumulate_gradient(public_model.parameters[0], &second, 1u, 1u,
                        UINT32_C(0x40000000));
    private_update_clear(&private_model, 2u, 2u, UINT32_C(0x40400000));
    public_update_clear(&public_model);
    model_compare(&private_model, &public_model);
  }

  {
    static const uint64_t completed_before[] = {0u, 1u, 5u, 6u};
    optimizer_config config = {
        ET_O2_CLIP_NONE,     0u, ET_O2_SCHEDULE_LINEAR, 2u, 6u,
        UINT32_C(0x3dcccccd)};
    size_t index;
    for (index = 0u;
         index < sizeof(completed_before) / sizeof(completed_before[0]);
         index++) {
      model_fixture private_model;
      model_fixture public_model;
      model_create(&private_model, 1u, &config);
      model_create(&public_model, 1u, &config);
      CHECK(et_o2_test_optimizer_set_completed_updates_v1(
                private_model.optimizer, completed_before[index]) == 0);
      CHECK(et_o2_test_optimizer_set_completed_updates_v1(
                public_model.optimizer, completed_before[index]) == 0);
      model_set_gradients(&private_model, 1u, UINT32_C(0x3f800000),
                          UINT32_C(0x3f000000));
      model_set_gradients(&public_model, 1u, UINT32_C(0x3f800000),
                          UINT32_C(0x3f000000));
      private_update_clear(&private_model, completed_before[index], 1u,
                           UINT32_C(0x3f800000));
      public_update_clear(&public_model);
      model_compare(&private_model, &public_model);
      CHECK(completed_updates(private_model.optimizer) ==
            completed_before[index] + 1u);
    }
  }

  {
    optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
    model_fixture private_model;
    model_fixture public_model;
    model_create(&private_model, 1u, &config);
    model_create(&public_model, 1u, &config);
    CHECK(et_o2_test_optimizer_set_completed_updates_v1(
              private_model.optimizer, (uint64_t)INT64_MAX - 1u) == 0);
    CHECK(et_o2_test_optimizer_set_completed_updates_v1(
              public_model.optimizer, (uint64_t)INT64_MAX - 1u) == 0);
    model_set_gradients(&private_model, 1u, UINT32_C(0x3f800000),
                        UINT32_C(0x3f800000));
    model_set_gradients(&public_model, 1u, UINT32_C(0x3f800000),
                        UINT32_C(0x3f800000));
    private_update_clear(&private_model, (uint64_t)INT64_MAX - 1u, 1u,
                         UINT32_C(0x3f800000));
    public_update_clear(&public_model);
    model_compare(&private_model, &public_model);
    CHECK(completed_updates(private_model.optimizer) == (uint64_t)INT64_MAX);
  }
}

static void test_n14_wrong_weight_atomic_retry(void) {
  optimizer_config config = {
      ET_O2_CLIP_GLOBAL_L2, UINT32_C(0x3f000000), ET_O2_SCHEDULE_LINEAR, 2u, 6u,
      UINT32_C(0x3dcccccd)};
  model_fixture private_model;
  model_fixture clean_model;
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  uint32_t before_parameters[N14][MAX_ELEMENTS];
  uint32_t before_moments[N14][2][MAX_ELEMENTS];
  const et_f32_tensor *value_identities[N14];
  const et_f32_tensor *moment_identities[N14][2];
  const void *gradient_identities[N14];
  size_t index;

  model_create(&private_model, N14, &config);
  model_create(&clean_model, N14, &config);
  model_set_gradients(&private_model, 2u, UINT32_C(0x40400000),
                      UINT32_C(0x3f000000));
  model_set_gradients(&clean_model, 2u, UINT32_C(0x40400000),
                      UINT32_C(0x3f000000));

  for (index = 0u; index < N14; index++) {
    uint32_t expected[MAX_ELEMENTS];
    size_t element;
    read_parameter_bits(private_model.parameters[index],
                        before_parameters[index],
                        private_model.elements[index]);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG,
              before_moments[index][0], private_model.elements[index],
              &error) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ,
              before_moments[index][1], private_model.elements[index],
              &error) == 0);
    value_identities[index] = value_storage(private_model.parameters[index]);
    moment_identities[index][0] =
        et_o2_trainer_step_clear_test_moment_storage_v1(
            private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG);
    moment_identities[index][1] =
        et_o2_trainer_step_clear_test_moment_storage_v1(
            private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ);
    CHECK(moment_identities[index][0] != NULL);
    CHECK(moment_identities[index][1] != NULL);
    for (element = 0u; element < private_model.elements[index]; element++) {
      expected[element] = ((element + index) & 1u) == 0u ? UINT32_C(0x3f000000)
                                                         : UINT32_C(0xbf000000);
    }
    gradient_identities[index] =
        gradient_storage(private_model.parameters[index], expected,
                         private_model.elements[index]);
  }

  expect_o2_error(
      et_o2_trainer_step_clear_prepare_v1(private_model.optimizer, 0u, 2u,
                                          UINT32_C(0x40000000), &plan, &error),
      &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_GRADIENT_METADATA);
  CHECK(plan == NULL);
  CHECK(completed_updates(private_model.optimizer) == 0u);
  for (index = 0u; index < N14; index++) {
    uint32_t after[MAX_ELEMENTS];
    uint32_t expected[MAX_ELEMENTS];
    uint32_t moment[MAX_ELEMENTS];
    size_t element;
    et_f32_gradient_metadata_v1 metadata =
        gradient_metadata(private_model.parameters[index]);
    read_parameter_bits(private_model.parameters[index], after,
                        private_model.elements[index]);
    CHECK(memcmp(after, before_parameters[index],
                 private_model.elements[index] * sizeof(*after)) == 0);
    CHECK(value_storage(private_model.parameters[index]) ==
          value_identities[index]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG) ==
          moment_identities[index][0]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ) ==
          moment_identities[index][1]);
    CHECK(metadata.state == ET_F32_GRADIENT_PRESENT);
    CHECK(metadata.contribution_count == 2u);
    CHECK(metadata.normalization_weight_bits == UINT32_C(0x40400000));
    for (element = 0u; element < private_model.elements[index]; element++) {
      expected[element] = ((element + index) & 1u) == 0u ? UINT32_C(0x3f000000)
                                                         : UINT32_C(0xbf000000);
    }
    CHECK(gradient_storage(private_model.parameters[index], expected,
                           private_model.elements[index]) ==
          gradient_identities[index]);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG, moment,
              private_model.elements[index], &error) == 0);
    CHECK(memcmp(moment, before_moments[index][0],
                 private_model.elements[index] * sizeof(*moment)) == 0);
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ, moment,
              private_model.elements[index], &error) == 0);
    CHECK(memcmp(moment, before_moments[index][1],
                 private_model.elements[index] * sizeof(*moment)) == 0);
  }

  private_update_clear(&private_model, 0u, 2u, UINT32_C(0x40400000));
  public_update_clear(&clean_model);
  model_compare(&private_model, &clean_model);
  for (index = 0u; index < N14; index++) {
    uint32_t zeros[MAX_ELEMENTS] = {0u};
    CHECK(value_storage(private_model.parameters[index]) ==
          value_identities[index]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG) ==
          moment_identities[index][0]);
    CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
              private_model.optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ) ==
          moment_identities[index][1]);
    et_f32_parameter_test_set_metadata_v1(private_model.parameters[index],
                                          ET_F32_GRADIENT_PRESENT, 1u,
                                          UINT32_C(0x3f800000));
    CHECK(gradient_storage(private_model.parameters[index], zeros,
                           private_model.elements[index]) ==
          gradient_identities[index]);
  }
  CHECK(et_o2_optimizer_zero_grad_v1(private_model.optimizer, &error) == 0);
}

static void test_plan_lifetime_busy_abort_retry(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  et_o2_error_v1 error;
  et_f32_tensor_error i2_error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  et_o2_trainer_step_clear_plan *retry = NULL;
  et_o2_optimizer_state *state = NULL;
  et_f32_gradient_reset_plan *reset_plan = NULL;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_test_live_counts_v1 i2_before;
  et_f32_test_live_counts_v1 i2_prepared;
  et_o2_trainer_step_clear_test_counts_v1 tx_before;
  et_o2_trainer_step_clear_test_counts_v1 tx_prepared;
  et_o2_trainer_step_clear_test_counts_v1 tx_aborted;
  const et_f32_tensor *value_identity;
  const et_f32_tensor *moment_identities[2];
  const et_f32_tensor *stage_identities[3];
  const uint32_t gradient = UINT32_C(0x40000000);
  size_t stage;

  model_create(&model, 1u, &config);
  model_set_gradients(&model, 1u, UINT32_C(0x3f800000), gradient);
  value_identity = value_storage(model.parameters[0]);
  moment_identities[0] = et_o2_trainer_step_clear_test_moment_storage_v1(
      model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG);
  moment_identities[1] = et_o2_trainer_step_clear_test_moment_storage_v1(
      model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ);
  CHECK(moment_identities[0] != NULL);
  CHECK(moment_identities[1] != NULL);
  i2_before = i2_live_counts();
  tx_before = transaction_counts();
  CHECK(et_o2_trainer_step_clear_prepare_v1(
            model.optimizer, 0u, 1u, UINT32_C(0x3f800000), &plan, &error) == 0);
  CHECK(plan != NULL);
  i2_prepared = i2_live_counts();
  tx_prepared = transaction_counts();
  CHECK(i2_prepared.tensors == i2_before.tensors + 3u);
  CHECK(i2_prepared.copy_plans == i2_before.copy_plans + 1u);
  CHECK(i2_prepared.reset_plans == i2_before.reset_plans + 1u);
  CHECK(i2_prepared.borrows == i2_before.borrows);
  CHECK(tx_prepared.live_plans == tx_before.live_plans + 1u);
  CHECK(tx_prepared.live_stages == tx_before.live_stages + 3u);
  for (stage = 0u; stage < 3u; stage++) {
    stage_identities[stage] =
        et_o2_trainer_step_clear_test_stage_v1(plan, stage);
    CHECK(stage_identities[stage] != NULL);
  }
  CHECK(et_o2_trainer_step_clear_test_stage_v1(plan, 3u) == NULL);

  {
    const uint32_t overwrite = UINT32_C(0x40400000);
    et_f32_tensor *attempt;
    et_f32_parameter *parameter_attempt = model.parameters[0];
    size_t tensor_index;
    const et_f32_tensor *pinned_tensors[3] = {
        value_identity, moment_identities[0], moment_identities[1]};
    for (tensor_index = 0u; tensor_index < 3u; tensor_index++) {
      expect_i2_error(et_f32_tensor_copy_bits_from_v1(
                          (et_f32_tensor *)pinned_tensors[tensor_index],
                          &overwrite, 1u, &i2_error),
                      &i2_error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                      ET_F32_TENSOR_CODE_INVALID_HANDLE);
    }
    for (stage = 0u; stage < 3u; stage++) {
      attempt = (et_f32_tensor *)stage_identities[stage];
      expect_i2_error(et_f32_tensor_destroy_v1(&attempt, &i2_error), &i2_error,
                      ET_F32_TENSOR_ERROR_INVALID_STATE,
                      ET_F32_TENSOR_CODE_INVALID_HANDLE);
      CHECK(attempt == stage_identities[stage]);
    }
    expect_i2_error(et_f32_parameter_gradient_borrow_begin_v1(
                        model.parameters[0], &borrow, &i2_error),
                    &i2_error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                    ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(borrow == NULL);
    expect_i2_error(et_f32_parameter_destroy_v1(&parameter_attempt, &i2_error),
                    &i2_error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                    ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(parameter_attempt == model.parameters[0]);
    expect_i2_error(et_f32_gradient_reset_plan_prepare_v1(
                        1u, model.parameters, &reset_plan, &i2_error),
                    &i2_error, ET_F32_TENSOR_ERROR_INVALID_STATE,
                    ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(reset_plan == NULL);
  }

  expect_o2_error(et_o2_trainer_step_clear_prepare_v1(model.optimizer, 0u, 1u,
                                                      UINT32_C(0x3f800000),
                                                      &retry, &error),
                  &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  CHECK(retry == NULL);
  expect_o2_error(et_o2_optimizer_step_v1(model.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  expect_o2_error(et_o2_optimizer_zero_grad_v1(model.optimizer, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  expect_o2_error(
      et_o2_optimizer_state_snapshot_v1(model.optimizer, &state, &error),
      &error, ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_ACTIVE_BORROW);
  CHECK(state == NULL);
  CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
  tx_aborted = transaction_counts();
  CHECK(tx_aborted.live_plans == tx_before.live_plans);
  CHECK(tx_aborted.live_stages == tx_before.live_stages);
  CHECK(tx_aborted.aborted_plans == tx_before.aborted_plans + 1u);
  CHECK(tx_aborted.retained_control_bytes > tx_before.retained_control_bytes);
  {
    et_f32_test_live_counts_v1 after = i2_live_counts();
    CHECK(after.tensors == i2_before.tensors);
    CHECK(after.copy_plans == i2_before.copy_plans);
    CHECK(after.reset_plans == i2_before.reset_plans);
    CHECK(after.borrows == i2_before.borrows);
  }
  for (stage = 0u; stage < 3u; stage++) {
    CHECK(et_f32_tensor_is_live_v1(stage_identities[stage]) == 0);
  }
  {
    const et_f32_tensor *released_tensors[3] = {
        value_identity, moment_identities[0], moment_identities[1]};
    size_t tensor_index;
    for (tensor_index = 0u; tensor_index < 3u; tensor_index++) {
      CHECK(et_f32_tensor_borrow_begin_v1(
                (et_f32_tensor *)released_tensors[tensor_index], &borrow,
                &i2_error) == 0);
      CHECK(et_f32_tensor_borrow_end_v1(&borrow, &i2_error) == 0);
    }
    CHECK(et_f32_parameter_gradient_borrow_begin_v1(model.parameters[0],
                                                    &borrow, &i2_error) == 0);
    CHECK(et_f32_tensor_borrow_end_v1(&borrow, &i2_error) == 0);
    CHECK(et_f32_gradient_reset_plan_prepare_v1(1u, model.parameters,
                                                &reset_plan, &i2_error) == 0);
    CHECK(et_f32_gradient_reset_plan_release_v1(&reset_plan, &i2_error) == 0);
  }
  check_n1_state_unchanged(&model, UINT32_C(0x3f800000), gradient, 1u,
                           UINT32_C(0x3f800000), 0u);
  CHECK(value_storage(model.parameters[0]) == value_identity);
  CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
            model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG) == moment_identities[0]);
  CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
            model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ) ==
        moment_identities[1]);

  expect_o2_error(et_o2_trainer_step_clear_commit_v1(plan, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  expect_o2_error(et_o2_trainer_step_clear_abort_v1(plan, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  expect_o2_error(
      et_o2_trainer_step_clear_commit_v1(
          (et_o2_trainer_step_clear_plan *)(uintptr_t)0x12345u, &error),
      &error, ET_O2_STATUS_INVALID_ARGUMENT, ET_O2_CODE_INVALID_HANDLE);
  expect_o2_error(et_o2_trainer_step_clear_abort_v1(
                      (et_o2_trainer_step_clear_plan *)model.optimizer, &error),
                  &error, ET_O2_STATUS_INVALID_ARGUMENT,
                  ET_O2_CODE_INVALID_HANDLE);

  CHECK(et_o2_trainer_step_clear_prepare_v1(model.optimizer, 0u, 1u,
                                            UINT32_C(0x3f800000), &retry,
                                            &error) == 0);
  CHECK(retry != NULL);
  CHECK(retry != plan);
  et_o2_trainer_step_clear_test_fail_alloc_after_v1(0u);
  et_f32_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_o2_trainer_step_clear_commit_v1(retry, &error) == 0);
  et_o2_trainer_step_clear_test_reset_failpoints_v1();
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(completed_updates(model.optimizer) == 1u);
  CHECK(value_storage(model.parameters[0]) == value_identity);
  CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
            model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG) == moment_identities[0]);
  CHECK(et_o2_trainer_step_clear_test_moment_storage_v1(
            model.optimizer, 0u, ET_O2_MOMENT_EXP_AVG_SQ) ==
        moment_identities[1]);
  expect_o2_error(et_o2_trainer_step_clear_abort_v1(retry, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
  expect_o2_error(et_o2_trainer_step_clear_commit_v1(retry, &error), &error,
                  ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_CONSUMED);
}

static void test_prepare_allocation_and_publication_failpoints(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  const uint32_t gradient = UINT32_C(0x40000000);
  et_f32_test_live_counts_v1 i2_baseline;
  et_o2_trainer_step_clear_test_counts_v1 tx_baseline;
  size_t allowed;
  size_t failures_seen = 0u;
  size_t success_boundary = SIZE_MAX;
  int succeeded = 0;

  model_create(&model, 1u, &config);
  model_set_gradients(&model, 1u, UINT32_C(0x3f800000), gradient);
  i2_baseline = i2_live_counts();
  tx_baseline = transaction_counts();
  for (allowed = 0u; allowed < 64u; allowed++) {
    et_o2_error_v1 error;
    et_o2_trainer_step_clear_plan *plan = NULL;
    int32_t result;
    et_o2_trainer_step_clear_test_fail_alloc_after_v1(allowed);
    result = et_o2_trainer_step_clear_prepare_v1(
        model.optimizer, 0u, 1u, UINT32_C(0x3f800000), &plan, &error);
    et_o2_trainer_step_clear_test_reset_failpoints_v1();
    if (result == 0) {
      CHECK(plan != NULL);
      CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
      success_boundary = allowed;
      succeeded = 1;
      break;
    }
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(allowed == failures_seen);
    failures_seen++;
    CHECK(plan == NULL);
    check_n1_state_unchanged(&model, UINT32_C(0x3f800000), gradient, 1u,
                             UINT32_C(0x3f800000), 0u);
    {
      et_f32_test_live_counts_v1 i2_after = i2_live_counts();
      et_o2_trainer_step_clear_test_counts_v1 tx_after = transaction_counts();
      CHECK(i2_after.tensors == i2_baseline.tensors);
      CHECK(i2_after.copy_plans == i2_baseline.copy_plans);
      CHECK(i2_after.reset_plans == i2_baseline.reset_plans);
      CHECK(i2_after.borrows == i2_baseline.borrows);
      CHECK(tx_after.live_plans == tx_baseline.live_plans);
      CHECK(tx_after.live_stages == tx_baseline.live_stages);
    }
  }
  CHECK(succeeded != 0);
  CHECK(failures_seen > 0u);
  CHECK(success_boundary == failures_seen);
  CHECK(success_boundary == 4u);
  {
    et_f32_test_live_counts_v1 i2_after = i2_live_counts();
    et_o2_trainer_step_clear_test_counts_v1 tx_after = transaction_counts();
    check_i2_live_counts_equal(&i2_after, &i2_baseline);
    CHECK(tx_after.live_plans == tx_baseline.live_plans);
    CHECK(tx_after.live_stages == tx_baseline.live_stages);
  }

  succeeded = 0;
  failures_seen = 0u;
  success_boundary = SIZE_MAX;
  for (allowed = 0u; allowed < 128u; allowed++) {
    et_o2_error_v1 error;
    et_o2_trainer_step_clear_plan *plan = NULL;
    int32_t result;
    et_f32_tensor_test_fail_alloc_after_v1(allowed);
    result = et_o2_trainer_step_clear_prepare_v1(
        model.optimizer, 0u, 1u, UINT32_C(0x3f800000), &plan, &error);
    et_f32_tensor_test_reset_allocator_v1();
    if (result == 0) {
      CHECK(plan != NULL);
      CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
      success_boundary = allowed;
      succeeded = 1;
      break;
    }
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
    CHECK(allowed == failures_seen);
    failures_seen++;
    CHECK(plan == NULL);
    check_n1_state_unchanged(&model, UINT32_C(0x3f800000), gradient, 1u,
                             UINT32_C(0x3f800000), 0u);
    {
      et_f32_test_live_counts_v1 after = i2_live_counts();
      CHECK(after.tensors == i2_baseline.tensors);
      CHECK(after.copy_plans == i2_baseline.copy_plans);
      CHECK(after.reset_plans == i2_baseline.reset_plans);
      CHECK(after.borrows == i2_baseline.borrows);
    }
  }
  CHECK(succeeded != 0);
  CHECK(failures_seen > 0u);
  CHECK(success_boundary == failures_seen);
  CHECK(success_boundary == 24u);
  {
    et_f32_test_live_counts_v1 i2_after = i2_live_counts();
    et_o2_trainer_step_clear_test_counts_v1 tx_after = transaction_counts();
    check_i2_live_counts_equal(&i2_after, &i2_baseline);
    CHECK(tx_after.live_plans == tx_baseline.live_plans);
    CHECK(tx_after.live_stages == tx_baseline.live_stages);
  }

  {
    et_o2_error_v1 error;
    et_o2_trainer_step_clear_plan *plan = NULL;
    et_o2_trainer_step_clear_test_fail_publication_v1(1);
    expect_o2_error(
        et_o2_trainer_step_clear_prepare_v1(
            model.optimizer, 0u, 1u, UINT32_C(0x3f800000), &plan, &error),
        &error, ET_O2_STATUS_INTERNAL, ET_O2_CODE_ALLOCATION_FAILED);
    et_o2_trainer_step_clear_test_reset_failpoints_v1();
    CHECK(plan == NULL);
    check_n1_state_unchanged(&model, UINT32_C(0x3f800000), gradient, 1u,
                             UINT32_C(0x3f800000), 0u);
    CHECK(et_o2_trainer_step_clear_prepare_v1(model.optimizer, 0u, 1u,
                                              UINT32_C(0x3f800000), &plan,
                                              &error) == 0);
    CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
  }
}

static void expect_prepare_failure(model_fixture *model, uint64_t current,
                                   uint64_t contributions, uint32_t weight_bits,
                                   uint32_t category, uint32_t code) {
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  expect_o2_error(
      et_o2_trainer_step_clear_prepare_v1(
          model->optimizer, current, contributions, weight_bits, &plan, &error),
      &error, category, code);
  CHECK(plan == NULL);
}

static void retry_abort(model_fixture *model, uint64_t current,
                        uint64_t contributions, uint32_t weight_bits) {
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  CHECK(et_o2_trainer_step_clear_prepare_v1(model->optimizer, current,
                                            contributions, weight_bits, &plan,
                                            &error) == 0);
  CHECK(plan != NULL);
  CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
}

static void test_prepare_argument_and_numeric_negatives(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  const uint32_t gradient = UINT32_C(0x40000000);
  static const uint32_t invalid_weights[] = {
      0u, UINT32_C(0x80000000), UINT32_C(0xbf800000), UINT32_C(0x7f800000),
      UINT32_C(0x7fc00000)};
  size_t index;

  model_create(&model, 1u, &config);
  CHECK(et_o2_test_optimizer_set_completed_updates_v1(model.optimizer, 1u) ==
        0);
  model_set_gradients(&model, 1u, UINT32_C(0x3f800000), gradient);
  expect_prepare_failure(&model, 0u, 1u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_STATE,
                         ET_O2_CODE_COUNTER_MISMATCH);
  expect_prepare_failure(&model, 2u, 1u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_STATE,
                         ET_O2_CODE_COUNTER_MISMATCH);
  expect_prepare_failure(&model, (uint64_t)INT64_MAX, 1u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_ARGUMENT,
                         ET_O2_CODE_INVALID_OPTION);
  expect_prepare_failure(&model, 1u, 0u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_ARGUMENT,
                         ET_O2_CODE_INVALID_OPTION);
  expect_prepare_failure(&model, 1u, 2u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_STATE,
                         ET_O2_CODE_GRADIENT_METADATA);
  for (index = 0u; index < sizeof(invalid_weights) / sizeof(invalid_weights[0]);
       index++) {
    expect_prepare_failure(&model, 1u, 1u, invalid_weights[index],
                           ET_O2_STATUS_INVALID_ARGUMENT,
                           ET_O2_CODE_INVALID_OPTION);
  }
  check_n1_state_unchanged(&model, UINT32_C(0x3f800000), gradient, 1u,
                           UINT32_C(0x3f800000), 1u);
  retry_abort(&model, 1u, 1u, UINT32_C(0x3f800000));

  et_f32_parameter_test_set_metadata_v1(model.parameters[0],
                                        ET_F32_GRADIENT_ABSENT, 0u, 0u);
  expect_prepare_failure(&model, 1u, 1u, UINT32_C(0x3f800000),
                         ET_O2_STATUS_INVALID_STATE,
                         ET_O2_CODE_GRADIENT_ABSENT);
  set_gradient(model.parameters[0], &gradient, 1u, 1u, UINT32_C(0x3f800000));
  retry_abort(&model, 1u, 1u, UINT32_C(0x3f800000));

  {
    const uint32_t infinity = UINT32_C(0x7f800000);
    set_gradient(model.parameters[0], &infinity, 1u, 1u, UINT32_C(0x3f800000));
    expect_prepare_failure(&model, 1u, 1u, UINT32_C(0x3f800000),
                           ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE);
    set_gradient(model.parameters[0], &gradient, 1u, 1u, UINT32_C(0x3f800000));
    retry_abort(&model, 1u, 1u, UINT32_C(0x3f800000));
  }

  {
    const uint32_t zero = 0u;
    set_gradient(model.parameters[0], &zero, 1u, 1u, 1u);
    retry_abort(&model, 1u, 1u, 1u);
  }

  CHECK(et_o2_test_optimizer_set_completed_updates_v1(
            model.optimizer, (uint64_t)INT64_MAX) == 0);
  set_gradient(model.parameters[0], &gradient, 1u, 1u, UINT32_C(0x3f800000));
  expect_prepare_failure(&model, (uint64_t)INT64_MAX - 1u, 1u,
                         UINT32_C(0x3f800000), ET_O2_STATUS_INVALID_STATE,
                         ET_O2_CODE_COUNTER_OVERFLOW);

  {
    model_fixture overflow;
    const uint32_t zero = 0u;
    model_create_n1_options(&overflow, &config, UINT32_C(0x7f7fffff),
                            UINT32_C(0x7f7fffff), UINT32_C(0x7f7fffff));
    set_gradient(overflow.parameters[0], &zero, 1u, 1u, UINT32_C(0x3f800000));
    expect_prepare_failure(&overflow, 0u, 1u, UINT32_C(0x3f800000),
                           ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE);
    CHECK(completed_updates(overflow.optimizer) == 0u);
    CHECK(gradient_metadata(overflow.parameters[0]).state ==
          ET_F32_GRADIENT_PRESENT);
  }
}

static void test_n14_first_middle_last_metadata_negatives(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  static const size_t positions[] = {0u, N14 / 2u, N14 - 1u};
  const uint32_t gradient = UINT32_C(0x3f000000);
  size_t position_index;
  model_create(&model, N14, &config);
  for (position_index = 0u;
       position_index < sizeof(positions) / sizeof(positions[0]);
       position_index++) {
    const size_t position = positions[position_index];
    model_state_snapshot snapshot;
    et_f32_gradient_metadata_v1 metadata;
    et_f32_test_retired_counts_v1 retired_before;
    et_f32_test_retired_counts_v1 retired_after;
    uint32_t poison[MAX_ELEMENTS];
    size_t element;

    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    capture_model_state(&model, &snapshot);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_ABSENT, 0u, 0u);
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE,
                           ET_O2_CODE_GRADIENT_ABSENT);
    check_model_state_matches(&model, &snapshot, position);
    metadata = gradient_metadata(model.parameters[position]);
    CHECK(metadata.state == ET_F32_GRADIENT_ABSENT);
    CHECK(metadata.contribution_count == 0u);
    CHECK(metadata.normalization_weight_bits == 0u);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 2u,
                                          UINT32_C(0x40400000));
    check_model_state_matches(&model, &snapshot, SIZE_MAX);
    retry_abort(&model, 0u, 2u, UINT32_C(0x40400000));

    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    for (element = 0u; element < model.elements[position]; element++) {
      poison[element] = element == 0u ? UINT32_C(0x7f800000) : gradient;
    }
    set_gradient(model.parameters[position], poison, model.elements[position],
                 2u, UINT32_C(0x40400000));
    capture_model_state(&model, &snapshot);
    retired_before = i2_retired_counts();
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE);
    retired_after = i2_retired_counts();
    CHECK(retired_after.tensors == retired_before.tensors);
    check_model_state_matches(&model, &snapshot, SIZE_MAX);
    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    retry_abort(&model, 0u, 2u, UINT32_C(0x40400000));

    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    for (element = 0u; element < model.elements[position]; element++) {
      poison[element] = element == 0u ? UINT32_C(0x7f7fffff) : 0u;
    }
    set_gradient(model.parameters[position], poison, model.elements[position],
                 2u, UINT32_C(0x40400000));
    capture_model_state(&model, &snapshot);
    retired_before = i2_retired_counts();
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE, ET_O2_CODE_NONFINITE);
    retired_after = i2_retired_counts();
    CHECK(retired_after.tensors - retired_before.tensors ==
          (position + 1u) * 3u);
    check_model_state_matches(&model, &snapshot, SIZE_MAX);
    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    retry_abort(&model, 0u, 2u, UINT32_C(0x40400000));

    model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
    capture_model_state(&model, &snapshot);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 3u,
                                          UINT32_C(0x40400000));
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE,
                           ET_O2_CODE_GRADIENT_METADATA);
    snapshot.metadata[position].contribution_count = 3u;
    check_model_state_matches(&model, &snapshot, SIZE_MAX);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 2u,
                                          UINT32_C(0x40000000));
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE,
                           ET_O2_CODE_GRADIENT_METADATA);
    snapshot.metadata[position].contribution_count = 2u;
    snapshot.metadata[position].normalization_weight_bits =
        UINT32_C(0x40000000);
    check_model_state_matches(&model, &snapshot, SIZE_MAX);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 2u,
                                          UINT32_C(0x40400000));
    retry_abort(&model, 0u, 2u, UINT32_C(0x40400000));
  }
}

static void expect_terminal_site(uint32_t site, size_t stage_index) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  et_o2_error_v1 error;
  et_o2_trainer_step_clear_plan *plan = NULL;
  pid_t child;
  int status = 0;
  model_create(&model, 1u, &config);
  model_set_gradients(&model, 1u, UINT32_C(0x3f800000), UINT32_C(0x40000000));
  CHECK(et_o2_trainer_step_clear_prepare_v1(
            model.optimizer, 0u, 1u, UINT32_C(0x3f800000), &plan, &error) == 0);
  child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    et_o2_trainer_step_clear_test_fail_terminal_v1(site, stage_index);
    (void)et_o2_trainer_step_clear_commit_v1(plan, &error);
    _Exit(99);
  }
  if (child > 0) {
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 134);
  }
  CHECK(et_o2_trainer_step_clear_abort_v1(plan, &error) == 0);
}

static void test_terminal_failstop_sites(void) {
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_COPY_COMMIT, 0u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_RESET_COMMIT, 0u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_RESET_RELEASE, 0u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_COPY_RELEASE, 0u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_STAGE_DESTROY, 0u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_STAGE_DESTROY, 1u);
  expect_terminal_site(ET_O2_TR3_TEST_TERMINAL_STAGE_DESTROY, 2u);
}

static uint64_t digest_u64(uint64_t digest, uint64_t value) {
  size_t index;
  for (index = 0u; index < sizeof(value); index++) {
    digest ^= (value >> (index * 8u)) & UINT64_C(0xff);
    digest *= UINT64_C(1099511628211);
  }
  return digest;
}

static uint64_t model_n1_digest(model_fixture *model) {
  et_o2_error_v1 error;
  et_f32_gradient_metadata_v1 metadata =
      gradient_metadata(model->parameters[0]);
  uint32_t parameter = 0u;
  uint32_t moment = 0u;
  uint64_t digest = UINT64_C(1469598103934665603);
  read_parameter_bits(model->parameters[0], &parameter, 1u);
  digest = digest_u64(digest, parameter);
  CHECK(et_o2_test_optimizer_moment_bits_v1(model->optimizer, 0u,
                                            ET_O2_MOMENT_EXP_AVG, &moment, 1u,
                                            &error) == 0);
  digest = digest_u64(digest, moment);
  CHECK(et_o2_test_optimizer_moment_bits_v1(model->optimizer, 0u,
                                            ET_O2_MOMENT_EXP_AVG_SQ, &moment,
                                            1u, &error) == 0);
  digest = digest_u64(digest, moment);
  digest = digest_u64(digest, completed_updates(model->optimizer));
  digest = digest_u64(digest, metadata.state);
  digest = digest_u64(digest, metadata.contribution_count);
  return digest_u64(digest, metadata.normalization_weight_bits);
}

static void test_retained_control_slopes(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  et_o2_trainer_step_clear_test_counts_v1 before;
  et_o2_trainer_step_clear_test_counts_v1 at_1 = {.struct_size = sizeof(at_1)};
  et_o2_trainer_step_clear_test_counts_v1 at_1024 = {.struct_size =
                                                         sizeof(at_1024)};
  et_o2_trainer_step_clear_test_counts_v1 at_8192 = {.struct_size =
                                                         sizeof(at_8192)};
  et_f32_test_live_counts_v1 i2_live_before;
  et_f32_test_live_counts_v1 i2_live_at_1;
  et_f32_test_live_counts_v1 i2_live_at_1024;
  et_f32_test_live_counts_v1 i2_live_at_8192;
  et_f32_test_retired_counts_v1 i2_retired_before;
  et_f32_test_retired_counts_v1 i2_retired_at_1;
  et_f32_test_retired_counts_v1 i2_retired_at_1024;
  et_f32_test_retired_counts_v1 i2_retired_at_8192;
  const uint32_t gradient = UINT32_C(0x3f800000);
  uint64_t update;

  model_create(&model, 1u, &config);
  before = transaction_counts();
  i2_live_before = i2_live_counts();
  i2_retired_before = i2_retired_counts();
  for (update = 0u; update < 8192u; update++) {
    set_gradient(model.parameters[0], &gradient, 1u, 1u, UINT32_C(0x3f800000));
    private_update_clear(&model, update, 1u, UINT32_C(0x3f800000));
    if (update == 0u) {
      at_1 = transaction_counts();
      i2_live_at_1 = i2_live_counts();
      i2_retired_at_1 = i2_retired_counts();
    } else if (update == 1023u) {
      at_1024 = transaction_counts();
      i2_live_at_1024 = i2_live_counts();
      i2_retired_at_1024 = i2_retired_counts();
    }
  }
  at_8192 = transaction_counts();
  i2_live_at_8192 = i2_live_counts();
  i2_retired_at_8192 = i2_retired_counts();
  CHECK(at_1.committed_plans == before.committed_plans + 1u);
  CHECK(at_1024.committed_plans == before.committed_plans + 1024u);
  CHECK(at_8192.committed_plans == before.committed_plans + 8192u);
  CHECK(at_1.live_plans == before.live_plans);
  CHECK(at_1024.live_plans == before.live_plans);
  CHECK(at_8192.live_plans == before.live_plans);
  CHECK(at_1.live_stages == before.live_stages);
  CHECK(at_1024.live_stages == before.live_stages);
  CHECK(at_8192.live_stages == before.live_stages);
  outer_retained_bytes_per_update =
      at_1.retained_control_bytes - before.retained_control_bytes;
  CHECK(outer_retained_bytes_per_update > 0u);
  CHECK(at_1024.retained_control_bytes - before.retained_control_bytes ==
        1024u * outer_retained_bytes_per_update);
  CHECK(at_8192.retained_control_bytes - before.retained_control_bytes ==
        8192u * outer_retained_bytes_per_update);
  check_i2_live_counts_equal(&i2_live_at_1, &i2_live_before);
  check_i2_live_counts_equal(&i2_live_at_1024, &i2_live_before);
  check_i2_live_counts_equal(&i2_live_at_8192, &i2_live_before);
#define CHECK_RETIRED_LINEAR(field)                                            \
  do {                                                                         \
    const size_t slope = i2_retired_at_1.field - i2_retired_before.field;      \
    CHECK(i2_retired_at_1024.field - i2_retired_before.field ==                \
          1024u * slope);                                                      \
    CHECK(i2_retired_at_8192.field - i2_retired_before.field ==                \
          8192u * slope);                                                      \
  } while (0)
  CHECK_RETIRED_LINEAR(tensors);
  CHECK_RETIRED_LINEAR(parameters);
  CHECK_RETIRED_LINEAR(borrows);
  CHECK_RETIRED_LINEAR(copy_plans);
  CHECK_RETIRED_LINEAR(gradient_plans);
  CHECK_RETIRED_LINEAR(reset_plans);
  CHECK_RETIRED_LINEAR(retained_control_bytes);
#undef CHECK_RETIRED_LINEAR
  CHECK(i2_retired_at_1.tensors - i2_retired_before.tensors == 3u);
  CHECK(i2_retired_at_1.parameters == i2_retired_before.parameters);
  CHECK(i2_retired_at_1.copy_plans - i2_retired_before.copy_plans == 1u);
  CHECK(i2_retired_at_1.gradient_plans == i2_retired_before.gradient_plans);
  CHECK(i2_retired_at_1.reset_plans - i2_retired_before.reset_plans == 1u);
  i2_retained_bytes_per_update = i2_retired_at_1.retained_control_bytes -
                                 i2_retired_before.retained_control_bytes;
  CHECK(i2_retained_bytes_per_update > 0u);
  CHECK(completed_updates(model.optimizer) == 8192u);
  repeated_state_digest = model_n1_digest(&model);
}

int main(void) {
  test_public_parity_cases();
  test_n14_wrong_weight_atomic_retry();
  test_plan_lifetime_busy_abort_retry();
  test_prepare_allocation_and_publication_failpoints();
  test_prepare_argument_and_numeric_negatives();
  test_n14_first_middle_last_metadata_negatives();
  test_terminal_failstop_sites();
  test_retained_control_slopes();
  if (failures != 0u) {
    (void)fprintf(
        stderr, "TR3-O update-clear adversarial FAIL: %zu/%zu checks failed\n",
        failures, checks);
    return 1;
  }
  (void)printf("TR3-O retained-control slope: outer=%zu i2=%zu bytes/update\n",
               outer_retained_bytes_per_update, i2_retained_bytes_per_update);
  (void)printf("TR3-O repeated-state digest: %016" PRIx64 "\n",
               repeated_state_digest);
  (void)printf("TR3-O update-clear adversarial PASS: %zu checks\n", checks);
  return 0;
}
