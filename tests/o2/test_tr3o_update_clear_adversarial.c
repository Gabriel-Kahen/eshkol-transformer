#include "o2_optimizer_internal.h"
#include "tr3_o2_step_clear_internal.h"

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

static const void *gradient_storage(et_f32_parameter *parameter,
                                    const uint32_t *expected, size_t count) {
  et_f32_tensor_error error;
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const void *result = NULL;
  CHECK(et_f32_parameter_gradient_borrow_begin_v1(parameter, &borrow, &error) ==
        0);
  CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL);
  if (view != NULL) {
    CHECK(view->byte_length == count * sizeof(*expected));
    CHECK(memcmp(view->data, expected, count * sizeof(*expected)) == 0);
    result = view->data;
  }
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
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
  et_o2_trainer_step_clear_plan *plan = NULL;
  et_o2_trainer_step_clear_plan *retry = NULL;
  et_o2_optimizer_state *state = NULL;
  et_f32_test_live_counts_v1 i2_before;
  et_f32_test_live_counts_v1 i2_prepared;
  et_o2_trainer_step_clear_test_counts_v1 tx_before;
  et_o2_trainer_step_clear_test_counts_v1 tx_prepared;
  et_o2_trainer_step_clear_test_counts_v1 tx_aborted;
  const et_f32_tensor *value_identity;
  const et_f32_tensor *moment_identities[2];
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
    CHECK(et_o2_trainer_step_clear_test_stage_v1(plan, stage) != NULL);
  }
  CHECK(et_o2_trainer_step_clear_test_stage_v1(plan, 3u) == NULL);

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
      succeeded = 1;
      break;
    }
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
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

  succeeded = 0;
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
      succeeded = 1;
      break;
    }
    expect_o2_error(result, &error, ET_O2_STATUS_INTERNAL,
                    ET_O2_CODE_ALLOCATION_FAILED);
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

static void check_n14_initial_state(model_fixture *model) {
  et_o2_error_v1 error;
  size_t index;
  CHECK(completed_updates(model->optimizer) == 0u);
  for (index = 0u; index < N14; index++) {
    uint32_t actual[MAX_ELEMENTS];
    uint32_t moment[MAX_ELEMENTS];
    size_t element;
    read_parameter_bits(model->parameters[index], actual,
                        model->elements[index]);
    for (element = 0u; element < model->elements[index]; element++) {
      CHECK(actual[element] == (((element + index) & 1u) == 0u
                                    ? UINT32_C(0x3f800000)
                                    : UINT32_C(0xbf000000)));
    }
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG, moment,
              model->elements[index], &error) == 0);
    for (element = 0u; element < model->elements[index]; element++) {
      CHECK(moment[element] == 0u);
    }
    CHECK(et_o2_test_optimizer_moment_bits_v1(
              model->optimizer, index, ET_O2_MOMENT_EXP_AVG_SQ, moment,
              model->elements[index], &error) == 0);
    for (element = 0u; element < model->elements[index]; element++) {
      CHECK(moment[element] == 0u);
    }
  }
}

static void test_n14_first_middle_last_metadata_negatives(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  static const size_t positions[] = {0u, N14 / 2u, N14 - 1u};
  const uint32_t gradient = UINT32_C(0x3f000000);
  size_t position_index;
  model_create(&model, N14, &config);
  model_set_gradients(&model, 2u, UINT32_C(0x40400000), gradient);
  for (position_index = 0u;
       position_index < sizeof(positions) / sizeof(positions[0]);
       position_index++) {
    const size_t position = positions[position_index];
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 3u,
                                          UINT32_C(0x40400000));
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE,
                           ET_O2_CODE_GRADIENT_METADATA);
    check_n14_initial_state(&model);
    et_f32_parameter_test_set_metadata_v1(model.parameters[position],
                                          ET_F32_GRADIENT_PRESENT, 2u,
                                          UINT32_C(0x40000000));
    expect_prepare_failure(&model, 0u, 2u, UINT32_C(0x40400000),
                           ET_O2_STATUS_INVALID_STATE,
                           ET_O2_CODE_GRADIENT_METADATA);
    check_n14_initial_state(&model);
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

static void test_retained_control_slopes(void) {
  optimizer_config config = constant_config(ET_O2_CLIP_NONE, 0u);
  model_fixture model;
  et_o2_trainer_step_clear_test_counts_v1 before;
  et_o2_trainer_step_clear_test_counts_v1 at_1024 = {.struct_size =
                                                         sizeof(at_1024)};
  et_o2_trainer_step_clear_test_counts_v1 at_8192 = {.struct_size =
                                                         sizeof(at_8192)};
  const uint32_t gradient = UINT32_C(0x3f800000);
  uint64_t update;

  model_create(&model, 1u, &config);
  before = transaction_counts();
  for (update = 0u; update < 8192u; update++) {
    set_gradient(model.parameters[0], &gradient, 1u, 1u, UINT32_C(0x3f800000));
    private_update_clear(&model, update, 1u, UINT32_C(0x3f800000));
    if (update == 1023u) {
      at_1024 = transaction_counts();
    }
  }
  at_8192 = transaction_counts();
  CHECK(at_1024.committed_plans == before.committed_plans + 1024u);
  CHECK(at_8192.committed_plans == before.committed_plans + 8192u);
  CHECK(at_1024.live_plans == before.live_plans);
  CHECK(at_8192.live_plans == before.live_plans);
  CHECK(at_1024.live_stages == before.live_stages);
  CHECK(at_8192.live_stages == before.live_stages);
  CHECK(at_1024.retained_control_bytes > before.retained_control_bytes);
  CHECK((at_1024.retained_control_bytes - before.retained_control_bytes) %
            1024u ==
        0u);
  CHECK(at_8192.retained_control_bytes - before.retained_control_bytes ==
        8u * (at_1024.retained_control_bytes - before.retained_control_bytes));
  CHECK(completed_updates(model.optimizer) == 8192u);
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
  (void)printf("TR3-O update-clear adversarial PASS: %zu checks\n", checks);
  return 0;
}
