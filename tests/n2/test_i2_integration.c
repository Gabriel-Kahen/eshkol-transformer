#include "eshkol_transformer/f32_tensor.h"
#include "eshkol_transformer/i64_tensor.h"
#include "eshkol_transformer/n2_primitives_abi.h"
#include "f32_parameter_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

typedef struct f32_lease {
  et_f32_tensor_borrow *borrow;
  const et_kernel_tensor_view_v1 *view;
} f32_lease;

typedef struct i64_lease {
  et_i64_tensor_borrow *borrow;
  const et_kernel_tensor_view_v1 *view;
} i64_lease;

static uint32_t float_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static et_f32_tensor *make_f32(size_t rank, const uint64_t *shape,
                               const float *values, size_t count) {
  et_f32_tensor *tensor = NULL;
  et_f32_tensor_error error;
  uint32_t bits[64];
  CHECK(count <= sizeof(bits) / sizeof(bits[0]));
  for (size_t index = 0; index < count; index++)
    bits[index] = float_bits(values[index]);
  CHECK(et_f32_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, count, &error) == 0);
  return tensor;
}

static et_i64_tensor *make_i64(size_t rank, const uint64_t *shape,
                               const int64_t *values, size_t count) {
  et_i64_tensor *tensor = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(et_i64_tensor_copy_from_v1(tensor, values, count, &error) == 0);
  return tensor;
}

static f32_lease borrow_f32(et_f32_tensor *tensor) {
  f32_lease lease = {0};
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_borrow_begin_v1(tensor, &lease.borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(lease.borrow, &lease.view, &error) == 0);
  CHECK(lease.view != NULL);
  return lease;
}

static i64_lease borrow_i64(et_i64_tensor *tensor) {
  i64_lease lease = {0};
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &lease.borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(lease.borrow, &lease.view, &error) == 0);
  CHECK(lease.view != NULL);
  return lease;
}

static void end_f32(f32_lease *lease) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_borrow_end_v1(&lease->borrow, &error) == 0);
  CHECK(lease->borrow == NULL);
  lease->view = NULL;
}

static void end_i64(i64_lease *lease) {
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_borrow_end_v1(&lease->borrow, &error) == 0);
  CHECK(lease->borrow == NULL);
  lease->view = NULL;
}

static void destroy_f32(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void destroy_i64(et_i64_tensor **tensor) {
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void expect_f32(const et_f32_tensor *tensor, const float *expected,
                       size_t count) {
  et_f32_tensor_error error;
  uint32_t actual[64];
  CHECK(count <= sizeof(actual) / sizeof(actual[0]));
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, actual, count, &error) == 0);
  for (size_t index = 0; index < count; index++)
    CHECK(actual[index] == float_bits(expected[index]));
}

static const et_kernel_provider_v1 *resolve_n2(void *context,
                                               const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_n2_kernel_provider_v1();
}

static et_kernel_runtime *make_runtime(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_n2, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static void dispatch(et_kernel_runtime *runtime, const char *capability,
                     const char *operation, size_t rank,
                     const uint64_t *shape, et_kernel_tensor_view_v1 *inputs,
                     size_t input_count, et_kernel_tensor_view_v1 *outputs,
                     size_t output_count) {
  et_kernel_request_v1 request = {sizeof(request), operation, "f32", "cpu",
                                  rank,            shape,     1u,    {0}};
  et_kernel_call_v1 call = {
      sizeof(call), capability, &request,
      input_count, sizeof(*inputs), input_count * sizeof(*inputs), inputs,
      output_count, sizeof(*outputs), output_count * sizeof(*outputs), outputs};
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) == 0);
}

static void test_embedding_and_gradient(et_kernel_runtime *runtime) {
  const uint64_t request_shape[] = {1, 4, 3, 2};
  const uint64_t ids_shape[] = {1, 4};
  const uint64_t weight_shape[] = {3, 2};
  const uint64_t output_shape[] = {1, 4, 2};
  const int64_t ids_values[] = {2, 0, 2, 1};
  const float weight_values[] = {1, 2, 3, 4, 5, 6};
  const float zero_weight[] = {0, 0, 0, 0, 0, 0};
  const float zero_output[] = {0, 0, 0, 0, 0, 0, 0, 0};
  const float upstream_values[] = {1, 2, 3, 4, 5, 6, 7, 8};
  const float expected_output[] = {5, 6, 1, 2, 5, 6, 3, 4};
  const float expected_gradient[] = {3, 4, 7, 8, 6, 8};
  et_i64_tensor *ids = make_i64(2u, ids_shape, ids_values, 4u);
  et_f32_tensor *weight = make_f32(2u, weight_shape, weight_values, 6u);
  et_f32_tensor *output = make_f32(3u, output_shape, zero_output, 8u);
  et_f32_tensor *upstream = make_f32(3u, output_shape, upstream_values, 8u);
  et_f32_tensor *dweight = make_f32(2u, weight_shape, zero_weight, 6u);
  i64_lease ids_lease = borrow_i64(ids);
  f32_lease weight_lease = borrow_f32(weight);
  f32_lease output_lease = borrow_f32(output);
  et_kernel_tensor_view_v1 inputs[] = {*ids_lease.view, *weight_lease.view};
  et_kernel_tensor_view_v1 outputs[] = {*output_lease.view};
  et_f32_tensor_error f32_error;
  et_i64_tensor_error i64_error;
  uint32_t replacement[6] = {0};
  int64_t replacement_ids[4] = {0};

  CHECK(et_f32_tensor_copy_bits_from_v1(weight, replacement, 6u, &f32_error) ==
        ET_F32_TENSOR_ERROR_INVALID_STATE);
  CHECK(f32_error.code == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(et_i64_tensor_copy_from_v1(ids, replacement_ids, 4u, &i64_error) ==
        ET_I64_TENSOR_ERROR_INVALID_STATE);
  CHECK(i64_error.code == ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  dispatch(runtime, "kernel.embedding-forward", "embedding.forward", 4u,
           request_shape, inputs, 2u, outputs, 1u);
  end_f32(&output_lease);
  end_f32(&weight_lease);
  end_i64(&ids_lease);
  expect_f32(output, expected_output, 8u);

  ids_lease = borrow_i64(ids);
  f32_lease upstream_lease = borrow_f32(upstream);
  f32_lease dweight_lease = borrow_f32(dweight);
  inputs[0] = *ids_lease.view;
  inputs[1] = *upstream_lease.view;
  outputs[0] = *dweight_lease.view;
  dispatch(runtime, "kernel.embedding-backward", "embedding.backward", 4u,
           request_shape, inputs, 2u, outputs, 1u);
  end_f32(&dweight_lease);
  end_f32(&upstream_lease);
  end_i64(&ids_lease);
  expect_f32(dweight, expected_gradient, 6u);

  {
    et_f32_parameter *parameter = NULL;
    et_f32_gradient_plan *plan = NULL;
    et_f32_tensor *snapshot = NULL;
    int identity = 0;
    et_f32_gradient_metadata_v1 metadata = {sizeof(metadata), 0u, 0u, 0u};
    et_f32_gradient_contribution_v1 contribution = {
        sizeof(contribution), NULL, dweight, 0u};
    CHECK(et_f32_parameter_create_v1(weight, &parameter, &f32_error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &f32_error) ==
          0);
    contribution.destination = parameter;
    CHECK(et_f32_gradient_plan_prepare_v1(1u, &contribution,
                                          UINT32_C(0x3f800000), &plan,
                                          &f32_error) == 0);
    CHECK(et_f32_tensor_destroy_v1(&dweight, &f32_error) ==
          ET_F32_TENSOR_ERROR_INVALID_STATE);
    CHECK(dweight != NULL);
    CHECK(et_f32_gradient_plan_commit_v1(plan, &f32_error) == 0);
    CHECK(et_f32_gradient_plan_release_v1(&plan, &f32_error) == 0);
    CHECK(et_f32_parameter_gradient_metadata_v1(parameter, &metadata,
                                                &f32_error) == 0);
    CHECK(metadata.state == ET_F32_GRADIENT_PRESENT);
    CHECK(metadata.contribution_count == 1u);
    CHECK(metadata.normalization_weight_bits == UINT32_C(0x3f800000));
    CHECK(et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot,
                                                &f32_error) == 0);
    expect_f32(snapshot, expected_gradient, 6u);
    destroy_f32(&snapshot);
    CHECK(et_f32_parameter_destroy_v1(&parameter, &f32_error) == 0);
  }

  destroy_f32(&dweight);
  destroy_f32(&upstream);
  destroy_f32(&output);
  destroy_f32(&weight);
  destroy_i64(&ids);
}

static void test_residual_unique_handle_accumulation(
    et_kernel_runtime *runtime) {
  const uint64_t shape[] = {1, 3, 4};
  const float upstream_values[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  const float zeros[12] = {0};
  const float combined_values[] = {2, 4, 6, 8, 10, 12, 14, 16,
                                   18, 20, 22, 24};
  et_f32_tensor *upstream = make_f32(3u, shape, upstream_values, 12u);
  et_f32_tensor *edge_x = make_f32(3u, shape, zeros, 12u);
  et_f32_tensor *edge_branch = make_f32(3u, shape, zeros, 12u);
  et_f32_tensor *combined = make_f32(3u, shape, zeros, 12u);
  et_f32_tensor *initial = make_f32(3u, shape, zeros, 12u);
  f32_lease upstream_lease = borrow_f32(upstream);
  f32_lease edge_x_lease = borrow_f32(edge_x);
  f32_lease edge_branch_lease = borrow_f32(edge_branch);
  et_kernel_tensor_view_v1 inputs[2] = {*upstream_lease.view};
  et_kernel_tensor_view_v1 outputs[2] = {*edge_x_lease.view,
                                         *edge_branch_lease.view};
  et_f32_tensor_error error;

  dispatch(runtime, "kernel.residual", "residual.backward", 3u, shape, inputs,
           1u, outputs, 2u);
  end_f32(&edge_branch_lease);
  end_f32(&edge_x_lease);
  end_f32(&upstream_lease);

  edge_x_lease = borrow_f32(edge_x);
  edge_branch_lease = borrow_f32(edge_branch);
  f32_lease combined_lease = borrow_f32(combined);
  inputs[0] = *edge_x_lease.view;
  inputs[1] = *edge_branch_lease.view;
  outputs[0] = *combined_lease.view;
  dispatch(runtime, "kernel.residual", "residual.forward", 3u, shape, inputs,
           2u, outputs, 1u);
  end_f32(&combined_lease);
  end_f32(&edge_branch_lease);
  end_f32(&edge_x_lease);
  expect_f32(combined, combined_values, 12u);

  {
    et_f32_parameter *parameter = NULL;
    et_f32_gradient_plan *plan = NULL;
    et_f32_gradient_contribution_v1 contributions[2];
    int identity = 0;
    CHECK(et_f32_parameter_create_v1(initial, &parameter, &error) == 0);
    CHECK(et_f32_parameter_bind_identity_v1(parameter, &identity, &error) == 0);
    for (size_t index = 0; index < 2u; index++) {
      contributions[index].struct_size = sizeof(contributions[index]);
      contributions[index].destination = parameter;
      contributions[index].weighted_numerator = combined;
      contributions[index].expected_ordinal = 0u;
    }
    CHECK(et_f32_gradient_plan_prepare_v1(2u, contributions,
                                          UINT32_C(0x3f800000), &plan,
                                          &error) ==
          ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
    CHECK(plan == NULL);
    CHECK(et_f32_gradient_plan_prepare_v1(1u, contributions,
                                          UINT32_C(0x3f800000), &plan,
                                          &error) == 0);
    CHECK(et_f32_gradient_plan_commit_v1(plan, &error) == 0);
    CHECK(et_f32_gradient_plan_release_v1(&plan, &error) == 0);
    {
      et_f32_tensor *snapshot = NULL;
      CHECK(et_f32_parameter_gradient_snapshot_v1(parameter, &snapshot,
                                                  &error) == 0);
      expect_f32(snapshot, combined_values, 12u);
      destroy_f32(&snapshot);
    }
    CHECK(et_f32_parameter_destroy_v1(&parameter, &error) == 0);
  }

  destroy_f32(&initial);
  destroy_f32(&combined);
  destroy_f32(&edge_branch);
  destroy_f32(&edge_x);
  destroy_f32(&upstream);
}

static void test_dropout_i1_state(et_kernel_runtime *runtime) {
  const uint64_t shape[] = {1, 1, 4};
  const uint64_t state_shape[] = {4};
  const float inputs_values[] = {1, 2, 3, 4};
  const float zero_values[] = {0, 0, 0, 0};
  const float probability_value[] = {0.25f};
  const int64_t state_values[] = {1, 0, 0, 0};
  const int64_t expected_next[] = {1, 0, 1, 0};
  et_f32_tensor *input = make_f32(3u, shape, inputs_values, 4u);
  et_f32_tensor *probability = make_f32(0u, NULL, probability_value, 1u);
  et_f32_tensor *output = make_f32(3u, shape, zero_values, 4u);
  et_i64_tensor *state = make_i64(1u, state_shape, state_values, 4u);
  et_i64_tensor *next = make_i64(1u, state_shape, state_values, 4u);
  f32_lease input_lease = borrow_f32(input);
  f32_lease probability_lease = borrow_f32(probability);
  f32_lease output_lease = borrow_f32(output);
  i64_lease state_lease = borrow_i64(state);
  i64_lease next_lease = borrow_i64(next);
  et_kernel_tensor_view_v1 inputs[] = {*input_lease.view,
                                       *probability_lease.view,
                                       *state_lease.view};
  et_kernel_tensor_view_v1 outputs[] = {*output_lease.view, *next_lease.view};
  int64_t actual_next[4] = {0};
  et_i64_tensor_error error;

  dispatch(runtime, "kernel.dropout", "dropout.train.forward", 3u, shape,
           inputs, 3u, outputs, 2u);
  end_i64(&next_lease);
  end_i64(&state_lease);
  end_f32(&output_lease);
  end_f32(&probability_lease);
  end_f32(&input_lease);
  CHECK(et_i64_tensor_copy_to_v1(next, actual_next, 4u, &error) == 0);
  CHECK(memcmp(actual_next, expected_next, sizeof(actual_next)) == 0);

  destroy_i64(&next);
  destroy_i64(&state);
  destroy_f32(&output);
  destroy_f32(&probability);
  destroy_f32(&input);
}

static void test_remaining_operations_use_i2_views(
    et_kernel_runtime *runtime) {
  const uint64_t shape3[] = {1, 1, 1};
  const uint64_t shape4[] = {1, 1, 1, 1};
  const uint64_t shape2[] = {1, 1};
  const uint64_t scalar_shape[] = {1};
  const uint64_t state_shape[] = {4};
  const float x_value[] = {2.0f};
  const float weight_value[] = {3.0f};
  const float bias_value[] = {1.0f};
  const float gamma_value[] = {2.0f};
  const float beta_value[] = {-1.0f};
  const float upstream_value[] = {4.0f};
  const float epsilon_value[] = {1.0e-5f};
  const float probability_value[] = {0.0f};
  const float zero_value[] = {0.0f};
  const int64_t state_value[] = {1, 0, 0, 0};
  et_f32_tensor *x = make_f32(3u, shape3, x_value, 1u);
  et_f32_tensor *weight = make_f32(2u, shape2, weight_value, 1u);
  et_f32_tensor *bias = make_f32(1u, scalar_shape, bias_value, 1u);
  et_f32_tensor *gamma = make_f32(1u, scalar_shape, gamma_value, 1u);
  et_f32_tensor *beta = make_f32(1u, scalar_shape, beta_value, 1u);
  et_f32_tensor *upstream =
      make_f32(3u, shape3, upstream_value, 1u);
  et_f32_tensor *epsilon = make_f32(0u, NULL, epsilon_value, 1u);
  et_f32_tensor *probability =
      make_f32(0u, NULL, probability_value, 1u);
  et_f32_tensor *y = make_f32(3u, shape3, zero_value, 1u);
  et_f32_tensor *dx = make_f32(3u, shape3, zero_value, 1u);
  et_f32_tensor *dw = make_f32(2u, shape2, zero_value, 1u);
  et_f32_tensor *d0 = make_f32(1u, scalar_shape, zero_value, 1u);
  et_f32_tensor *d1 = make_f32(1u, scalar_shape, zero_value, 1u);
  et_i64_tensor *state = make_i64(1u, state_shape, state_value, 4u);
  et_i64_tensor *next = make_i64(1u, state_shape, state_value, 4u);
  f32_lease x_l = borrow_f32(x);
  f32_lease weight_l = borrow_f32(weight);
  f32_lease bias_l = borrow_f32(bias);
  f32_lease gamma_l = borrow_f32(gamma);
  f32_lease beta_l = borrow_f32(beta);
  f32_lease upstream_l = borrow_f32(upstream);
  f32_lease epsilon_l = borrow_f32(epsilon);
  f32_lease probability_l = borrow_f32(probability);
  f32_lease y_l = borrow_f32(y);
  f32_lease dx_l = borrow_f32(dx);
  f32_lease dw_l = borrow_f32(dw);
  f32_lease d0_l = borrow_f32(d0);
  f32_lease d1_l = borrow_f32(d1);
  i64_lease state_l = borrow_i64(state);
  i64_lease next_l = borrow_i64(next);
  et_kernel_tensor_view_v1 inputs[4];
  et_kernel_tensor_view_v1 outputs[3];

  inputs[0] = *x_l.view;
  inputs[1] = *weight_l.view;
  inputs[2] = *bias_l.view;
  outputs[0] = *y_l.view;
  dispatch(runtime, "kernel.matmul", "linear.forward", 4u, shape4, inputs,
           3u, outputs, 1u);
  dispatch(runtime, "kernel.matmul", "linear.forward-no-bias", 4u, shape4,
           inputs, 2u, outputs, 1u);
  inputs[2] = *upstream_l.view;
  outputs[0] = *dx_l.view;
  outputs[1] = *dw_l.view;
  outputs[2] = *d0_l.view;
  dispatch(runtime, "kernel.matmul", "linear.backward", 4u, shape4, inputs,
           3u, outputs, 3u);
  dispatch(runtime, "kernel.matmul", "linear.backward-no-bias", 4u, shape4,
           inputs, 3u, outputs, 2u);

  inputs[0] = *x_l.view;
  inputs[1] = *gamma_l.view;
  inputs[2] = *beta_l.view;
  inputs[3] = *epsilon_l.view;
  outputs[0] = *y_l.view;
  dispatch(runtime, "kernel.norm", "layer-norm.forward", 3u, shape3, inputs,
           4u, outputs, 1u);
  inputs[2] = *epsilon_l.view;
  inputs[3] = *upstream_l.view;
  outputs[0] = *dx_l.view;
  outputs[1] = *d0_l.view;
  outputs[2] = *d1_l.view;
  dispatch(runtime, "kernel.norm", "layer-norm.backward", 3u, shape3, inputs,
           4u, outputs, 3u);

  inputs[0] = *x_l.view;
  outputs[0] = *y_l.view;
  dispatch(runtime, "kernel.activation", "gelu.forward", 3u, shape3, inputs,
           1u, outputs, 1u);
  dispatch(runtime, "kernel.activation", "relu.forward", 3u, shape3, inputs,
           1u, outputs, 1u);
  inputs[1] = *upstream_l.view;
  outputs[0] = *dx_l.view;
  dispatch(runtime, "kernel.activation", "gelu.backward", 3u, shape3, inputs,
           2u, outputs, 1u);
  dispatch(runtime, "kernel.activation", "relu.backward", 3u, shape3, inputs,
           2u, outputs, 1u);

  inputs[0] = *x_l.view;
  outputs[0] = *y_l.view;
  dispatch(runtime, "kernel.dropout", "dropout.eval.forward", 3u, shape3,
           inputs, 1u, outputs, 1u);
  inputs[0] = *upstream_l.view;
  outputs[0] = *dx_l.view;
  dispatch(runtime, "kernel.dropout", "dropout.eval.backward", 3u, shape3,
           inputs, 1u, outputs, 1u);
  inputs[1] = *probability_l.view;
  inputs[2] = *state_l.view;
  dispatch(runtime, "kernel.dropout", "dropout.train.backward", 3u, shape3,
           inputs, 3u, outputs, 1u);
  inputs[0] = *x_l.view;
  outputs[0] = *y_l.view;
  outputs[1] = *next_l.view;
  dispatch(runtime, "kernel.dropout", "dropout.train.forward", 3u, shape3,
           inputs, 3u, outputs, 2u);

  end_i64(&next_l);
  end_i64(&state_l);
  end_f32(&d1_l);
  end_f32(&d0_l);
  end_f32(&dw_l);
  end_f32(&dx_l);
  end_f32(&y_l);
  end_f32(&probability_l);
  end_f32(&epsilon_l);
  end_f32(&upstream_l);
  end_f32(&beta_l);
  end_f32(&gamma_l);
  end_f32(&bias_l);
  end_f32(&weight_l);
  end_f32(&x_l);
  expect_f32(y, x_value, 1u);
  expect_f32(dx, upstream_value, 1u);

  destroy_i64(&next);
  destroy_i64(&state);
  destroy_f32(&d1);
  destroy_f32(&d0);
  destroy_f32(&dw);
  destroy_f32(&dx);
  destroy_f32(&y);
  destroy_f32(&probability);
  destroy_f32(&epsilon);
  destroy_f32(&upstream);
  destroy_f32(&beta);
  destroy_f32(&gamma);
  destroy_f32(&bias);
  destroy_f32(&weight);
  destroy_f32(&x);
}

int main(void) {
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_v1 before;
  et_f32_test_live_counts_v1 after;
  memset(&before, 0, sizeof(before));
  memset(&after, 0, sizeof(after));
  before.struct_size = sizeof(before);
  after.struct_size = sizeof(after);
  et_f32_test_live_counts_snapshot_v1(&before);
#endif
  et_kernel_runtime *runtime = make_runtime();
  test_embedding_and_gradient(runtime);
  test_residual_unique_handle_accumulation(runtime);
  test_dropout_i1_state(runtime);
  test_remaining_operations_use_i2_views(runtime);
  et_kernel_runtime_destroy(runtime);
#ifdef ET_F32_TENSOR_TESTING
  et_f32_test_live_counts_snapshot_v1(&after);
  if (memcmp(&before, &after, sizeof(before)) != 0) {
    (void)fprintf(stderr, "FAIL: I2 live allocation counts did not return to baseline\n");
    return 1;
  }
#endif
  (void)printf("N2 I1/I2 integration PASS (%zu checks)\n", checks);
  return 0;
}
