#include "eshkol_transformer/indexed_cross_entropy.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
static int failures;

#define CHECK(condition)                                                     \
  do {                                                                       \
    checks++;                                                                \
    if (!(condition)) {                                                      \
      failures++;                                                            \
      fprintf(stderr, "L2 check failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                                   \
    }                                                                        \
  } while (0)

static const et_kernel_provider_v1 *resolve_l2(void *context,
                                                const char *symbol) {
  (void)context;
  return strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0
             ? et_l2_indexed_cross_entropy_provider_v1()
             : NULL;
}

static et_kernel_tensor_view_v1 view(void *data, size_t bytes,
                                     const char *dtype, const char *device,
                                     size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = device,
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = rank,
      .shape = shape,
  };
  return result;
}

static int close_float(float actual, float expected, float absolute,
                       float relative) {
  return fabsf(actual - expected) <= absolute + relative * fabsf(expected);
}

typedef struct fixture {
  uint64_t logits_shape[3];
  uint64_t leading_shape[2];
  float logits[6];
  int64_t targets[2];
  float upstream[2];
  float losses[2];
  float gradients[6];
  et_kernel_tensor_view_v1 forward_inputs[2];
  et_kernel_tensor_view_v1 backward_inputs[3];
  et_kernel_tensor_view_v1 forward_output;
  et_kernel_tensor_view_v1 backward_output;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
} fixture;

static void fixture_init(fixture *item) {
  const float logits[] = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  memset(item, 0, sizeof(*item));
  item->logits_shape[0] = 1u;
  item->logits_shape[1] = 2u;
  item->logits_shape[2] = 3u;
  item->leading_shape[0] = 1u;
  item->leading_shape[1] = 2u;
  memcpy(item->logits, logits, sizeof(logits));
  item->targets[0] = 2;
  item->targets[1] = 0;
  item->upstream[0] = 0.7f;
  item->upstream[1] = -1.25f;
  item->forward_inputs[0] =
      view(item->logits, sizeof(item->logits), "f32", "cpu", 3u,
           item->logits_shape);
  item->forward_inputs[1] =
      view(item->targets, sizeof(item->targets), "i64", "cpu", 2u,
           item->leading_shape);
  item->backward_inputs[0] = item->forward_inputs[0];
  item->backward_inputs[1] = item->forward_inputs[1];
  item->backward_inputs[2] =
      view(item->upstream, sizeof(item->upstream), "f32", "cpu", 2u,
           item->leading_shape);
  item->forward_output =
      view(item->losses, sizeof(item->losses), "f32", "cpu", 2u,
           item->leading_shape);
  item->backward_output =
      view(item->gradients, sizeof(item->gradients), "f32", "cpu", 3u,
           item->logits_shape);
  item->request.struct_size = sizeof(item->request);
  item->request.operation = ET_L2_INDEXED_CROSS_ENTROPY_FORWARD;
  item->request.dtype = "f32";
  item->request.device = "cpu";
  item->request.rank = 3u;
  item->request.shape = item->logits_shape;
  item->request.deterministic = 1u;
  item->call.struct_size = sizeof(item->call);
  item->call.capability = ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY;
  item->call.request = &item->request;
  item->call.input_count = 2u;
  item->call.input_stride = sizeof(et_kernel_tensor_view_v1);
  item->call.input_bytes = sizeof(item->forward_inputs);
  item->call.inputs = item->forward_inputs;
  item->call.output_count = 1u;
  item->call.output_stride = sizeof(et_kernel_tensor_view_v1);
  item->call.output_bytes = sizeof(item->forward_output);
  item->call.outputs = &item->forward_output;
}

static int32_t run_forward(et_kernel_runtime *runtime, fixture *item,
                           et_kernel_error *error) {
  item->request.operation = ET_L2_INDEXED_CROSS_ENTROPY_FORWARD;
  item->call.input_count = 2u;
  item->call.input_bytes = sizeof(item->forward_inputs);
  item->call.inputs = item->forward_inputs;
  item->call.output_bytes = sizeof(item->forward_output);
  item->call.outputs = &item->forward_output;
  return et_kernel_runtime_dispatch(runtime, &item->call, error);
}

static int32_t run_backward(et_kernel_runtime *runtime, fixture *item,
                            et_kernel_error *error) {
  item->request.operation = ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD;
  item->call.input_count = 3u;
  item->call.input_bytes = sizeof(item->backward_inputs);
  item->call.inputs = item->backward_inputs;
  item->call.output_bytes = sizeof(item->backward_output);
  item->call.outputs = &item->backward_output;
  return et_kernel_runtime_dispatch(runtime, &item->call, error);
}

static void expect_error_unchanged(int32_t status,
                                   const et_kernel_error *error,
                                   et_kernel_error_category category,
                                   const unsigned char *before,
                                   const void *after, size_t bytes) {
  CHECK(status == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code != ET_KERNEL_CODE_OK);
  CHECK(memcmp(before, after, bytes) == 0);
}

static void test_discovery_and_report(void) {
  et_kernel_runtime *baseline = NULL;
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  const et_kernel_capability_v1 *entry;
  size_t required = 0u;
  char *report;
  CHECK(et_l2_indexed_cross_entropy_abi_major_v1() == 1);
  CHECK(et_l2_indexed_cross_entropy_abi_minor_v1() == 0);
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);
  entry = et_kernel_runtime_capability_find(
      baseline, ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY);
  CHECK(entry != NULL && entry->status == ET_KERNEL_CAPABILITY_UNVERIFIED);
  et_kernel_runtime_destroy(baseline);
  CHECK(et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) == 0);
  entry = et_kernel_runtime_capability_find(
      runtime, ET_L2_INDEXED_CROSS_ENTROPY_CAPABILITY);
  CHECK(entry != NULL && entry->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(entry->operation_count == 2u);
  CHECK(strcmp(entry->operations[0],
               ET_L2_INDEXED_CROSS_ENTROPY_BACKWARD) == 0);
  CHECK(strcmp(entry->operations[1],
               ET_L2_INDEXED_CROSS_ENTROPY_FORWARD) == 0);
  CHECK(entry->dtype_count == 1u && strcmp(entry->dtypes[0], "f32") == 0);
  CHECK(entry->device_count == 1u && strcmp(entry->devices[0], "cpu") == 0);
  CHECK(entry->deterministic == 1u && entry->shape_range_count == 1u);
  CHECK(entry->shape_ranges[0].rank == 3u);
  CHECK(entry->shape_ranges[0].dimensions[0].maximum ==
        ET_L2_INDEXED_CROSS_ENTROPY_MAX_EXTENT);
  CHECK(et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) ==
        0);
  report = (char *)malloc(required);
  CHECK(report != NULL);
  if (report != NULL) {
    CHECK(et_kernel_runtime_report_json(runtime, report, required, &required,
                                        &error) == 0);
    CHECK(strstr(report, "\"status\":\"verified\"") != NULL);
    CHECK(strstr(report, "L2:cpu-f32-indexed-cross-entropy-v1") != NULL);
    CHECK(strstr(report, "\"provider_abi\":{\"major\":1,\"minor\":0}") !=
          NULL);
    free(report);
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_forward_backward_and_determinism(void) {
  static const float expected_loss[] = {0.4076059461f, 2.4076058865f};
  static const float expected_gradient[] = {
      0.0630213991f, 0.1713099331f, -0.2343313396f,
      1.1374617815f, -0.3059105873f, -0.8315511942f,
  };
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  fixture item;
  float first_losses[2];
  float first_gradients[6];
  fixture_init(&item);
  CHECK(et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) == 0);
  memset(item.losses, 0xa5, sizeof(item.losses));
  CHECK(run_forward(runtime, &item, &error) == 0);
  for (size_t index = 0u; index < 2u; index++) {
    CHECK(close_float(item.losses[index], expected_loss[index], 2.0e-6f,
                      2.0e-5f));
  }
  memset(item.gradients, 0xa5, sizeof(item.gradients));
  CHECK(run_backward(runtime, &item, &error) == 0);
  for (size_t index = 0u; index < 6u; index++) {
    CHECK(close_float(item.gradients[index], expected_gradient[index], 3.0e-6f,
                      3.0e-5f));
  }
  CHECK(close_float(item.gradients[0] + item.gradients[1] + item.gradients[2],
                    0.0f, 2.0e-6f, 0.0f));
  CHECK(close_float(item.gradients[3] + item.gradients[4] + item.gradients[5],
                    0.0f, 2.0e-6f, 0.0f));
  memcpy(first_losses, item.losses, sizeof(first_losses));
  memcpy(first_gradients, item.gradients, sizeof(first_gradients));
  for (size_t repeat = 0u; repeat < 1000u; repeat++) {
    CHECK(run_forward(runtime, &item, &error) == 0);
    CHECK(run_backward(runtime, &item, &error) == 0);
    CHECK(memcmp(first_losses, item.losses, sizeof(first_losses)) == 0);
    CHECK(memcmp(first_gradients, item.gradients, sizeof(first_gradients)) == 0);
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_extremes_and_vocabulary_one(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  fixture item;
  CHECK(et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) == 0);
  fixture_init(&item);
  item.logits_shape[1] = 1u;
  item.leading_shape[1] = 1u;
  item.forward_inputs[0].byte_length = 3u * sizeof(float);
  item.forward_inputs[1].byte_length = sizeof(int64_t);
  item.backward_inputs[0].byte_length = 3u * sizeof(float);
  item.backward_inputs[1].byte_length = sizeof(int64_t);
  item.backward_inputs[2].byte_length = sizeof(float);
  item.forward_output.byte_length = sizeof(float);
  item.backward_output.byte_length = 3u * sizeof(float);
  item.logits[0] = FLT_MAX;
  item.logits[1] = FLT_MAX;
  item.logits[2] = -FLT_MAX;
  item.targets[0] = 1;
  item.upstream[0] = 1.0f;
  CHECK(run_forward(runtime, &item, &error) == 0);
  CHECK(close_float(item.losses[0], logf(2.0f), 2.0e-6f, 2.0e-5f));
  CHECK(run_backward(runtime, &item, &error) == 0);
  CHECK(item.gradients[0] == 0.5f && item.gradients[1] == -0.5f &&
        item.gradients[2] == 0.0f);
  item.logits[0] = 1000.0f;
  item.logits[1] = -1000.0f;
  item.logits[2] = 0.0f;
  item.targets[0] = 1;
  CHECK(run_forward(runtime, &item, &error) == 0);
  CHECK(item.losses[0] == 2000.0f);
  CHECK(run_backward(runtime, &item, &error) == 0);
  CHECK(item.gradients[0] == 1.0f && item.gradients[1] == -1.0f &&
        item.gradients[2] == 0.0f);

  item.logits_shape[2] = 1u;
  item.forward_inputs[0].byte_length = sizeof(float);
  item.backward_inputs[0].byte_length = sizeof(float);
  item.backward_output.byte_length = sizeof(float);
  item.logits[0] = -FLT_MAX;
  item.targets[0] = 0;
  item.upstream[0] = -3.0f;
  CHECK(run_forward(runtime, &item, &error) == 0 && item.losses[0] == 0.0f);
  CHECK(run_backward(runtime, &item, &error) == 0 &&
        item.gradients[0] == 0.0f);
  et_kernel_runtime_destroy(runtime);
}

static float finite_difference_objective(et_kernel_runtime *runtime,
                                         fixture *item) {
  et_kernel_error error;
  if (run_forward(runtime, item, &error) != 0) {
    return NAN;
  }
  return item->losses[0] * item->upstream[0] +
         item->losses[1] * item->upstream[1];
}

static void test_native_finite_difference(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  fixture item;
  float analytic[6];
  fixture_init(&item);
  CHECK(et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) == 0);
  CHECK(run_backward(runtime, &item, &error) == 0);
  memcpy(analytic, item.gradients, sizeof(analytic));
  for (size_t index = 0u; index < 6u; index++) {
    const float original = item.logits[index];
    const float step = 0.0009765625f * fmaxf(1.0f, fabsf(original));
    float high;
    float low;
    item.logits[index] = original + step;
    high = finite_difference_objective(runtime, &item);
    item.logits[index] = original - step;
    low = finite_difference_objective(runtime, &item);
    item.logits[index] = original;
    CHECK(close_float((high - low) / (2.0f * step), analytic[index], 2.0e-3f,
                      5.0e-3f));
  }
  et_kernel_runtime_destroy(runtime);
}

static void test_failure_atomicity(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  fixture item;
  uint64_t alternate_shape[2] = {1u, 1u};
  _Alignas(8) unsigned char overlapping_inputs[32];
  unsigned char misaligned_targets[sizeof(item.targets) + 1u];
  unsigned char before[sizeof(item.gradients)];
  int32_t status;
  fixture_init(&item);
  CHECK(et_kernel_runtime_discover(resolve_l2, NULL, &runtime, &error) == 0);

#define POISON_FORWARD()                                                     \
  do {                                                                       \
    memset(item.losses, 0xa5, sizeof(item.losses));                          \
    memcpy(before, item.losses, sizeof(item.losses));                        \
  } while (0)
#define POISON_BACKWARD()                                                    \
  do {                                                                       \
    memset(item.gradients, 0x5a, sizeof(item.gradients));                    \
    memcpy(before, item.gradients, sizeof(item.gradients));                  \
  } while (0)

  for (size_t index = 0u; index < 4u; index++) {
    static const int64_t invalid[] = {INT64_MIN, -1, 3, INT64_MAX};
    item.targets[1] = invalid[index];
    POISON_FORWARD();
    status = run_forward(runtime, &item, &error);
    expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                           before, item.losses, sizeof(item.losses));
  }
  item.targets[1] = 0;
  item.logits[5] = NAN;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(item.losses));
  item.logits[5] = 1.0f;
  item.upstream[1] = INFINITY;
  POISON_BACKWARD();
  status = run_backward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.gradients, sizeof(item.gradients));
  item.upstream[1] = -1.25f;

  item.logits_shape[1] = 1u;
  item.leading_shape[1] = 1u;
  item.forward_inputs[0].byte_length = 3u * sizeof(float);
  item.forward_inputs[1].byte_length = sizeof(int64_t);
  item.forward_output.byte_length = sizeof(float);
  item.logits[0] = FLT_MAX;
  item.logits[1] = -FLT_MAX;
  item.logits[2] = 0.0f;
  item.targets[0] = 1;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(float));

  fixture_init(&item);
  item.forward_inputs[1].data = item.logits;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_output.data = item.logits;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[0].layout = (et_kernel_layout_v1)0u;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_NONCONTIGUOUS,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[1].dtype = "f32";
  item.forward_inputs[1].byte_length = 2u * sizeof(float);
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_DTYPE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[1].rank = 1u;
  item.forward_inputs[1].byte_length = sizeof(int64_t);
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[1].shape = alternate_shape;
  item.forward_inputs[1].byte_length = sizeof(int64_t);
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_output.rank = 1u;
  item.forward_output.shape = &item.leading_shape[1];
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.backward_inputs[2].shape = alternate_shape;
  item.backward_inputs[2].byte_length = sizeof(float);
  POISON_BACKWARD();
  status = run_backward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         before, item.gradients, sizeof(item.gradients));
  fixture_init(&item);
  memcpy(misaligned_targets + 1u, item.targets, sizeof(item.targets));
  item.forward_inputs[1].data = misaligned_targets + 1u;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  memset(overlapping_inputs, 0, sizeof(overlapping_inputs));
  item.forward_inputs[0].data = overlapping_inputs;
  item.forward_inputs[1].data = overlapping_inputs + 8u;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[0].byte_length--;
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.forward_inputs[0].device = "gpu:0";
  POISON_FORWARD();
  status = run_forward(runtime, &item, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_DEVICE_MISMATCH,
                         before, item.losses, sizeof(item.losses));
  fixture_init(&item);
  item.logits_shape[0] = 0u;
  item.leading_shape[0] = 0u;
  item.forward_inputs[0].byte_length = 0u;
  item.forward_inputs[0].data = NULL;
  item.forward_inputs[1].byte_length = 0u;
  item.forward_inputs[1].data = NULL;
  item.forward_output.byte_length = 0u;
  item.forward_output.data = NULL;
  status = run_forward(runtime, &item, &error);
  CHECK(status == ET_KERNEL_ERROR_UNSUPPORTED);
  fixture_init(&item);
  item.logits_shape[1] = 0u;
  item.leading_shape[1] = 0u;
  item.forward_inputs[0].byte_length = 0u;
  item.forward_inputs[0].data = NULL;
  item.forward_inputs[1].byte_length = 0u;
  item.forward_inputs[1].data = NULL;
  item.forward_output.byte_length = 0u;
  item.forward_output.data = NULL;
  status = run_forward(runtime, &item, &error);
  CHECK(status == ET_KERNEL_ERROR_UNSUPPORTED);
  fixture_init(&item);
  item.logits_shape[2] = 0u;
  item.forward_inputs[0].byte_length = 0u;
  item.forward_inputs[0].data = NULL;
  status = run_forward(runtime, &item, &error);
  CHECK(status == ET_KERNEL_ERROR_UNSUPPORTED);
  fixture_init(&item);
  POISON_FORWARD();
  item.request.operation = "indexed-cross-entropy.unknown";
  item.call.input_count = 2u;
  item.call.input_bytes = sizeof(item.forward_inputs);
  item.call.inputs = item.forward_inputs;
  item.call.output_bytes = sizeof(item.forward_output);
  item.call.outputs = &item.forward_output;
  status = et_kernel_runtime_dispatch(runtime, &item.call, &error);
  expect_error_unchanged(status, &error, ET_KERNEL_ERROR_UNSUPPORTED,
                         before, item.losses, sizeof(item.losses));

#undef POISON_FORWARD
#undef POISON_BACKWARD
  et_kernel_runtime_destroy(runtime);
}

int main(void) {
  test_discovery_and_report();
  test_forward_backward_and_determinism();
  test_extremes_and_vocabulary_one();
  test_native_finite_difference();
  test_failure_atomicity();
  if (failures != 0) {
    fprintf(stderr, "L2 FAIL: %d of %d checks failed\n", failures, checks);
    return EXIT_FAILURE;
  }
  printf("L2 PASS: %d capability, numerical, gradient, and adversarial checks\n",
         checks);
  return EXIT_SUCCESS;
}
