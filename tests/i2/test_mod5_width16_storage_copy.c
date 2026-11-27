#include "f32_parameter_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

enum { LEGACY_MATRIX_COUNT = 5, MATRIX_COUNT = 9, BUFFER_WORDS = 1026 };
static const uint64_t matrices[MATRIX_COUNT][2] = {
    {2u, 4u},  {4u, 4u},   {4u, 8u},   {8u, 4u},   {256u, 4u},
    {8u, 16u}, {16u, 16u}, {16u, 32u}, {32u, 16u},
};
static const size_t sorted_matrices[MATRIX_COUNT] = {0u, 1u, 2u, 3u, 5u,
                                                     6u, 7u, 8u, 4u};
static const char legacy_context;
static const char width16_context;

static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  if (symbol == NULL || strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0) {
    return NULL;
  }
  if (context == &legacy_context) {
    return et_f32_tensor_provider_v1();
  }
  if (context == &width16_context) {
    return et_i2_private_mod5_width16_storage_provider_v1();
  }
  return NULL;
}

static void expect_range(const et_kernel_dimension_range_v1 *dimension,
                         uint64_t minimum, uint64_t maximum) {
  const uint8_t zero[7] = {0};
  CHECK(dimension->minimum == minimum && dimension->maximum == maximum);
  CHECK(dimension->maximum_unbounded == 0u);
  CHECK(memcmp(dimension->reserved, zero, sizeof(zero)) == 0);
}

static void expect_row(const et_kernel_capability_v1 *row, const char *version,
                       const char *evidence, size_t matrix_count,
                       const size_t *order) {
  const uint8_t zero[7] = {0};
  CHECK(row != NULL && row->struct_size == ET_KERNEL_CAPABILITY_V1_0_SIZE);
  CHECK(strcmp(row->name, "tensor.f32") == 0);
  CHECK(row->status == ET_KERNEL_CAPABILITY_VERIFIED &&
        row->deterministic == 1u);
  CHECK(memcmp(row->reserved, zero, sizeof(zero)) == 0);
  CHECK(strcmp(row->implementation, "eshkol-transformer-f32") == 0);
  CHECK(strcmp(row->version, version) == 0 &&
        strcmp(row->evidence, evidence) == 0);
  CHECK(row->operation_count == 1u &&
        strcmp(row->operations[0], "storage.copy") == 0);
  CHECK(row->dtype_count == 1u && strcmp(row->dtypes[0], "f32") == 0);
  CHECK(row->device_count == 1u && strcmp(row->devices[0], "cpu") == 0);
  CHECK(row->shape_range_count == matrix_count + 2u);
  CHECK(row->shape_ranges[0].rank == 0u &&
        row->shape_ranges[0].dimensions == NULL);
  CHECK(row->shape_ranges[1].rank == 1u);
  expect_range(row->shape_ranges[1].dimensions, 0u, SIZE_MAX / sizeof(float));
  for (size_t i = 0u; i < matrix_count; i++) {
    const et_kernel_shape_range_v1 *shape = &row->shape_ranges[i + 2u];
    const uint64_t *expected = matrices[order == NULL ? i : order[i]];
    CHECK(shape->rank == 2u);
    expect_range(&shape->dimensions[0], expected[0], expected[0]);
    expect_range(&shape->dimensions[1], expected[1], expected[1]);
  }
}

static et_kernel_runtime *discover(const char *context, const char *version,
                                   const char *evidence, size_t matrix_count) {
  const et_kernel_provider_v1 *provider =
      resolve((void *)context, ET_KERNEL_PROVIDER_SYMBOL_V1);
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(provider != NULL &&
        provider->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE);
  CHECK(provider->abi_major == ET_KERNEL_ABI_MAJOR &&
        provider->abi_minor == ET_KERNEL_ABI_MINOR);
  CHECK(provider->required_features == 0u);
  CHECK(strcmp(provider->name, "eshkol-transformer-f32") == 0);
  CHECK(strcmp(provider->version, version) == 0 &&
        strcmp(provider->evidence, evidence) == 0);
  CHECK(provider->capability_count == 1u);
  CHECK(provider->capability_stride == ET_KERNEL_CAPABILITY_V1_0_SIZE);
  CHECK(provider->capability_bytes == ET_KERNEL_CAPABILITY_V1_0_SIZE);
  expect_row(provider->capabilities, version, evidence, matrix_count, NULL);
  CHECK(et_kernel_runtime_discover(resolve, (void *)context, &runtime,
                                   &error) == 0);
  CHECK(et_kernel_runtime_capability_count(runtime) == 11u);
  expect_row(et_kernel_runtime_capability_find(runtime, "tensor.f32"), version,
             evidence, matrix_count,
             matrix_count == MATRIX_COUNT ? sorted_matrices : NULL);
  return runtime;
}

static void copy_case(const et_kernel_runtime *runtime,
                      const et_kernel_provider_v1 *provider, size_t rank,
                      const uint64_t *shape, const uint64_t *output_shape,
                      int32_t expected) {
  static const uint32_t patterns[] = {0x80000000u, 0x7fc01234u, 0x7f800000u,
                                      1u};
  uint32_t input[BUFFER_WORDS];
  uint32_t output[BUFFER_WORDS];
  uint32_t before_input[BUFFER_WORDS];
  uint32_t expected_output[BUFFER_WORDS];
  size_t word_count = 1u;
  for (size_t i = 0u; i < rank; i++) {
    word_count *= (size_t)shape[i];
  }
  CHECK(word_count <= BUFFER_WORDS - 2u);
  const size_t bytes = word_count * sizeof(uint32_t);
  for (size_t i = 0u; i < BUFFER_WORDS; i++) {
    input[i] = patterns[i % 4u] ^ (uint32_t)(i / 8u);
  }
  memset(output, 0xa5, sizeof(output));
  memcpy(before_input, input, sizeof(input));
  memcpy(expected_output, output, sizeof(output));
  const et_kernel_request_v1 request = {
      sizeof(request), "storage.copy", "f32", "cpu", rank, shape, 1u, {0},
  };
  const et_kernel_tensor_view_v1 input_view = {
      sizeof(input_view),
      input + 1u,
      bytes,
      "f32",
      "cpu",
      ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      0u,
      rank,
      shape,
  };
  et_kernel_tensor_view_v1 output_view = input_view;
  output_view.data = output + 1u;
  output_view.shape = output_shape;
  const et_kernel_call_v1 call = {
      .struct_size = sizeof(call),
      .capability = "tensor.f32",
      .request = &request,
      .input_count = 1u,
      .input_stride = sizeof(input_view),
      .input_bytes = sizeof(input_view),
      .inputs = &input_view,
      .output_count = 1u,
      .output_stride = sizeof(output_view),
      .output_bytes = sizeof(output_view),
      .outputs = &output_view,
  };
  const int32_t require_status =
      expected == ET_KERNEL_ERROR_UNSUPPORTED ? expected : ET_KERNEL_ERROR_NONE;
  const et_kernel_capability_v1 *row = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_capability_require(runtime, "tensor.f32", &request,
                                             &row, &error) == require_status);
  CHECK((row != NULL) == (require_status == ET_KERNEL_ERROR_NONE));
  CHECK(provider->validate_call(&call, &error) == expected);
  CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
  if (expected == 0) {
    memcpy(expected_output + 1u, input + 1u, bytes);
  }
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) == expected);
  if (expected == ET_KERNEL_ERROR_UNSUPPORTED) {
    CHECK(error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  }
  CHECK(memcmp(input, before_input, sizeof(input)) == 0);
  CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
}

static void write_report(const et_kernel_runtime *runtime) {
  et_kernel_error error;
  size_t required = 0u;
  CHECK(et_kernel_runtime_report_json(runtime, NULL, 0u, &required, &error) ==
        0);
  CHECK(required > 0u);
  char *report = malloc(required);
  CHECK(report != NULL);
  CHECK(et_kernel_runtime_report_json(runtime, report, required, &required,
                                      &error) == 0);
  CHECK(fwrite(report, 1u, required - 1u, stdout) == required - 1u);
  free(report);
}

int main(void) {
  static const uint64_t legacy_vectors[] = {0u, 16u};
  static const uint64_t rejected[][2] = {
      {0u, 16u},  {1u, 16u},  {8u, 15u},  {8u, 32u},  {16u, 8u},
      {16u, 17u}, {16u, 31u}, {16u, 33u}, {32u, 17u},
  };
  static const uint64_t rank3[] = {8u, 16u, 1u};
  const et_kernel_provider_v1 *legacy = et_f32_tensor_provider_v1();
  const et_kernel_provider_v1 *width16 =
      et_i2_private_mod5_width16_storage_provider_v1();
  CHECK(legacy != width16 && legacy->validate_call != width16->validate_call);
  CHECK(legacy->invoke_call != NULL &&
        legacy->invoke_call == width16->invoke_call);
  CHECK(resolve(NULL, ET_KERNEL_PROVIDER_SYMBOL_V1) == NULL);
  CHECK(resolve((void *)&width16_context, "wrong_symbol") == NULL);
  et_kernel_runtime *legacy_runtime =
      discover(&legacy_context, "1.1", "I2:bounded-exact-f32-storage.copy-v2",
               LEGACY_MATRIX_COUNT);
  et_kernel_runtime *width16_runtime =
      discover(&width16_context, "1.2", "I2:bounded-exact-f32-storage.copy-v3",
               MATRIX_COUNT);
  copy_case(legacy_runtime, legacy, 0u, NULL, NULL, 0);
  copy_case(width16_runtime, width16, 0u, NULL, NULL, 0);
  for (size_t i = 0u; i < 2u; i++) {
    copy_case(legacy_runtime, legacy, 1u, &legacy_vectors[i],
              &legacy_vectors[i], 0);
    copy_case(width16_runtime, width16, 1u, &legacy_vectors[i],
              &legacy_vectors[i], 0);
  }
  for (size_t i = 0u; i < MATRIX_COUNT; i++) {
    copy_case(legacy_runtime, legacy, 2u, matrices[i], matrices[i],
              i < LEGACY_MATRIX_COUNT ? 0 : ET_KERNEL_ERROR_UNSUPPORTED);
    copy_case(width16_runtime, width16, 2u, matrices[i], matrices[i], 0);
  }
  for (size_t i = 0u; i < sizeof(rejected) / sizeof(rejected[0]); i++) {
    copy_case(width16_runtime, width16, 2u, rejected[i], rejected[i],
              ET_KERNEL_ERROR_UNSUPPORTED);
  }
  copy_case(width16_runtime, width16, 3u, rank3, rank3,
            ET_KERNEL_ERROR_UNSUPPORTED);
  /* These two admitted shapes have identical byte lengths and different axes.
   */
  copy_case(width16_runtime, width16, 2u, matrices[7], matrices[8],
            ET_KERNEL_ERROR_SHAPE_MISMATCH);
  write_report(width16_runtime);
  et_kernel_runtime_destroy(width16_runtime);
  et_kernel_runtime_destroy(legacy_runtime);
  return 0;
}
