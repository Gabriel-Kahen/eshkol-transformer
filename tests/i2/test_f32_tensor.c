#include "eshkol_transformer/f32_tensor.h"

#include <float.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
static int failures;

#define CHECK(condition)                                                        \
  do {                                                                          \
    checks++;                                                                    \
    if (!(condition)) {                                                          \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,             \
                    #condition);                                                 \
      failures++;                                                                \
    }                                                                            \
  } while (0)

_Static_assert(ET_F32_TENSOR_ABI_MAJOR == 1u, "I2 ABI major changed");
_Static_assert(ET_F32_TENSOR_ABI_MINOR == 0u, "I2 ABI minor changed");
_Static_assert(sizeof(float) == 4u && sizeof(uint32_t) == 4u,
               "I2 requires four-byte binary32 storage");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128,
               "I2 requires IEEE-754 binary32");
_Static_assert(sizeof(uintptr_t) == 8u && UINTPTR_MAX == UINT64_MAX,
               "I2 probes freeze the x86-64 ABI");
_Static_assert(sizeof(et_f32_tensor_error) == 264u,
               "I2 error layout changed");

static void expect_error(int32_t result, const et_f32_tensor_error *error,
                         et_f32_tensor_error_category category,
                         et_f32_tensor_error_code code) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
  CHECK(memchr(error->operation, '\0', sizeof(error->operation)) != NULL);
  CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
}

static void expect_k1_error(int32_t result, const et_kernel_error *error,
                            et_kernel_error_category category,
                            et_kernel_error_code code) {
  CHECK(result == (int32_t)category);
  CHECK(error->category == category);
  CHECK(error->code == code);
  CHECK(error->operation[0] != '\0');
  CHECK(error->message[0] != '\0');
}

static et_f32_tensor *create_tensor(size_t rank, const uint64_t *shape) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  CHECK(et_f32_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(error.category == ET_F32_TENSOR_ERROR_NONE);
  CHECK(error.code == ET_F32_TENSOR_CODE_OK);
  return tensor;
}

static void destroy_tensor(et_f32_tensor **tensor) {
  et_f32_tensor_error error;
  CHECK(et_f32_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void expect_bits(const et_f32_tensor *tensor,
                        const uint32_t *expected, size_t count) {
  uint32_t actual[128];
  et_f32_tensor_error error;
  CHECK(count <= sizeof(actual) / sizeof(actual[0]));
  if (count > sizeof(actual) / sizeof(actual[0])) {
    return;
  }
  memset(actual, 0xa5, sizeof(actual));
  CHECK(et_f32_tensor_copy_bits_to_v1(tensor, actual, count, &error) == 0);
  CHECK(memcmp(actual, expected, count * sizeof(*expected)) == 0);
}

static void test_version_nulls_and_forgery(void) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  size_t size_output = 0x5a5a5a5au;
  uint64_t extent = UINT64_C(0x7777777777777777);
  int foreign = 0;

  CHECK(et_f32_tensor_abi_major_v1() == 1);
  CHECK(et_f32_tensor_abi_minor_v1() == 0);
  CHECK(strcmp(ET_F32_TENSOR_ERROR_SOURCE_DOMAIN, "f32-tensor") == 0);
  CHECK(et_f32_tensor_abi_require_v1(1u, 0u, &error) == 0);
  expect_error(et_f32_tensor_abi_require_v1(2u, 0u, &error), &error,
               ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
               ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH);
  expect_error(et_f32_tensor_abi_require_v1(1u, 1u, &error), &error,
               ET_F32_TENSOR_ERROR_VERSION_MISMATCH,
               ET_F32_TENSOR_CODE_ABI_VERSION_MISMATCH);
  CHECK(et_f32_tensor_abi_require_v1(2u, 0u, NULL) ==
        ET_F32_TENSOR_ERROR_VERSION_MISMATCH);

  expect_error(et_f32_tensor_create_v1(0u, NULL, NULL, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_f32_tensor_rank_v1(NULL, &size_output, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(size_output == 0x5a5a5a5au);
  expect_error(et_f32_tensor_shape_at_v1(NULL, 0u, &extent, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(extent == UINT64_C(0x7777777777777777));
  expect_error(et_f32_tensor_rank_v1((const et_f32_tensor *)&foreign,
                                     &size_output, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(size_output == 0x5a5a5a5au);

  tensor = create_tensor(0u, NULL);
  {
    et_f32_tensor *saved = tensor;
    expect_error(et_f32_tensor_create_v1(0u, NULL, &tensor, &error), &error,
                 ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_F32_TENSOR_CODE_INVALID_BUFFER);
    CHECK(tensor == saved);
  }
  destroy_tensor(&tensor);
  CHECK(et_f32_tensor_destroy_v1(&tensor, &error) == 0);
  expect_error(et_f32_tensor_destroy_v1(NULL, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_NULL_ARGUMENT);
}

static void test_shapes_storage_and_exact_bits(void) {
  const uint64_t shape[] = {2u, 3u, 4u};
  const size_t expected_strides[] = {48u, 16u, 4u};
  const uint32_t exact[] = {
      UINT32_C(0x00000000), UINT32_C(0x80000000),
      UINT32_C(0x00000001), UINT32_C(0x007fffff),
      UINT32_C(0x00800000), UINT32_C(0x3f800000),
      UINT32_C(0x7f7fffff), UINT32_C(0x7f800000),
      UINT32_C(0xff800000), UINT32_C(0x7fc12345),
      UINT32_C(0x7fa54321), UINT32_C(0xffc00001)};
  const uint64_t exact_shape[] = {3u, 4u};
  const uint64_t zeros[][3] = {{0u, 3u, 4u}, {2u, 0u, 4u},
                               {2u, 3u, 0u}};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = create_tensor(3u, shape);
  size_t value;
  uint64_t extent;

  CHECK(et_f32_tensor_rank_v1(tensor, &value, &error) == 0 && value == 3u);
  for (size_t index = 0u; index < 3u; index++) {
    CHECK(et_f32_tensor_shape_at_v1(tensor, index, &extent, &error) == 0);
    CHECK(extent == shape[index]);
    CHECK(et_f32_tensor_stride_bytes_at_v1(tensor, index, &value, &error) ==
          0);
    CHECK(value == expected_strides[index]);
  }
  CHECK(et_f32_tensor_element_count_v1(tensor, &value, &error) == 0);
  CHECK(value == 24u);
  CHECK(et_f32_tensor_byte_length_v1(tensor, &value, &error) == 0);
  CHECK(value == 96u);
  extent = 99u;
  expect_error(et_f32_tensor_shape_at_v1(tensor, 3u, &extent, &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INVALID_SHAPE);
  CHECK(extent == 99u);
  destroy_tensor(&tensor);

  {
    uint64_t rank64[ET_KERNEL_MAX_RANK];
    for (size_t index = 0u; index < ET_KERNEL_MAX_RANK; index++) {
      rank64[index] = 1u;
    }
    tensor = create_tensor(ET_KERNEL_MAX_RANK, rank64);
    CHECK(et_f32_tensor_element_count_v1(tensor, &value, &error) == 0);
    CHECK(value == 1u);
    CHECK(et_f32_tensor_byte_length_v1(tensor, &value, &error) == 0);
    CHECK(value == sizeof(uint32_t));
    destroy_tensor(&tensor);
  }

  tensor = create_tensor(0u, NULL);
  CHECK(et_f32_tensor_element_count_v1(tensor, &value, &error) == 0);
  CHECK(value == 1u);
  CHECK(et_f32_tensor_byte_length_v1(tensor, &value, &error) == 0);
  CHECK(value == sizeof(uint32_t));
  destroy_tensor(&tensor);

  for (size_t index = 0u; index < 3u; index++) {
    et_f32_tensor_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *view = NULL;
    tensor = create_tensor(3u, zeros[index]);
    CHECK(et_f32_tensor_element_count_v1(tensor, &value, &error) == 0);
    CHECK(value == 0u);
    CHECK(et_f32_tensor_byte_length_v1(tensor, &value, &error) == 0);
    CHECK(value == 0u);
    CHECK(et_f32_tensor_copy_bits_from_v1(tensor, NULL, 0u, &error) == 0);
    CHECK(et_f32_tensor_copy_bits_to_v1(tensor, NULL, 0u, &error) == 0);
    CHECK(et_f32_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
    CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
    CHECK(view->data == NULL && view->byte_length == 0u);
    CHECK(view->rank == 3u && view->shape != NULL);
    CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
    destroy_tensor(&tensor);
  }

  tensor = create_tensor(2u, exact_shape);
  CHECK(et_f32_tensor_copy_bits_from_v1(
            tensor, exact, sizeof(exact) / sizeof(exact[0]), &error) == 0);
  expect_bits(tensor, exact, sizeof(exact) / sizeof(exact[0]));
  {
    uint32_t before[12];
    uint32_t output[12];
    memcpy(before, exact, sizeof(before));
    memset(output, 0x6b, sizeof(output));
    expect_error(et_f32_tensor_copy_bits_from_v1(tensor, exact, 11u, &error),
                 &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                 ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
    expect_bits(tensor, before, 12u);
    expect_error(et_f32_tensor_copy_bits_to_v1(tensor, output, 11u, &error),
                 &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
                 ET_F32_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
    for (size_t index = 0u; index < 12u; index++) {
      CHECK(output[index] == UINT32_C(0x6b6b6b6b));
    }
  }
  destroy_tensor(&tensor);
}

static void test_shape_and_span_failures(void) {
  et_f32_tensor_error error;
  et_f32_tensor *tensor = NULL;
  uint64_t rank65[ET_KERNEL_MAX_RANK + 1u] = {0};
  uint64_t byte_overflow = (uint64_t)(SIZE_MAX / sizeof(uint32_t)) + 1u;
  uint64_t product_overflow[] = {UINT64_C(4294967296),
                                 UINT64_C(4294967296)};
  uint64_t one = 1u;
  _Alignas(uint64_t) unsigned char misaligned[sizeof(uint64_t) + 1u];
  union {
    et_f32_tensor_error error;
    uint64_t shape[33];
    unsigned char bytes[264];
  } alias;
  unsigned char before[sizeof(alias)];

  expect_error(et_f32_tensor_create_v1(1u, NULL, &tensor, &error), &error,
               ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INVALID_SHAPE);
  CHECK(tensor == NULL);
  expect_error(et_f32_tensor_create_v1(ET_KERNEL_MAX_RANK + 1u, rank65,
                                       &tensor, &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(tensor == NULL);
  expect_error(et_f32_tensor_create_v1(1u, &byte_overflow, &tensor, &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INTEGER_OVERFLOW);
  expect_error(et_f32_tensor_create_v1(2u, product_overflow, &tensor, &error),
               &error, ET_F32_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_F32_TENSOR_CODE_INTEGER_OVERFLOW);
  memcpy(misaligned + 1u, &one, sizeof(one));
  expect_error(et_f32_tensor_create_v1(
                   1u, (const uint64_t *)(const void *)(misaligned + 1u),
                   &tensor, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);

  memset(&alias, 0xa7, sizeof(alias));
  memcpy(before, alias.bytes, sizeof(before));
  CHECK(et_f32_tensor_create_v1(33u, alias.shape, &tensor, &alias.error) ==
        ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(tensor == NULL);
  CHECK(memcmp(alias.bytes, before, sizeof(before)) == 0);

  memset(&error, 0xd3, sizeof(error));
  memcpy(before, &error, sizeof(error));
  CHECK(et_f32_tensor_create_v1(
            65u, (const uint64_t *)(uintptr_t)(UINTPTR_MAX - 7u), &tensor,
            &error) == ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(tensor == NULL);
  CHECK(memcmp(&error, before, sizeof(error)) == 0);
}

static void test_borrow_lifetime_and_alignment(void) {
  const uint64_t shape[] = {2u, 3u};
  const uint32_t bits[] = {1u, 2u, 3u, 4u, 5u, 6u};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = create_tensor(2u, shape);
  et_f32_tensor *saved_tensor;
  et_f32_tensor_borrow *borrow = NULL;
  et_f32_tensor_borrow *saved_borrow;
  const et_kernel_tensor_view_v1 *view = NULL;
  const et_kernel_tensor_view_v1 *saved_view;

  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, 6u, &error) == 0);
  CHECK(et_f32_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  CHECK(borrow != NULL);
  saved_borrow = borrow;
  expect_error(et_f32_tensor_borrow_begin_v1(tensor, &saved_borrow, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL && view->struct_size == sizeof(*view));
  CHECK(strcmp(view->dtype, "f32") == 0);
  CHECK(strcmp(view->device, "cpu") == 0);
  CHECK(view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  CHECK(view->offset_bytes == 0u);
  CHECK(view->rank == 2u && view->shape[0] == 2u && view->shape[1] == 3u);
  CHECK(view->byte_length == sizeof(bits));
  CHECK(view->data != NULL);
  CHECK((uintptr_t)view->data % _Alignof(float) == 0u);
  expect_error(et_f32_tensor_copy_bits_from_v1(tensor, bits, 6u, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  saved_tensor = tensor;
  expect_error(et_f32_tensor_destroy_v1(&tensor, &error), &error,
               ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(tensor == saved_tensor);
  expect_bits(tensor, bits, 6u);

  saved_borrow = borrow;
  saved_view = view;
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
  view = NULL;
  expect_error(et_f32_tensor_borrow_view_v1(saved_borrow, &view, &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_STATE,
               ET_F32_TENSOR_CODE_INVALID_HANDLE);
  CHECK(view == NULL);
  CHECK(saved_view != NULL); /* Pointer identity only; never dereference stale view. */

  saved_tensor = tensor;
  destroy_tensor(&tensor);
  {
    size_t rank = 17u;
    expect_error(et_f32_tensor_rank_v1(saved_tensor, &rank, &error), &error,
                 ET_F32_TENSOR_ERROR_INVALID_STATE,
                 ET_F32_TENSOR_CODE_INVALID_HANDLE);
    CHECK(rank == 17u);
  }
}

#ifdef ET_F32_TENSOR_TESTING
static void test_owned_aliases_and_error_clear(void) {
  const uint64_t shape[] = {2u, 3u};
  const uint32_t bits[] = {UINT32_C(0x3f800000), UINT32_C(0x40000000),
                           UINT32_C(0x40400000), UINT32_C(0x40800000),
                           UINT32_C(0x40a00000), UINT32_C(0x40c00000)};
  et_f32_tensor_error error;
  et_f32_tensor *tensor = create_tensor(2u, shape);
  et_f32_tensor *other = create_tensor(2u, shape);
  et_f32_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const void *regions[3];

  CHECK(et_f32_tensor_copy_bits_from_v1(tensor, bits, 6u, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(other, bits, 6u, &error) == 0);
  regions[0] = et_f32_tensor_test_shape_storage_v1(other);
  regions[1] = et_f32_tensor_test_stride_storage_v1(other);
  regions[2] = et_f32_tensor_test_data_storage_v1(other);
  for (size_t index = 0u; index < 3u; index++) {
    CHECK(regions[index] != NULL);
    expect_error(et_f32_tensor_copy_bits_from_v1(
                     tensor, (const uint32_t *)regions[index], 6u, &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_F32_TENSOR_CODE_INVALID_BUFFER);
    expect_bits(tensor, bits, 6u);
    expect_error(et_f32_tensor_copy_bits_to_v1(
                     tensor, (uint32_t *)(void *)regions[index], 6u, &error),
                 &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_F32_TENSOR_CODE_INVALID_BUFFER);
    expect_bits(tensor, bits, 6u);
    expect_bits(other, bits, 6u);
  }

  CHECK(et_f32_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  expect_error(et_f32_tensor_borrow_view_v1(
                   borrow,
                   (const et_kernel_tensor_view_v1 **)(void *)
                       et_f32_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_F32_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_F32_TENSOR_CODE_INVALID_BUFFER);
  expect_bits(other, bits, 6u);
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);

  destroy_tensor(&tensor);
  destroy_tensor(&other);

  {
    const uint64_t error_shape[] = {66u};
    uint32_t zero[66] = {0};
    et_f32_tensor_copy_plan *plan = NULL;
    tensor = create_tensor(1u, error_shape);
    CHECK(et_f32_tensor_copy_bits_from_v1(tensor, zero, 66u, &error) == 0);
    CHECK(et_f32_tensor_copy_plan_prepare_v1(
              0u, NULL, &plan,
              (et_f32_tensor_error *)(void *)
                  et_f32_tensor_test_data_storage_v1(tensor)) ==
          ET_F32_TENSOR_ERROR_INVALID_ARGUMENT);
    CHECK(plan == NULL);
    expect_bits(tensor, zero, 66u);
    et_f32_tensor_error_clear_v1(
        (et_f32_tensor_error *)(void *)
            et_f32_tensor_test_data_storage_v1(tensor));
    expect_bits(tensor, zero, 66u);
    destroy_tensor(&tensor);
  }
}

static void test_allocation_failpoints(void) {
  const uint64_t nonempty[] = {2u, 3u};
  const uint64_t empty[] = {2u, 0u};
  const struct {
    size_t rank;
    const uint64_t *shape;
  } cases[] = {{2u, nonempty}, {2u, empty}, {0u, NULL}};
  et_f32_tensor_error error;

  for (size_t case_index = 0u;
       case_index < sizeof(cases) / sizeof(cases[0]); case_index++) {
    int succeeded = 0;
    size_t failed_sites = 0u;
    for (size_t allowed = 0u; allowed < 16u; allowed++) {
      et_f32_tensor *tensor = NULL;
      et_f32_tensor_test_fail_alloc_after_v1(allowed);
      int32_t result = et_f32_tensor_create_v1(
          cases[case_index].rank, cases[case_index].shape, &tensor, &error);
      if (result == 0) {
        succeeded = 1;
        CHECK(tensor != NULL);
        et_f32_tensor_test_reset_allocator_v1();
        destroy_tensor(&tensor);
        break;
      }
      expect_error(result, &error, ET_F32_TENSOR_ERROR_INTERNAL,
                   ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
      CHECK(tensor == NULL);
      failed_sites++;
    }
    CHECK(succeeded);
    CHECK(failed_sites > 0u);
    et_f32_tensor_test_reset_allocator_v1();
  }

  {
    et_f32_tensor *tensor = create_tensor(0u, NULL);
    et_f32_tensor_borrow *borrow = NULL;
    et_f32_tensor_test_fail_alloc_after_v1(0u);
    expect_error(et_f32_tensor_borrow_begin_v1(tensor, &borrow, &error),
                 &error, ET_F32_TENSOR_ERROR_INTERNAL,
                 ET_F32_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(borrow == NULL);
    et_f32_tensor_test_reset_allocator_v1();
    destroy_tensor(&tensor);
  }
}
#endif

static const et_kernel_provider_v1 *resolve_i2(void *context,
                                               const char *symbol) {
  CHECK(context == NULL);
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_f32_tensor_provider_v1();
}

static void test_provider_and_dispatch(void) {
  const uint64_t shape[] = {5u};
  const uint64_t rank2[] = {1u, 5u};
  uint64_t rank64[ET_KERNEL_MAX_RANK];
  const uint32_t input_bits[] = {UINT32_C(0x00000000),
                                 UINT32_C(0x80000000),
                                 UINT32_C(0x00000001),
                                 UINT32_C(0x7f800000),
                                 UINT32_C(0x7fc12345)};
  const uint32_t sentinels[] = {UINT32_C(0x11111111), UINT32_C(0x22222222),
                                UINT32_C(0x33333333), UINT32_C(0x44444444),
                                UINT32_C(0x55555555)};
  et_f32_tensor_error error;
  et_kernel_error kernel_error;
  et_kernel_runtime *runtime = NULL;
  et_f32_tensor *input = create_tensor(1u, shape);
  et_f32_tensor *output = create_tensor(1u, shape);
  et_f32_tensor_borrow *input_borrow = NULL;
  et_f32_tensor_borrow *output_borrow = NULL;
  const et_kernel_tensor_view_v1 *input_view = NULL;
  const et_kernel_tensor_view_v1 *output_view = NULL;
  et_kernel_request_v1 request;
  et_kernel_call_v1 call;
  et_kernel_tensor_view_v1 copied_input;
  et_kernel_tensor_view_v1 copied_output;
  unsigned char misaligned[sizeof(input_bits) + 1u];

  const et_kernel_provider_v1 *provider = et_f32_tensor_provider_v1();
  CHECK(provider != NULL);
  CHECK(provider->abi_major == 1u);
  CHECK(provider->struct_size >= ET_KERNEL_PROVIDER_V1_0_SIZE);
  CHECK(provider->capability_count == 1u);
  CHECK(((const et_kernel_capability_v1 *)provider->capabilities)
            ->shape_range_count == 2u);
  CHECK(provider->validate_call != NULL && provider->invoke_call != NULL);
  CHECK(et_kernel_runtime_discover(resolve_i2, NULL, &runtime,
                                   &kernel_error) == 0);
  CHECK(runtime != NULL);
  CHECK(et_kernel_runtime_capability_find(runtime, "tensor.f32") != NULL);

  memset(&request, 0, sizeof(request));
  request.struct_size = sizeof(request);
  request.operation = "storage.copy";
  request.dtype = "f32";
  request.device = "cpu";
  request.rank = 1u;
  request.shape = shape;
  request.deterministic = 1u;
  CHECK(et_kernel_runtime_capability_require(runtime, "tensor.f32", &request,
                                             NULL, &kernel_error) == 0);
  for (size_t index = 0u; index < ET_KERNEL_MAX_RANK; index++) {
    rank64[index] = 1u;
  }
  request.rank = ET_KERNEL_MAX_RANK;
  request.shape = rank64;
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.f32", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.rank = 2u;
  request.shape = rank2;
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.f32", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.rank = 1u;
  request.shape = shape;
  request.dtype = "i64";
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.f32", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.dtype = "f32";
  request.device = "gpu:0";
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.f32", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.device = "cpu";

  CHECK(et_f32_tensor_copy_bits_from_v1(input, input_bits, 5u, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(output, sentinels, 5u, &error) == 0);
  CHECK(et_f32_tensor_borrow_begin_v1(input, &input_borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_begin_v1(output, &output_borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(input_borrow, &input_view, &error) == 0);
  CHECK(et_f32_tensor_borrow_view_v1(output_borrow, &output_view, &error) ==
        0);
  memset(&call, 0, sizeof(call));
  call.struct_size = sizeof(call);
  call.capability = "tensor.f32";
  call.request = &request;
  call.input_count = 1u;
  call.input_stride = sizeof(*input_view);
  call.input_bytes = sizeof(*input_view);
  call.inputs = input_view;
  call.output_count = 1u;
  call.output_stride = sizeof(*output_view);
  call.output_bytes = sizeof(*output_view);
  call.outputs = (void *)output_view;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &kernel_error) == 0);
  expect_bits(input, input_bits, 5u);
  expect_bits(output, input_bits, 5u);

  copied_input = *input_view;
  copied_output = *output_view;
  copied_output.data = copied_input.data;
  call.inputs = &copied_input;
  call.outputs = &copied_output;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT);
  expect_bits(input, input_bits, 5u);
  expect_bits(output, input_bits, 5u);

  memcpy(misaligned + 1u, input_bits, sizeof(input_bits));
  copied_input = *input_view;
  copied_output = *output_view;
  copied_input.data = misaligned + 1u;
  call.inputs = &copied_input;
  call.outputs = &copied_output;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER);
  expect_bits(output, input_bits, 5u);

  CHECK(et_f32_tensor_borrow_end_v1(&input_borrow, &error) == 0);
  CHECK(et_f32_tensor_borrow_end_v1(&output_borrow, &error) == 0);
  et_kernel_runtime_destroy(runtime);
  destroy_tensor(&input);
  destroy_tensor(&output);
}

int main(void) {
  test_version_nulls_and_forgery();
  test_shapes_storage_and_exact_bits();
  test_shape_and_span_failures();
  test_borrow_lifetime_and_alignment();
  test_provider_and_dispatch();
#ifdef ET_F32_TENSOR_TESTING
  test_owned_aliases_and_error_clear();
  test_allocation_failpoints();
#endif
  if (failures != 0) {
    (void)fprintf(stderr, "I2 storage FAIL: %d of %d checks failed\n", failures,
                  checks);
    return 1;
  }
  (void)printf("I2 storage PASS: %d exact-f32, ownership, and K1 checks\n",
               checks);
  return 0;
}
