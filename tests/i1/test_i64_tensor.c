#include "eshkol_transformer/i64_tensor.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define I1_TEST_HAS_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define I1_TEST_HAS_ASAN 1
#endif
#if defined(I1_TEST_HAS_ASAN)
#include <sanitizer/asan_interface.h>
#endif

static int checks;
static int failures;

#define CHECK(condition)                                                        \
  do {                                                                          \
    checks++;                                                                    \
    if (!(condition)) {                                                          \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);      \
      failures++;                                                                \
    }                                                                            \
  } while (0)

_Static_assert(ET_I64_TENSOR_ABI_MAJOR == 1u, "I1 ABI major changed");
_Static_assert(ET_I64_TENSOR_ABI_MINOR == 0u, "I1 ABI minor changed");
_Static_assert(sizeof(int64_t) == 8u, "I1 requires exact signed 64-bit values");
_Static_assert(sizeof(uintptr_t) == 8u && UINTPTR_MAX == UINT64_MAX,
               "I1 address-end probes require the frozen x86-64 ABI");
_Static_assert(_Alignof(uint64_t) == 8u,
               "I1 shape alignment changed on the frozen x86-64 ABI");
_Static_assert(sizeof(et_i64_tensor_error) == 264u,
               "I1 error layout changed");
_Static_assert(offsetof(et_i64_tensor_error, category) == 0u,
               "I1 error category offset changed");
_Static_assert(offsetof(et_i64_tensor_error, code) == 4u,
               "I1 error code offset changed");
_Static_assert(offsetof(et_i64_tensor_error, operation) == 8u,
               "I1 error operation offset changed");
_Static_assert(offsetof(et_i64_tensor_error, message) == 72u,
               "I1 error message offset changed");

typedef union i1_alias_storage {
  et_i64_tensor_error error;
  et_i64_tensor *tensor;
  et_i64_tensor_borrow *borrow;
  const et_kernel_tensor_view_v1 *view;
  size_t size_value;
  uint64_t extent;
  int64_t values[33];
  unsigned char bytes[sizeof(et_i64_tensor_error)];
} i1_alias_storage;

_Static_assert(sizeof(i1_alias_storage) == sizeof(et_i64_tensor_error),
               "alias fixture must cover one complete I1 error record");

typedef union i1_rank65_shape_alias_storage {
  et_i64_tensor_error error_alignment;
  uint64_t shape[65];
  unsigned char bytes[65u * sizeof(uint64_t)];
} i1_rank65_shape_alias_storage;

typedef union i1_rank66_shape_alias_storage {
  et_i64_tensor_error error_alignment;
  uint64_t shape[66];
  unsigned char bytes[66u * sizeof(uint64_t)];
} i1_rank66_shape_alias_storage;

typedef union i1_rank257_shape_alias_storage {
  et_i64_tensor_error error_alignment;
  uint64_t shape[257];
  unsigned char bytes[257u * sizeof(uint64_t)];
} i1_rank257_shape_alias_storage;

_Static_assert(sizeof(i1_rank65_shape_alias_storage) ==
                   65u * sizeof(uint64_t),
               "rank-65 fixture must be the exact authorized shape span");
_Static_assert(sizeof(i1_rank66_shape_alias_storage) ==
                   66u * sizeof(uint64_t),
               "rank-66 fixture must be the exact authorized shape span");
_Static_assert(sizeof(i1_rank257_shape_alias_storage) ==
                   257u * sizeof(uint64_t),
               "rank-257 fixture must be the exact authorized shape span");

static void poison_invalid_rank_fixture(void *storage, size_t size) {
#if defined(I1_TEST_HAS_ASAN)
  __asan_poison_memory_region(storage, size);
#else
  (void)storage;
  (void)size;
#endif
}

static void unpoison_invalid_rank_fixture(void *storage, size_t size) {
#if defined(I1_TEST_HAS_ASAN)
  __asan_unpoison_memory_region(storage, size);
#else
  (void)storage;
  (void)size;
#endif
}

static void expect_alias_rejection(int32_t result,
                                   const i1_alias_storage *storage,
                                   const unsigned char *before) {
  CHECK(result == ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(storage->bytes, before, sizeof(storage->bytes)) == 0);
}

static void expect_invalid_rank_shape_alias_rejection(
    size_t rank, unsigned char *storage, size_t storage_size,
    size_t error_offset, et_i64_tensor *initial_output) {
  unsigned char before[sizeof(i1_rank257_shape_alias_storage)];
  unsigned char output_before[sizeof(initial_output)];
  et_i64_tensor *output = initial_output;
  et_i64_tensor_error *error;
  int32_t result;

  CHECK(rank <= SIZE_MAX / sizeof(uint64_t));
  CHECK(storage_size == rank * sizeof(uint64_t));
  CHECK(storage_size <= sizeof(before));
  CHECK(error_offset % _Alignof(et_i64_tensor_error) == 0u);
  CHECK(error_offset <= storage_size);
  CHECK(sizeof(*error) <= storage_size - error_offset);
  if (rank > SIZE_MAX / sizeof(uint64_t) ||
      storage_size != rank * sizeof(uint64_t) ||
      storage_size > sizeof(before) ||
      error_offset % _Alignof(et_i64_tensor_error) != 0u ||
      error_offset > storage_size ||
      sizeof(*error) > storage_size - error_offset) {
    return;
  }

  memset(storage, 0xa7, storage_size);
  memcpy(before, storage, storage_size);
  memcpy(output_before, &output, sizeof(output));
  error = (et_i64_tensor_error *)(void *)(storage + error_offset);

  /*
   * Poisoning proves the invalid-rank preflight uses the authorized span only
   * for address-range validation: it may neither read shape extents nor write
   * the aliased diagnostic before rejecting the overlap.
   */
  poison_invalid_rank_fixture(storage, storage_size);
  result = et_i64_tensor_create_v1(
      rank, (const uint64_t *)(const void *)storage, &output, error);
  unpoison_invalid_rank_fixture(storage, storage_size);

  CHECK(result == ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(storage, before, storage_size) == 0);
  CHECK(memcmp(&output, output_before, sizeof(output)) == 0);
}

static void expect_error(int32_t result, const et_i64_tensor_error *error,
                         et_i64_tensor_error_category category,
                         et_i64_tensor_error_code code) {
  if (result != (int32_t)category || error->category != category ||
      error->code != code) {
    fprintf(stderr,
            "I1 error mismatch: result=%d category=%u code=%u operation=%s "
            "(expected category=%u code=%u)\n",
            result, error->category, error->code, error->operation, category,
            code);
  }
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

static const et_kernel_provider_v1 *resolve_i1(void *context,
                                               const char *symbol) {
  CHECK(context == NULL);
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_i64_tensor_provider_v1();
}

static et_i64_tensor *create_tensor(size_t rank, const uint64_t *shape) {
  et_i64_tensor *tensor = NULL;
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_create_v1(rank, shape, &tensor, &error) == 0);
  CHECK(tensor != NULL);
  CHECK(error.category == ET_I64_TENSOR_ERROR_NONE);
  CHECK(error.code == ET_I64_TENSOR_CODE_OK);
  return tensor;
}

static void destroy_tensor(et_i64_tensor **tensor) {
  et_i64_tensor_error error;
  CHECK(et_i64_tensor_destroy_v1(tensor, &error) == 0);
  CHECK(*tensor == NULL);
}

static void expect_values(const et_i64_tensor *tensor,
                          const int64_t *expected, size_t count) {
  int64_t actual[16];
  et_i64_tensor_error error;
  CHECK(count <= sizeof(actual) / sizeof(actual[0]));
  memset(actual, 0x5a, sizeof(actual));
  CHECK(et_i64_tensor_copy_to_v1(tensor, actual, count, &error) == 0);
  CHECK(memcmp(actual, expected, count * sizeof(*expected)) == 0);
}

static void test_version_and_nulls(void) {
  et_i64_tensor_error error;
  et_i64_tensor *tensor = NULL;
  size_t result = 77u;
  uint64_t extent = 88u;

  CHECK(et_i64_tensor_abi_major_v1() == 1);
  CHECK(et_i64_tensor_abi_minor_v1() == 0);
  CHECK(strcmp(ET_I64_TENSOR_ERROR_SOURCE_DOMAIN, "i64-tensor") == 0);
  CHECK(et_i64_tensor_abi_require_v1(1u, 0u, &error) == 0);
  CHECK(error.category == ET_I64_TENSOR_ERROR_NONE);
  expect_error(et_i64_tensor_abi_require_v1(2u, 0u, &error), &error,
               ET_I64_TENSOR_ERROR_VERSION_MISMATCH,
               ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH);
  expect_error(et_i64_tensor_abi_require_v1(1u, 1u, &error), &error,
               ET_I64_TENSOR_ERROR_VERSION_MISMATCH,
               ET_I64_TENSOR_CODE_ABI_VERSION_MISMATCH);
  CHECK(et_i64_tensor_abi_require_v1(2u, 0u, NULL) ==
        ET_I64_TENSOR_ERROR_VERSION_MISMATCH);

  expect_error(et_i64_tensor_create_v1(0u, NULL, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  tensor = create_tensor(0u, NULL);
  {
    et_i64_tensor *const saved = tensor;
    expect_error(et_i64_tensor_create_v1(0u, NULL, &tensor, &error), &error,
                 ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    CHECK(tensor == saved);
  }
  expect_error(et_i64_tensor_rank_v1(NULL, &result, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(result == 77u);
  expect_error(et_i64_tensor_rank_v1((const et_i64_tensor *)tensor, NULL,
                                     &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_shape_at_v1(NULL, 0u, &extent, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(extent == 88u);
  CHECK(et_i64_tensor_destroy_v1(&tensor, &error) == 0);
  CHECK(tensor == NULL);
  CHECK(et_i64_tensor_destroy_v1(&tensor, &error) == 0);
  expect_error(et_i64_tensor_destroy_v1(NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
}

static void test_shapes_and_strides(void) {
  const uint64_t shape[] = {2u, 3u, 4u};
  const size_t expected_strides[] = {96u, 32u, 8u};
  const uint64_t zero_shapes[][3] = {{0u, 3u, 4u}, {2u, 0u, 4u},
                                    {2u, 3u, 0u}};
  et_i64_tensor_error error;
  et_i64_tensor *tensor = create_tensor(3u, shape);
  size_t value;
  uint64_t extent;

  CHECK(et_i64_tensor_rank_v1(tensor, &value, &error) == 0);
  CHECK(value == 3u);
  for (size_t index = 0; index < 3u; index++) {
    CHECK(et_i64_tensor_shape_at_v1(tensor, index, &extent, &error) == 0);
    CHECK(extent == shape[index]);
    CHECK(et_i64_tensor_stride_bytes_at_v1(tensor, index, &value, &error) ==
          0);
    CHECK(value == expected_strides[index]);
  }
  CHECK(et_i64_tensor_element_count_v1(tensor, &value, &error) == 0);
  CHECK(value == 24u);
  CHECK(et_i64_tensor_byte_length_v1(tensor, &value, &error) == 0);
  CHECK(value == 192u);
  extent = 992u;
  expect_error(et_i64_tensor_shape_at_v1(tensor, 3u, &extent, &error), &error,
               ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INVALID_SHAPE);
  CHECK(extent == 992u);
  value = 991u;
  expect_error(et_i64_tensor_stride_bytes_at_v1(tensor, 3u, &value, &error),
               &error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INVALID_SHAPE);
  CHECK(value == 991u);
  expect_error(et_i64_tensor_rank_v1(tensor, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_shape_at_v1(tensor, 0u, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_stride_bytes_at_v1(tensor, 0u, NULL, &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_element_count_v1(tensor, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_byte_length_v1(tensor, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  destroy_tensor(&tensor);

  tensor = create_tensor(0u, NULL);
  CHECK(et_i64_tensor_element_count_v1(tensor, &value, &error) == 0);
  CHECK(value == 1u);
  CHECK(et_i64_tensor_byte_length_v1(tensor, &value, &error) == 0);
  CHECK(value == 8u);
  {
    et_i64_tensor_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *view = NULL;
    CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
    CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
    CHECK(view->rank == 0u && view->shape == NULL);
    CHECK(view->data != NULL && view->byte_length == sizeof(int64_t));
    CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  }
  destroy_tensor(&tensor);

  for (size_t index = 0; index < 3u; index++) {
    tensor = create_tensor(3u, zero_shapes[index]);
    CHECK(et_i64_tensor_element_count_v1(tensor, &value, &error) == 0);
    CHECK(value == 0u);
    CHECK(et_i64_tensor_byte_length_v1(tensor, &value, &error) == 0);
    CHECK(value == 0u);
    {
      et_i64_tensor_borrow *borrow = NULL;
      const et_kernel_tensor_view_v1 *view = NULL;
      CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
      CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
      CHECK(view->data == NULL && view->byte_length == 0u);
      CHECK(view->rank == 3u && view->shape != NULL);
      CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
    }
    destroy_tensor(&tensor);
  }

  {
    uint64_t mutable_shape[] = {2u, 3u};
    tensor = create_tensor(2u, mutable_shape);
    mutable_shape[0] = 99u;
    mutable_shape[1] = 88u;
    CHECK(et_i64_tensor_shape_at_v1(tensor, 0u, &extent, &error) == 0);
    CHECK(extent == 2u);
    CHECK(et_i64_tensor_shape_at_v1(tensor, 1u, &extent, &error) == 0);
    CHECK(extent == 3u);
    destroy_tensor(&tensor);
  }
}

static void test_shape_failures(void) {
  et_i64_tensor_error error;
  et_i64_tensor *tensor = NULL;
  uint64_t one = 1u;
  uint64_t invalid_rank_shape[ET_KERNEL_MAX_RANK + 1u];
  uint64_t overflow = (uint64_t)(SIZE_MAX / sizeof(int64_t)) + 1u;
  uint64_t product_overflow[] = {UINT64_C(4294967296),
                                 UINT64_C(4294967296)};
  _Alignas(uint64_t) unsigned char shape_storage[sizeof(uint64_t) + 1u];

  memset(invalid_rank_shape, 0, sizeof(invalid_rank_shape));

  expect_error(et_i64_tensor_create_v1(1u, NULL, &tensor, &error), &error,
               ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INVALID_SHAPE);
  CHECK(tensor == NULL);
  expect_error(et_i64_tensor_create_v1(ET_KERNEL_MAX_RANK + 1u,
                                       invalid_rank_shape, &tensor, &error),
               &error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(tensor == NULL);
  expect_error(et_i64_tensor_create_v1(1u, &overflow, &tensor, &error), &error,
               ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(tensor == NULL);
  expect_error(et_i64_tensor_create_v1(2u, product_overflow, &tensor, &error),
               &error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(tensor == NULL);

  memcpy(shape_storage + 1u, &one, sizeof(one));
  expect_error(et_i64_tensor_create_v1(
                   1u,
                   (const uint64_t *)(const void *)(shape_storage + 1u),
                   &tensor, &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(tensor == NULL);
}

static void test_invalid_rank_shape_error_alias_atomicity(void) {
  i1_rank65_shape_alias_storage rank65;
  i1_rank66_shape_alias_storage rank66;
  i1_rank257_shape_alias_storage rank257;
  i1_alias_storage unrepresentable;
  unsigned char unrepresentable_before[sizeof(unrepresentable)];
  unsigned char output_before[sizeof(et_i64_tensor *)];
  et_i64_tensor_error ordinary_error;
  et_i64_tensor_error address_end_error;
  unsigned char address_end_before[sizeof(address_end_error)];
  et_i64_tensor *output = NULL;
  et_i64_tensor *preserved = create_tensor(0u, NULL);
  const uint64_t *const address_end_shape =
      (const uint64_t *)(uintptr_t)(UINTPTR_MAX - 7u);
  const size_t unrepresentable_rank =
      SIZE_MAX / sizeof(uint64_t) + 1u;
  int32_t result;

  /* The rank-65 regression: error starts at the authorized shape span. */
  expect_invalid_rank_shape_alias_rejection(
      65u, rank65.bytes, sizeof(rank65), 0u, NULL);

  /* Exercise an interior overlap and exact preservation of a non-NULL slot. */
  expect_invalid_rank_shape_alias_rejection(
      66u, rank66.bytes, sizeof(rank66), 16u, preserved);

  /* A larger bounded rank puts the complete diagnostic at the span's tail. */
  expect_invalid_rank_shape_alias_rejection(
      257u, rank257.bytes, sizeof(rank257),
      sizeof(rank257) - sizeof(et_i64_tensor_error), NULL);

  /*
   * This deliberately malformed declaration has no representable byte span.
   * The conservative path must reject it without reading the small probe or
   * attempting to write its aliased error record.
   */
  memset(&unrepresentable, 0xbc, sizeof(unrepresentable));
  memcpy(unrepresentable_before, unrepresentable.bytes,
         sizeof(unrepresentable_before));
  memcpy(output_before, &output, sizeof(output));
  poison_invalid_rank_fixture(&unrepresentable, sizeof(unrepresentable));
  result = et_i64_tensor_create_v1(
      unrepresentable_rank,
      (const uint64_t *)(const void *)unrepresentable.bytes, &output,
      &unrepresentable.error);
  unpoison_invalid_rank_fixture(&unrepresentable, sizeof(unrepresentable));
  CHECK(result == ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(unrepresentable.bytes, unrepresentable_before,
               sizeof(unrepresentable_before)) == 0);
  CHECK(memcmp(&output, output_before, sizeof(output)) == 0);

  /*
   * Multiplication is representable here, but the aligned numeric pointer's
   * exclusive shape end is not.  It is intentionally non-dereferenceable.
   */
  memset(&address_end_error, 0xd3, sizeof(address_end_error));
  memcpy(address_end_before, &address_end_error, sizeof(address_end_before));
  output = preserved;
  memcpy(output_before, &output, sizeof(output));
  result = et_i64_tensor_create_v1(65u, address_end_shape, &output,
                                   &address_end_error);
  CHECK(result == ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
  CHECK(memcmp(&address_end_error, address_end_before,
               sizeof(address_end_before)) == 0);
  CHECK(memcmp(&output, output_before, sizeof(output)) == 0);

  /* A NULL shape authorizes no storage and permits the ordinary rank error. */
  output = NULL;
  expect_error(et_i64_tensor_create_v1(SIZE_MAX, NULL, &output,
                                       &ordinary_error),
               &ordinary_error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_INTEGER_OVERFLOW);
  CHECK(output == NULL);

  destroy_tensor(&preserved);
}

static void test_exact_round_trip_and_atomicity(void) {
  static const int64_t boundaries[] = {
      INT64_MIN, INT64_MIN + 1, -INT64_C(9007199254740993), -1, 0, 1,
      INT64_C(9007199254740993), INT64_MAX - 1, INT64_MAX};
  const uint64_t shape[] = {
      sizeof(boundaries) / sizeof(boundaries[0])};
  int64_t replacement[sizeof(boundaries) / sizeof(boundaries[0])];
  int64_t destination[sizeof(boundaries) / sizeof(boundaries[0])];
  unsigned char misaligned[sizeof(boundaries) + 1u];
  et_i64_tensor_error error;
  et_i64_tensor *tensor = create_tensor(1u, shape);

  expect_error(et_i64_tensor_copy_from_v1(NULL, boundaries, shape[0], &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_copy_to_v1(NULL, destination, shape[0], &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);

  CHECK(et_i64_tensor_copy_from_v1(tensor, boundaries, shape[0], &error) == 0);
  expect_values(tensor, boundaries, shape[0]);
  memset(replacement, 0x2d, sizeof(replacement));
  expect_error(et_i64_tensor_copy_from_v1(tensor, replacement, shape[0] - 1u,
                                          &error),
               &error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
  expect_values(tensor, boundaries, shape[0]);
  expect_error(et_i64_tensor_copy_from_v1(tensor, NULL, shape[0], &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_values(tensor, boundaries, shape[0]);

  memset(destination, 0x6b, sizeof(destination));
  expect_error(et_i64_tensor_copy_to_v1(tensor, destination, shape[0] - 1u,
                                        &error),
               &error, ET_I64_TENSOR_ERROR_SHAPE_MISMATCH,
               ET_I64_TENSOR_CODE_BUFFER_SIZE_MISMATCH);
  for (size_t index = 0; index < sizeof(destination); index++) {
    CHECK(((const unsigned char *)destination)[index] == 0x6bu);
  }
  expect_error(et_i64_tensor_copy_to_v1(tensor, NULL, shape[0], &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);

  expect_error(et_i64_tensor_copy_from_v1(
                   tensor, (const int64_t *)(const void *)(misaligned + 1u),
                   shape[0], &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  expect_values(tensor, boundaries, shape[0]);
  expect_error(et_i64_tensor_copy_to_v1(
                   tensor, (int64_t *)(void *)(misaligned + 1u), shape[0],
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  destroy_tensor(&tensor);

  {
    const uint64_t empty_shape[] = {0u};
    tensor = create_tensor(1u, empty_shape);
    CHECK(et_i64_tensor_copy_from_v1(tensor, NULL, 0u, &error) == 0);
    CHECK(et_i64_tensor_copy_to_v1(tensor, NULL, 0u, &error) == 0);
    destroy_tensor(&tensor);
  }
}

static void test_error_alias_atomicity(void) {
  const uint64_t shape[] = {33u};
  const uint64_t invalid_rank_shape[ET_KERNEL_MAX_RANK + 1u] = {0u};
  int64_t expected[33];
  int64_t actual[33];
  et_i64_tensor_error error;
  et_i64_tensor *tensor = create_tensor(1u, shape);
  et_i64_tensor *handle;
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  i1_alias_storage alias;
  unsigned char before[sizeof(alias.bytes)];

  for (size_t index = 0u; index < 33u; index++) {
    expected[index] = index == 0u ? INT64_MIN
                                  : index == 32u ? INT64_MAX
                                                  : (int64_t)index - 16;
  }
  CHECK(et_i64_tensor_copy_from_v1(tensor, expected, 33u, &error) == 0);

#define PREPARE_ALIAS(byte)                                                     \
  do {                                                                          \
    memset(&alias, (byte), sizeof(alias));                                      \
    memcpy(before, alias.bytes, sizeof(before));                                \
  } while (0)

  PREPARE_ALIAS(0x91);
  expect_alias_rejection(
      et_i64_tensor_rank_v1(tensor, &alias.size_value, &alias.error), &alias,
      before);
  PREPARE_ALIAS(0x92);
  expect_alias_rejection(
      et_i64_tensor_shape_at_v1(tensor, 0u, &alias.extent, &alias.error),
      &alias, before);
  PREPARE_ALIAS(0x93);
  expect_alias_rejection(et_i64_tensor_stride_bytes_at_v1(
                             tensor, 0u, &alias.size_value, &alias.error),
                         &alias, before);
  PREPARE_ALIAS(0x94);
  expect_alias_rejection(et_i64_tensor_element_count_v1(
                             tensor, &alias.size_value, &alias.error),
                         &alias, before);
  PREPARE_ALIAS(0x95);
  expect_alias_rejection(et_i64_tensor_byte_length_v1(
                             tensor, &alias.size_value, &alias.error),
                         &alias, before);

  /* The invalid dimension proves failure paths reject the alias too. */
  PREPARE_ALIAS(0x96);
  expect_alias_rejection(
      et_i64_tensor_shape_at_v1(tensor, 1u, &alias.extent, &alias.error),
      &alias, before);

  memset(&alias, 0, sizeof(alias));
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_create_v1(0u, NULL, &alias.tensor, &alias.error), &alias,
      before);
  memset(&alias, 0, sizeof(alias));
  alias.extent = 1u;
  handle = NULL;
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_create_v1(1u, &alias.extent, &handle, &alias.error),
      &alias, before);
  CHECK(handle == NULL);
  memset(&alias, 0, sizeof(alias));
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(et_i64_tensor_create_v1(
                             ET_KERNEL_MAX_RANK + 1u, invalid_rank_shape,
                             &alias.tensor, &alias.error),
                         &alias, before);

  memset(&alias, 0, sizeof(alias));
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_borrow_begin_v1(tensor, &alias.borrow, &alias.error),
      &alias, before);
  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  CHECK(borrow != NULL);
  memset(&alias, 0, sizeof(alias));
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_borrow_view_v1(borrow, &alias.view, &alias.error), &alias,
      before);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  CHECK(view != NULL);

  memset(&alias, 0, sizeof(alias));
  alias.borrow = borrow;
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_borrow_end_v1(&alias.borrow, &alias.error), &alias,
      before);
  CHECK(alias.borrow == borrow);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);

  PREPARE_ALIAS(0xa1);
  expect_alias_rejection(
      et_i64_tensor_copy_to_v1(tensor, alias.values, 33u, &alias.error),
      &alias, before);
  PREPARE_ALIAS(0xa2);
  expect_alias_rejection(
      et_i64_tensor_copy_to_v1(tensor, alias.values, 32u, &alias.error),
      &alias, before);
  PREPARE_ALIAS(0xa3);
  expect_alias_rejection(
      et_i64_tensor_copy_from_v1(tensor, alias.values, 33u, &alias.error),
      &alias, before);
  PREPARE_ALIAS(0xa4);
  expect_alias_rejection(
      et_i64_tensor_copy_from_v1(tensor, alias.values, 32u, &alias.error),
      &alias, before);
  memset(actual, 0, sizeof(actual));
  CHECK(et_i64_tensor_copy_to_v1(tensor, actual, 33u, &error) == 0);
  CHECK(memcmp(actual, expected, sizeof(expected)) == 0);

  memset(&alias, 0, sizeof(alias));
  alias.tensor = tensor;
  memcpy(before, alias.bytes, sizeof(before));
  expect_alias_rejection(
      et_i64_tensor_destroy_v1(&alias.tensor, &alias.error), &alias, before);
  CHECK(alias.tensor == tensor);
  handle = alias.tensor;
  destroy_tensor(&handle);
  tensor = NULL;

#undef PREPARE_ALIAS
}

static void test_borrows_and_view(void) {
  const uint64_t shape[] = {2u, 3u};
  const int64_t values[] = {INT64_MIN, -1, 0, 1, INT64_MAX - 1, INT64_MAX};
  et_i64_tensor_error error;
  et_i64_tensor *tensor = create_tensor(2u, shape);
  et_i64_tensor_borrow *first = NULL;
  et_i64_tensor_borrow *second = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  size_t bytes = 0u;

  CHECK(et_i64_tensor_copy_from_v1(tensor, values, 6u, &error) == 0);
  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &first, &error) == 0);
  CHECK(first != NULL);
  second = first;
  expect_error(et_i64_tensor_borrow_begin_v1(NULL, &second, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(second == first);
  expect_error(et_i64_tensor_borrow_begin_v1(tensor, &second, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(second == first);
  view = (const et_kernel_tensor_view_v1 *)(const void *)first;
  expect_error(et_i64_tensor_borrow_view_v1(NULL, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(view == (const et_kernel_tensor_view_v1 *)(const void *)first);
  view = NULL;
  expect_error(et_i64_tensor_borrow_view_v1(NULL, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(view == NULL);
  CHECK(et_i64_tensor_borrow_view_v1(first, &view, &error) == 0);
  expect_error(et_i64_tensor_borrow_view_v1(first, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(view != NULL);
  CHECK(view->struct_size == ET_KERNEL_TENSOR_VIEW_V1_0_SIZE);
  CHECK(strcmp(view->dtype, "i64") == 0);
  CHECK(strcmp(view->device, "cpu") == 0);
  CHECK(view->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
  CHECK(view->offset_bytes == 0u);
  CHECK(view->rank == 2u);
  CHECK(view->shape != shape);
  CHECK(view->shape[0] == 2u && view->shape[1] == 3u);
  CHECK(view->byte_length == sizeof(values));
  CHECK(view->data != NULL);
  CHECK((uintptr_t)view->data % _Alignof(int64_t) == 0u);
  CHECK(memcmp(view->data, values, sizeof(values)) == 0);

  expect_error(et_i64_tensor_copy_from_v1(tensor, values, 6u, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_STATE,
               ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  expect_values(tensor, values, 6u);
  expect_error(et_i64_tensor_destroy_v1(&tensor, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_STATE,
               ET_I64_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(tensor != NULL);
  CHECK(et_i64_tensor_byte_length_v1(tensor, &bytes, &error) == 0);
  CHECK(bytes == sizeof(values));

  CHECK(et_i64_tensor_borrow_end_v1(&first, &error) == 0);
  CHECK(first == NULL);
  view = (const et_kernel_tensor_view_v1 *)(const void *)tensor;
  expect_error(et_i64_tensor_borrow_view_v1(first, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(view == (const et_kernel_tensor_view_v1 *)(const void *)tensor);
  view = NULL;
  expect_error(et_i64_tensor_borrow_view_v1(first, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(view == NULL);
  CHECK(et_i64_tensor_borrow_end_v1(&first, &error) == 0);
  destroy_tensor(&tensor);

  expect_error(et_i64_tensor_borrow_begin_v1(NULL, &first, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_borrow_begin_v1(tensor, NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  expect_error(et_i64_tensor_borrow_end_v1(NULL, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);

}

static void test_provider_and_dispatch(void) {
  const et_kernel_provider_v1 *provider = et_i64_tensor_provider_v1();
  const uint64_t shape[] = {5u};
  const uint64_t rank_three[] = {1u, 1u, 1u};
  const uint64_t rank_two[] = {1u, 1u};
  const int64_t values[] = {INT64_MIN, -INT64_C(9007199254740993), 0,
                            INT64_C(9007199254740993), INT64_MAX};
  const int64_t sentinels[] = {11, 22, 33, 44, 55};
  _Alignas(int64_t) unsigned char misaligned_data[sizeof(values) + 1u];
  et_i64_tensor *input_tensor = create_tensor(1u, shape);
  et_i64_tensor *output_tensor = create_tensor(1u, shape);
  et_i64_tensor_borrow *input_borrow = NULL;
  et_i64_tensor_borrow *output_borrow = NULL;
  const et_kernel_tensor_view_v1 *input_view = NULL;
  const et_kernel_tensor_view_v1 *output_view = NULL;
  et_kernel_tensor_view_v1 input;
  et_kernel_tensor_view_v1 output;
  et_kernel_request_v1 request = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = "storage.copy",
      .dtype = "i64",
      .device = "cpu",
      .rank = 1u,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  et_kernel_call_v1 call;
  et_kernel_runtime *runtime = NULL;
  const et_kernel_capability_v1 *entry = NULL;
  et_kernel_error kernel_error;
  et_i64_tensor_error error;

  CHECK(provider != NULL);
  CHECK(provider->abi_major == ET_KERNEL_ABI_MAJOR);
  CHECK(provider->abi_minor == ET_KERNEL_ABI_MINOR);
  CHECK(provider->required_features == 0u);
  CHECK(provider->capability_count == 1u);
  CHECK(provider->capability_stride == sizeof(et_kernel_capability_v1));
  CHECK(provider->capability_bytes == sizeof(et_kernel_capability_v1));
  CHECK(provider->capabilities != NULL);
  CHECK(provider->validate_call != NULL && provider->invoke_call != NULL);
  CHECK(et_kernel_runtime_discover(resolve_i1, NULL, &runtime, &kernel_error) ==
        0);
  CHECK(runtime != NULL);
  entry = et_kernel_runtime_capability_find(runtime, "tensor.i64");
  CHECK(entry != NULL);
  CHECK(entry->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(entry->deterministic == 1u);
  CHECK(entry->operation_count == 1u);
  CHECK(strcmp(entry->operations[0], "storage.copy") == 0);
  CHECK(entry->dtype_count == 1u && strcmp(entry->dtypes[0], "i64") == 0);
  CHECK(entry->device_count == 1u && strcmp(entry->devices[0], "cpu") == 0);
  CHECK(entry->shape_range_count == 2u);
  CHECK(entry->shape_ranges[0].rank == 0u &&
        entry->shape_ranges[0].dimensions == NULL);
  CHECK(entry->shape_ranges[1].rank == 1u);
  CHECK(entry->shape_ranges[1].dimensions[0].minimum == 0u);
  CHECK(entry->shape_ranges[1].dimensions[0].maximum ==
        UINT64_C(2305843009213693951));
  CHECK(entry->shape_ranges[1].dimensions[0].maximum ==
        (uint64_t)(SIZE_MAX / sizeof(int64_t)));
  CHECK(et_kernel_runtime_capability_require(runtime, "tensor.i64", &request,
                                             &entry, &kernel_error) == 0);
  CHECK(entry != NULL);

  request.rank = 0u;
  request.shape = NULL;
  CHECK(et_kernel_runtime_capability_require(runtime, "tensor.i64", &request,
                                             NULL, &kernel_error) == 0);
  request.rank = 2u;
  request.shape = rank_two;
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.i64", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);

  request.rank = 3u;
  request.shape = rank_three;
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.i64", &request, &entry, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  CHECK(entry == NULL);
  request.rank = 1u;
  request.shape = shape;
  request.operation = "storage.allocate";
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.i64", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.operation = "storage.copy";
  request.dtype = "f32";
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.i64", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.dtype = "i64";
  request.device = "gpu:0";
  expect_k1_error(et_kernel_runtime_capability_require(
                      runtime, "tensor.i64", &request, NULL, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_UNSUPPORTED,
                  ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  request.device = "cpu";

  CHECK(et_i64_tensor_copy_from_v1(input_tensor, values, 5u, &error) == 0);
  CHECK(et_i64_tensor_copy_from_v1(output_tensor, sentinels, 5u, &error) == 0);
  CHECK(et_i64_tensor_borrow_begin_v1(input_tensor, &input_borrow, &error) ==
        0);
  CHECK(et_i64_tensor_borrow_begin_v1(output_tensor, &output_borrow, &error) ==
        0);
  input_view = NULL;
  output_view = NULL;
  CHECK(et_i64_tensor_borrow_view_v1(input_borrow, &input_view, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(output_borrow, &output_view, &error) == 0);
  memset(&call, 0, sizeof(call));
  call.struct_size = sizeof(call);
  call.capability = "tensor.i64";
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
  expect_values(input_tensor, values, 5u);
  expect_values(output_tensor, values, 5u);

  CHECK(et_i64_tensor_copy_from_v1(output_tensor, sentinels, 5u, &error) != 0);
  expect_values(output_tensor, values, 5u);

  call.input_count = 0u;
  call.input_stride = 0u;
  call.input_bytes = 0u;
  call.inputs = NULL;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER);
  expect_values(output_tensor, values, 5u);
  call.input_count = 1u;
  input = *input_view;
  call.input_stride = sizeof(input);
  call.input_bytes = sizeof(input);
  call.inputs = &input;

  memcpy(misaligned_data + 1u, values, sizeof(values));
  input.data = misaligned_data + 1u;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_INVALID_BUFFER);
  expect_values(output_tensor, values, 5u);
  input = *input_view;

  CHECK(et_i64_tensor_borrow_end_v1(&input_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_end_v1(&output_borrow, &error) == 0);
  CHECK(et_i64_tensor_copy_from_v1(output_tensor, sentinels, 5u, &error) == 0);
  CHECK(et_i64_tensor_borrow_begin_v1(input_tensor, &input_borrow, &error) ==
        0);
  CHECK(et_i64_tensor_borrow_begin_v1(output_tensor, &output_borrow, &error) ==
        0);
  input_view = NULL;
  output_view = NULL;
  CHECK(et_i64_tensor_borrow_view_v1(input_borrow, &input_view, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(output_borrow, &output_view, &error) == 0);
  input = *input_view;
  output = *output_view;
  output.shape = rank_three;
  output.rank = 3u;
  call.inputs = &input;
  call.outputs = &output;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_SHAPE_MISMATCH,
                  ET_KERNEL_CODE_INVALID_BUFFER);
  expect_values(output_tensor, sentinels, 5u);

  output = *output_view;
  output.data = input.data;
  expect_k1_error(et_kernel_runtime_dispatch(runtime, &call, &kernel_error),
                  &kernel_error, ET_KERNEL_ERROR_INVALID_ARGUMENT,
                  ET_KERNEL_CODE_ALIASING_OUTPUT);
  expect_values(input_tensor, values, 5u);
  expect_values(output_tensor, sentinels, 5u);

  CHECK(et_i64_tensor_borrow_end_v1(&input_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_end_v1(&output_borrow, &error) == 0);
  et_kernel_runtime_destroy(runtime);
  expect_values(output_tensor, sentinels, 5u);
  destroy_tensor(&input_tensor);
  destroy_tensor(&output_tensor);
}

#ifdef ET_I64_TENSOR_TESTING
static void check_rank9_state(const et_i64_tensor *tensor,
                              const int64_t *expected) {
  et_i64_tensor_error error;
  size_t rank = 0u;
  size_t count = 0u;
  size_t bytes = 0u;
  uint64_t extent = 0u;
  size_t stride = 0u;

  CHECK(et_i64_tensor_rank_v1(tensor, &rank, &error) == 0);
  CHECK(rank == 1u);
  CHECK(et_i64_tensor_shape_at_v1(tensor, 0u, &extent, &error) == 0);
  CHECK(extent == 9u);
  CHECK(et_i64_tensor_stride_bytes_at_v1(tensor, 0u, &stride, &error) == 0);
  CHECK(stride == sizeof(int64_t));
  CHECK(et_i64_tensor_element_count_v1(tensor, &count, &error) == 0);
  CHECK(count == 9u);
  CHECK(et_i64_tensor_byte_length_v1(tensor, &bytes, &error) == 0);
  CHECK(bytes == 9u * sizeof(int64_t));
  expect_values(tensor, expected, 9u);
}

static void test_owned_storage_alias_rejection(void) {
  const uint64_t shape[] = {9u};
  const int64_t values[] = {INT64_MIN, -7, -1, 0, 1,
                            7,         9,  17, INT64_MAX};
  const int64_t other_values[] = {0, -19, -11, -3, 2, 8, 14, 20, 26};
  et_i64_tensor_error error;
  et_i64_tensor *tensor = create_tensor(1u, shape);
  et_i64_tensor *other = create_tensor(1u, shape);
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_borrow *other_borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  const et_kernel_tensor_view_v1 *other_view = NULL;
  const void *local_regions[6];
  const void *other_regions[6];
  size_t scalar;
  unsigned char zero_slot[sizeof(void *)] = {0};

  CHECK(et_i64_tensor_copy_from_v1(tensor, values, 9u, &error) == 0);
  CHECK(et_i64_tensor_copy_from_v1(other, other_values, 9u, &error) == 0);
  CHECK(et_i64_tensor_test_tensor_control_bytes_v1() != 0u);
  CHECK(et_i64_tensor_test_metadata_bytes_v1(tensor) == sizeof(uint64_t));
  CHECK(et_i64_tensor_test_borrow_bytes_v1() != 0u);
  local_regions[0] = tensor;
  local_regions[1] = et_i64_tensor_test_shape_storage_v1(tensor);
  local_regions[2] = et_i64_tensor_test_stride_storage_v1(tensor);
  local_regions[3] = et_i64_tensor_test_data_storage_v1(tensor);

  for (size_t index = 0u; index < 4u; index++) {
    CHECK(local_regions[index] != NULL);
    expect_error(et_i64_tensor_copy_to_v1(
                     tensor, (int64_t *)(void *)local_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    expect_error(et_i64_tensor_copy_from_v1(
                     tensor, (const int64_t *)local_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
  }

  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  local_regions[4] = borrow;
  local_regions[5] = view;

  /* With a live lease, copy validation still rejects every owned region first. */
  for (size_t index = 0u; index < 6u; index++) {
    expect_error(et_i64_tensor_copy_to_v1(
                     tensor, (int64_t *)(void *)local_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    expect_error(et_i64_tensor_copy_from_v1(
                     tensor, (const int64_t *)local_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    view = NULL;
    CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
    CHECK(view->rank == 1u && view->shape[0] == 9u);
    CHECK(view->byte_length == sizeof(values));
  }

  /* An error record spanning any same-owner allocation is rejected unwritten. */
  for (size_t index = 0u; index < 6u; index++) {
    scalar = 0x5a5a5a5au;
    CHECK(et_i64_tensor_rank_v1(
              tensor, &scalar,
              (et_i64_tensor_error *)(void *)local_regions[index]) ==
          ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
    CHECK(scalar == 0x5a5a5a5au);
    check_rank9_state(tensor, values);
    view = NULL;
    CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
    et_i64_tensor_error_clear_v1(
        (et_i64_tensor_error *)(void *)local_regions[index]);
    check_rank9_state(tensor, values);
    view = NULL;
    CHECK(et_i64_tensor_borrow_view_v1(borrow, &view, &error) == 0);
  }

  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
  check_rank9_state(tensor, values);

  /* The same exclusions are process-wide, not limited to the receiver. */
  other_regions[0] = other;
  other_regions[1] = et_i64_tensor_test_shape_storage_v1(other);
  other_regions[2] = et_i64_tensor_test_stride_storage_v1(other);
  other_regions[3] = et_i64_tensor_test_data_storage_v1(other);
  CHECK(et_i64_tensor_borrow_begin_v1(other, &other_borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_view_v1(other_borrow, &other_view, &error) == 0);
  other_regions[4] = other_borrow;
  other_regions[5] = other_view;
  for (size_t index = 0u; index < 6u; index++) {
    expect_error(et_i64_tensor_copy_to_v1(
                     tensor, (int64_t *)(void *)other_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    check_rank9_state(other, other_values);
    expect_error(et_i64_tensor_copy_from_v1(
                     tensor, (const int64_t *)other_regions[index], 9u,
                     &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    check_rank9_state(other, other_values);

    scalar = 0x6b6b6b6bu;
    CHECK(et_i64_tensor_rank_v1(
              tensor, &scalar,
              (et_i64_tensor_error *)(void *)other_regions[index]) ==
          ET_I64_TENSOR_ERROR_INVALID_ARGUMENT);
    CHECK(scalar == 0x6b6b6b6bu);
    check_rank9_state(tensor, values);
    check_rank9_state(other, other_values);
  }

  /* Writable scalar outputs cannot target another live I1 allocation. */
  for (size_t index = 0u; index < 6u; index++) {
    expect_error(et_i64_tensor_rank_v1(
                     tensor, (size_t *)(void *)other_regions[index], &error),
                 &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
                 ET_I64_TENSOR_CODE_INVALID_BUFFER);
    check_rank9_state(tensor, values);
    check_rank9_state(other, other_values);
  }

  CHECK(et_i64_tensor_borrow_end_v1(&other_borrow, &error) == 0);
  CHECK(other_borrow == NULL);
  check_rank9_state(other, other_values);

  /* A zero pointer slot inside another tensor's data must not be published. */
  CHECK(memcmp(et_i64_tensor_test_data_storage_v1(other), zero_slot,
               sizeof(zero_slot)) == 0);
  expect_error(et_i64_tensor_create_v1(
                   0u, NULL,
                   (et_i64_tensor **)(void *)
                       et_i64_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  check_rank9_state(other, other_values);
  expect_error(et_i64_tensor_borrow_begin_v1(
                   tensor,
                   (et_i64_tensor_borrow **)(void *)
                       et_i64_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  check_rank9_state(tensor, values);
  check_rank9_state(other, other_values);

  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  expect_error(et_i64_tensor_borrow_view_v1(
                   borrow,
                   (const et_kernel_tensor_view_v1 **)(void *)
                       et_i64_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  check_rank9_state(tensor, values);
  check_rank9_state(other, other_values);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);

  expect_error(et_i64_tensor_destroy_v1(
                   (et_i64_tensor **)(void *)
                       et_i64_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  expect_error(et_i64_tensor_borrow_end_v1(
                   (et_i64_tensor_borrow **)(void *)
                       et_i64_tensor_test_data_storage_v1(other),
                   &error),
               &error, ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  check_rank9_state(tensor, values);
  check_rank9_state(other, other_values);

  destroy_tensor(&tensor);
  destroy_tensor(&other);
}

static void test_borrow_tracking_failures(void) {
  const uint64_t shape[] = {1u};
  et_i64_tensor *owner = create_tensor(1u, shape);
  et_i64_tensor *other = create_tensor(1u, shape);
  et_i64_tensor_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *view = NULL;
  et_i64_tensor_error error;

  CHECK(et_i64_tensor_borrow_begin_v1(owner, &borrow, &error) == 0);
  CHECK(borrow != NULL);

  et_i64_tensor_test_set_borrow_owner_v1(borrow, other);
  view = (const et_kernel_tensor_view_v1 *)(const void *)owner;
  expect_error(et_i64_tensor_borrow_view_v1(borrow, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(view == (const et_kernel_tensor_view_v1 *)(const void *)owner);
  view = NULL;
  expect_error(et_i64_tensor_borrow_view_v1(borrow, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_STATE,
               ET_I64_TENSOR_CODE_INVALID_HANDLE);
  CHECK(view == NULL);
  et_i64_tensor_test_set_borrow_owner_v1(borrow, owner);

  et_i64_tensor_test_set_active_borrow_v1(owner, NULL);
  view = NULL;
  expect_error(et_i64_tensor_borrow_view_v1(borrow, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_STATE,
               ET_I64_TENSOR_CODE_INVALID_HANDLE);
  CHECK(view == NULL);
  expect_error(et_i64_tensor_borrow_end_v1(&borrow, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_STATE,
               ET_I64_TENSOR_CODE_INVALID_HANDLE);
  CHECK(borrow != NULL);
  et_i64_tensor_test_set_active_borrow_v1(owner, borrow);

  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  CHECK(borrow == NULL);
  view = NULL;
  expect_error(et_i64_tensor_borrow_view_v1(borrow, &view, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_NULL_ARGUMENT);
  CHECK(view == NULL);

  destroy_tensor(&owner);
  destroy_tensor(&other);
}

static void test_allocation_failpoints(void) {
  const uint64_t nonempty_shape[] = {2u, 2u};
  const uint64_t empty_shape[] = {2u, 0u};
  et_i64_tensor_error error;
  et_i64_tensor *tensor = NULL;
  et_i64_tensor *saved = NULL;
  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_borrow *saved_borrow = NULL;

  for (size_t allowed = 0u; allowed < 4u; allowed++) {
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    expect_error(et_i64_tensor_create_v1(2u, nonempty_shape, &tensor, &error),
                 &error, ET_I64_TENSOR_ERROR_INTERNAL,
                 ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(tensor == NULL);
  }
  et_i64_tensor_test_reset_allocator_v1();
  tensor = create_tensor(2u, nonempty_shape);
  destroy_tensor(&tensor);

  for (size_t allowed = 0u; allowed < 3u; allowed++) {
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    expect_error(et_i64_tensor_create_v1(2u, empty_shape, &tensor, &error),
                 &error, ET_I64_TENSOR_ERROR_INTERNAL,
                 ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(tensor == NULL);
  }
  et_i64_tensor_test_reset_allocator_v1();

  for (size_t allowed = 0u; allowed < 2u; allowed++) {
    et_i64_tensor_test_fail_alloc_after_v1(allowed);
    expect_error(et_i64_tensor_create_v1(0u, NULL, &tensor, &error), &error,
                 ET_I64_TENSOR_ERROR_INTERNAL,
                 ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
    CHECK(tensor == NULL);
  }
  et_i64_tensor_test_reset_allocator_v1();

  tensor = create_tensor(0u, NULL);
  saved = tensor;
  et_i64_tensor_test_fail_alloc_after_v1(0u);
  expect_error(et_i64_tensor_create_v1(0u, NULL, &tensor, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(tensor == saved);
  et_i64_tensor_test_reset_allocator_v1();

  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  saved_borrow = borrow;
  et_i64_tensor_test_fail_alloc_after_v1(0u);
  expect_error(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error), &error,
               ET_I64_TENSOR_ERROR_INVALID_ARGUMENT,
               ET_I64_TENSOR_CODE_INVALID_BUFFER);
  CHECK(borrow == saved_borrow);
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);

  et_i64_tensor_test_fail_alloc_after_v1(0u);
  expect_error(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error), &error,
               ET_I64_TENSOR_ERROR_INTERNAL,
               ET_I64_TENSOR_CODE_ALLOCATION_FAILED);
  CHECK(borrow == NULL);
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(et_i64_tensor_borrow_begin_v1(tensor, &borrow, &error) == 0);
  CHECK(et_i64_tensor_borrow_end_v1(&borrow, &error) == 0);
  destroy_tensor(&tensor);
}
#endif

int main(void) {
  test_version_and_nulls();
  test_shapes_and_strides();
  test_shape_failures();
  test_invalid_rank_shape_error_alias_atomicity();
  test_exact_round_trip_and_atomicity();
  test_error_alias_atomicity();
  test_borrows_and_view();
  test_provider_and_dispatch();
#ifdef ET_I64_TENSOR_TESTING
  test_owned_storage_alias_rejection();
  test_borrow_tracking_failures();
  test_allocation_failpoints();
#endif
  if (failures != 0) {
    fprintf(stderr, "I1 FAIL: %d of %d checks failed\n", failures, checks);
    return 1;
  }
  printf("I1 PASS: %d exact-i64 boundary, ownership, and K1 checks\n", checks);
  return 0;
}
