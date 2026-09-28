/* Focused native admission/bit/atomicity probe. Source inclusion exposes only
 * test fixture construction; the production snapshot still authenticates. */
#define ET_G3T_PREFILL_SAMPLE_PRIVATE
#define ET_G3T_MANUAL_LOGITS_PRIVATE
#define ET_G3T_MANUAL_LOGITS_MATERIALIZE_PRIVATE
#include "src/eshkol_transformer/g3t_transport.c"

#include <stdio.h>

static int checks;
#define CHECK(condition) do { \
  if (!(condition)) { fprintf(stderr, "snapshot check failed: %s\n", #condition); return 1; } \
  ++checks; \
} while (0)

typedef struct snapshot_bytes {
  int64_t length;
  uint32_t words[256];
} snapshot_bytes;

static int sentinel(const snapshot_bytes *out) {
  for (size_t i = 0; i < 256; ++i)
    if (out->words[i] != UINT32_C(0x5a5a5a5a)) return 0;
  return 1;
}
static void reset(snapshot_bytes *out) {
  out->length = 1024;
  memset(out->words, 0x5a, sizeof(out->words));
}

int main(void) {
  static const uint64_t shape[2] = {1, 256};
  g3t_logits fixture = {0};
  et_f32_tensor_error error;
  uint32_t expected[256];
  for (size_t i = 0; i < 256; ++i) expected[i] = UINT32_C(0x3f000000) + (uint32_t)i;
  expected[0] = UINT32_C(0x80000000);
  expected[1] = UINT32_C(0x7fc01234);
  CHECK(et_f32_tensor_create_v1(2, shape, &fixture.tensor, &error) == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(fixture.tensor, expected, 256, &error) == 0);
  fixture.h.kind = G3T_LOGITS;
  fixture.h.state = G3T_LIVE;
  g3t_registry = &fixture.h;
  snapshot_bytes out;
  reset(&out);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) == 0);
  CHECK(memcmp(out.words, expected, sizeof(expected)) == 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) == 0);
  CHECK(memcmp(out.words, expected, sizeof(expected)) == 0);

  reset(&out);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1023) != 0);
  CHECK(sentinel(&out));
  out.length = 1023;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  reset(&out);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, NULL, 1024) != 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(
            &fixture, (void *)(UINTPTR_MAX - (uintptr_t)4u), 1024) != 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&out, &out, 1024) != 0);
  CHECK(sentinel(&out));

  uint32_t alias_header[256] = {1024, 0};
  CHECK(et_f32_tensor_copy_bits_from_v1(fixture.tensor, alias_header, 256, &error) == 0);
  void *alias = (void *)et_f32_tensor_test_data_storage_v1(fixture.tensor);
  CHECK(alias != NULL);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, alias, 1024) != 0);
  CHECK(et_f32_tensor_copy_bits_to_v1(fixture.tensor, alias_header, 256, &error) == 0);
  CHECK(alias_header[0] == 1024 && alias_header[1] == 0);
  CHECK(et_f32_tensor_copy_bits_from_v1(fixture.tensor, expected, 256, &error) == 0);

  fixture.h.kind = G3T_INPUT;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  fixture.h.kind = G3T_LOGITS;
  fixture.h.state = G3T_PENDING;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  fixture.h.state = G3T_LIVE;
  fixture.h.busy = 1;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  fixture.h.busy = 0;
  fixture.parent_ctx = (g3t_context *)&out;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  fixture.parent_ctx = NULL;

  et_f32_tensor_borrow *borrow = NULL;
  CHECK(et_f32_tensor_borrow_begin_v1(fixture.tensor, &borrow, &error) == 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(et_g3t_private_last_error_domain_v1() == 3);
  CHECK(et_g3t_private_last_error_code_v1() == ET_F32_TENSOR_CODE_ACTIVE_BORROW);
  CHECK(sentinel(&out));
  CHECK(et_f32_tensor_borrow_end_v1(&borrow, &error) == 0);

  uint64_t *mutable_shape = (uint64_t *)et_f32_tensor_test_shape_storage_v1(fixture.tensor);
  CHECK(mutable_shape != NULL);
  mutable_shape[1] = 255;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(et_g3t_private_last_error_category_v1() == G3T_INTERNAL);
  CHECK(et_g3t_private_last_error_code_v1() == G3T_INVARIANT);
  CHECK(sentinel(&out));
  mutable_shape[1] = 256;
  size_t *mutable_stride = (size_t *)et_f32_tensor_test_stride_storage_v1(fixture.tensor);
  CHECK(mutable_stride != NULL);
  mutable_stride[1] = 8;
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(et_g3t_private_last_error_category_v1() == G3T_INTERNAL);
  CHECK(et_g3t_private_last_error_code_v1() == G3T_INVARIANT);
  CHECK(sentinel(&out));
  mutable_stride[1] = 4;

  et_f32_test_tensor_metadata_v1 snapshot = {
      .struct_size = sizeof(snapshot)};
  CHECK(et_f32_tensor_test_metadata_snapshot_v1(
            fixture.tensor, &snapshot) == 0);
  reset(&out);
  CHECK(et_f32_tensor_test_metadata_corrupt_v1(
            fixture.tensor, ET_F32_TEST_TENSOR_METADATA_DATA, 0) == 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(et_g3t_private_last_error_category_v1() == G3T_INTERNAL);
  CHECK(et_g3t_private_last_error_code_v1() == G3T_INVARIANT);
  CHECK(sentinel(&out));
  CHECK(et_f32_tensor_test_metadata_restore_v1(
            fixture.tensor, &snapshot) == 0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) == 0);
  CHECK(memcmp(out.words, expected, sizeof(expected)) == 0);

  reset(&out);
  et_f32_tensor_test_fail_alloc_after_v1(0);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  et_f32_tensor_test_reset_allocator_v1();
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) == 0);
  CHECK(memcmp(out.words, expected, sizeof(expected)) == 0);
  CHECK(et_g3t_private_tensor_release_v1(&fixture) == 0);
  reset(&out);
  CHECK(et_g3t_private_logits_copy_bits_v1(&fixture, &out, 1024) != 0);
  CHECK(sentinel(&out));
  printf("G3-T manual logits materialization PASS: checks=%d\n", checks);
  return 0;
}
