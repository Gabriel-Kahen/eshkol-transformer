#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_P2_G1_COPYOUT_PRIVATE 1
#define ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN \
  et_g3c4_p2g1_publication_predecessor_main
#include "test_p2_g1_publication.c"
#undef ET_G3C4_P2_G1_PUBLICATION_TEST_MAIN

typedef struct copyout_carrier {
  int64_t length;
  unsigned char bytes[56];
} copyout_carrier;

static int64_t copied_word(const copyout_carrier *carrier, size_t index) {
  uint64_t word = 0u;
  for (size_t byte = 0u; byte < 8u; byte++)
    word |= (uint64_t)carrier->bytes[index * 8u + byte] << (8u * byte);
  return (int64_t)word;
}

static void unchanged(const copyout_carrier *carrier) {
  for (size_t index = 0u; index < sizeof(carrier->bytes); index++)
    CHECK(carrier->bytes[index] == 0xa5u);
}

static void run_copyout(et_g3c4_model_owner_internal *owner,
                        int categorical) {
  publication_fixture fixture = publication_ready(owner, categorical);
  et_g3c4_context_internal *context = fixture.context;
  et_g3c4_output_internal *output = fixture.output;
  raw_carrier raw = {1, {(unsigned char)fixture.token}};
  copyout_carrier destination = {56, {0}};
  memset(destination.bytes, 0xa5, sizeof(destination.bytes));
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) == ET_G3C4_INVALID_STATE);
  unchanged(&destination);
  OK(et_g3c4_private_generation_frame_prepare_v1(context, output));
  OK(et_g3c4_private_call_prepare_end_v1(context));
  OK(et_g3c4_private_generation_frame_commit_v1(context, output));
  OK(et_g3c4_private_call_finish_v1(context));
  OK(et_g3c4_private_tensor_release_v1(fixture.input));
  OK(et_g3c4_private_generator_close_v1(context));

  OK(et_g3c4_private_output_copy_snapshot_v1(output, &raw, &destination));
  CHECK(copied_word(&destination, 0u) == fixture.token);
  CHECK(copied_word(&destination, 1u) == 1);
  CHECK(copied_word(&destination, 2u) == 3);
  for (size_t index = 0u; index < 4u; index++)
    CHECK(copied_word(&destination, index + 3u) == fixture.successor[index]);
  copyout_carrier first = destination;
  memset(destination.bytes, 0xa5, sizeof(destination.bytes));

  raw.bytes[0] ^= 1u;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) == ET_G3C4_INVALID_ARGUMENT);
  unchanged(&destination);
  raw.bytes[0] ^= 1u;
  raw.length = 2;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) == ET_G3C4_SHAPE_MISMATCH);
  unchanged(&destination);
  raw.length = 1;
  destination.length = 55;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) == ET_G3C4_SHAPE_MISMATCH);
  unchanged(&destination);
  destination.length = 56;

  et_i64_tensor_borrow *borrow = NULL;
  et_i64_tensor_error ids_error;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &borrow, &ids_error));
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) != 0);
  unchanged(&destination);
  OK(et_i64_tensor_borrow_end_v1(&borrow, &ids_error));
  et_i64_tensor_test_fail_alloc_after_v1(0u);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) != 0);
  unchanged(&destination);
  et_i64_tensor_test_reset_allocator_v1();
  int64_t *id = (int64_t *)et_i64_tensor_test_data_storage_v1(output->ids);
  int64_t original = *id;
  for (size_t index = 0u; index < 2u; index++) {
    *id = index == 0u ? -1 : 256;
    CHECK(et_g3c4_private_output_copy_snapshot_v1(
              output, &raw, &destination) == ET_G3C4_INTERNAL);
    unchanged(&destination);
  }
  *id = original;
  uint64_t *shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      output->ids);
  size_t *stride = (size_t *)et_i64_tensor_test_stride_storage_v1(
      output->ids);
  shape[0] = 2u;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) != 0);
  unchanged(&destination);
  shape[0] = 1u;
  stride[0] = 16u;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) != 0);
  unchanged(&destination);
  stride[0] = sizeof(int64_t);
  et_i64_tensor *owned_ids = output->ids;
  output->ids = (et_i64_tensor *)(uintptr_t)0x1000u;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) != 0);
  unchanged(&destination);
  output->ids = owned_ids;
  for (int mode = 1; mode <= 3; mode++) {
    malformed_output_view = mode;
    CHECK(et_g3c4_private_output_copy_snapshot_v1(
              output, &raw, &destination) != 0);
    unchanged(&destination);
  }
  malformed_output_view = 0;

  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &raw) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, output->ids) == ET_G3C4_INVALID_ARGUMENT);
  et_f32_tensor *model_value = owner->parameters[0]->value;
  CHECK(model_value->byte_length >= sizeof(destination));
  et_f32_tensor_borrow *model_borrow = NULL;
  et_f32_tensor_error f32_error;
  OK(et_f32_tensor_borrow_begin_v1(
      model_value, &model_borrow, &f32_error));
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, model_value->data) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, model_value->data, &destination) ==
        ET_G3C4_INVALID_ARGUMENT);
  OK(et_f32_tensor_borrow_end_v1(&model_borrow, &f32_error));
  unchanged(&destination);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, NULL, &destination) ==
        ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, (void *)(uintptr_t)UINTPTR_MAX) ==
        ET_G3C4_INVALID_ARGUMENT);
  unchanged(&destination);
  int foreign = 0;
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            &foreign, &raw, &destination) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            context, &raw, &destination) == ET_G3C4_INVALID_ARGUMENT);
  unchanged(&destination);
  CHECK(memcmp(output->rng, fixture.successor,
               sizeof(fixture.successor)) == 0);

  OK(et_g3c4_private_output_copy_snapshot_v1(output, &raw, &destination));
  CHECK(memcmp(&first, &destination, sizeof(first)) == 0);
  OK(et_g3c4_private_output_release_v1(output));
  memset(destination.bytes, 0xa5, sizeof(destination.bytes));
  CHECK(et_g3c4_private_output_copy_snapshot_v1(
            output, &raw, &destination) == ET_G3C4_INVALID_STATE);
  unchanged(&destination);
  CHECK(copied_word(&first, 0u) == fixture.token);
}

int main(void) {
  CHECK(et_g3c4_p2g1_pending_predecessor_main() == 0);
  et_g3c4_model_owner_internal *owner = p2g1_test_owner;
  run_copyout(owner, 0);
  run_copyout(owner, 1);
  printf("G3-C4 P2/G1 private copy-out PASS: checks=%zu\n", checks);
  return 0;
}
