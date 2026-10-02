#define ET_G3C4_P2_G2_TERMINAL_IDS_CLONE_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN terminal_clone_predecessor_main
#include "test_p2_g2_native_terminal.c"

typedef struct clone_control {
  et_g3c4_transport_header_internal transport;
  et_i64_tensor *ids;
  int64_t logical_length;
} clone_control;

typedef struct clone_bytes {
  int64_t length;
  unsigned char payload[16];
  unsigned char guard[8];
} clone_bytes;

static int64_t cloned_word(const clone_bytes *copy, size_t index) {
  uint64_t value = 0u;
  for (size_t byte = 0u; byte < 8u; byte++)
    value |= (uint64_t)copy->payload[index * 8u + byte] << (8u * byte);
  return (int64_t)value;
}

static size_t clone_live_count(void) {
  size_t count = 0u;
  const et_g3c4_transport_header_internal *record;
  for (record = et_g3c4_transport_registry; record != NULL;
       record = record->registry_next)
    if (record->kind == ET_G3C4_TERMINAL_IDS_KIND &&
        record->state == ET_G3C4_CONTEXT_LIVE)
      count++;
  return count;
}

static void clone_alias_walkers(et_g3c4_model_owner_internal *owner,
                                clone_control *control, int dead) {
  et_g3c4_context_internal *context = create_generator(owner);
  const void *tail = &control->logical_length;
  CHECK(et_g3c4_output_decode_owned_alias(context, tail, 8u) != 0);
  CHECK(et_g3c4_p2g2_carrier_owned_alias(context, tail, 8u) != 0);
  CHECK(et_g3c4_terminal_snapshot_alias(tail, 8u) != 0);
  if (!dead) {
    const void *data = et_i64_tensor_test_data_storage_v1(control->ids);
    const void *shape = et_i64_tensor_test_shape_storage_v1(control->ids);
    const void *stride = et_i64_tensor_test_stride_storage_v1(control->ids);
    CHECK(data != NULL);
    CHECK(shape != NULL && stride != NULL);
    CHECK(et_g3c4_output_decode_owned_alias(context, data, 8u) != 0);
    CHECK(et_g3c4_p2g2_carrier_owned_alias(context, data, 8u) != 0);
    CHECK(et_g3c4_terminal_snapshot_alias(data, 8u) != 0);
    CHECK(et_g3c4_terminal_snapshot_alias(control->ids, 8u) != 0);
    CHECK(et_g3c4_terminal_snapshot_alias(shape, 8u) != 0);
    CHECK(et_g3c4_terminal_snapshot_alias(stride, sizeof(size_t)) != 0);
  }
  CHECK(et_g3c4_output_decode_owned_alias(context, owner, 8u) != 0);
  CHECK(et_g3c4_p2g2_carrier_owned_alias(context, owner, 8u) != 0);
  CHECK(et_g3c4_terminal_snapshot_alias(owner, 8u) != 0);
  CHECK(et_g3c4_output_decode_owned_alias(
            context, owner->parameters[0]->value, 8u) != 0);
  CHECK(et_g3c4_p2g2_carrier_owned_alias(
            context, owner->parameters[0]->value, 8u) != 0);
  CHECK(et_g3c4_terminal_snapshot_alias(
            owner->parameters[0]->value, 8u) != 0);
  CHECK(et_g3c4_output_decode_owned_alias(
            context, owner->parameters[0]->value->data, 8u) != 0);
  CHECK(et_g3c4_p2g2_carrier_owned_alias(
            context, owner->parameters[0]->value->data, 8u) != 0);
  CHECK(et_g3c4_terminal_snapshot_alias(
            owner->parameters[0]->value->data, 8u) != 0);
  CHECK(et_g3c4_output_decode_owned_alias(context, context->cache, 8u) != 0);
  CHECK(et_g3c4_p2g2_carrier_owned_alias(context, context->cache, 8u) != 0);
  CHECK(et_g3c4_terminal_snapshot_alias(context->cache, 8u) != 0);
  if (!dead)
    CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
        control, control->logical_length, context->cache,
        8 + 8 * control->logical_length) != 0);
  OK(et_g3c4_private_generator_close_v1(context));
}

static et_g3c4_output_internal *published_terminal(
    et_g3c4_model_owner_internal *owner, int first_eos,
    int64_t expected[2]) {
  const int64_t prompt[2] = {7, 11};
  et_g3c4_context_internal *context;
  et_g3c4_input_internal *input;
  et_g3c4_output_internal *output;
  terminal_raw first, second;
  terminal_ready(owner, prompt, first_eos, &context, &input, &output,
                 &first, &second);
  if (expected != NULL) {
    expected[0] = first.byte;
    expected[1] = first_eos ? 0 : second.byte;
  }
  OK(et_g3c4_private_p2g2_terminal_commit_v1(
      context, output, &first, 9, first_eos ? NULL : &second,
      first_eos ? 0 : 9));
  OK(et_g3c4_private_tensor_release_v1(input));
  OK(et_g3c4_private_generator_close_v1(context));
  return output;
}

static void clone_case(et_g3c4_model_owner_internal *owner, int first_eos) {
  int64_t expected[2] = {-1, -1};
  et_g3c4_output_internal *output =
      published_terminal(owner, first_eos, expected);
  int64_t count = first_eos ? 1 : 2;
  terminal_snapshot source = {0};
  source.length = 64;
  OK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &source, sizeof(source)));
  CHECK(word(&source, 2) == count);
  CHECK(word(&source, 0) == expected[0]);
  CHECK(word(&source, 1) == expected[1]);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(NULL) == NULL);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(
      (unsigned char *)output + 8u) == NULL);
  void *first = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
  void *second = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
  CHECK(first != NULL && second != NULL && first != second);
  clone_control *control = first;
  CHECK(control->transport.kind == 9u &&
        control->transport.magic == UINT64_C(0x4733433454494431) &&
        control->logical_length == count && control->ids != NULL);
  uint64_t shape = 0u;
  et_i64_tensor_error error;
  OK(et_i64_tensor_shape_at_v1(control->ids, 0u, &shape, &error));
  CHECK(shape == (uint64_t)count);
  clone_alias_walkers(owner, control, 0);

  clone_bytes copied;
  memset(&copied, 0xa5, sizeof(copied));
  copied.length = 8 * count;
  clone_bytes unchanged = copied;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count - 1) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, 3 - count, &copied, 8 + 8 * count) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, (unsigned char *)&copied + 1u, 8 + 8 * count) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &control->logical_length, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, owner, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, owner->parameters[0]->value, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, control->ids, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_snapshot_v1(
      output, &control->logical_length, 72) != 0);
  CHECK(control->logical_length == count);
  copied.length--;
  unchanged = copied;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  copied.length++;

  et_i64_tensor_borrow *held = NULL;
  OK(et_i64_tensor_borrow_begin_v1(control->ids, &held, &error));
  unchanged = copied;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  OK(et_i64_tensor_borrow_end_v1(&held, &error));

  if (count == 1) {
    malformed_output_view = 1;
    unchanged = copied;
    CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
        first, count, &copied, 8 + 8 * count) != 0);
    CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first) != 0);
    CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
    malformed_output_view = 0;
  }
  int64_t *clone_data = (int64_t *)et_i64_tensor_test_data_storage_v1(
      control->ids);
  uint64_t *clone_shape = (uint64_t *)et_i64_tensor_test_shape_storage_v1(
      control->ids);
  CHECK(clone_data != NULL && clone_shape != NULL);
  clone_data[0] = 256;
  unchanged = copied;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  clone_data[0] = expected[0];
  clone_shape[0] = 3u;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first) != 0);
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  clone_shape[0] = (uint64_t)count;

  et_i64_tensor_test_fail_alloc_after_v1(0u);
  unchanged = copied;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  et_i64_tensor_test_reset_allocator_v1();
  CHECK(memcmp(&copied, &unchanged, sizeof(copied)) == 0);
  OK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count));
  CHECK(cloned_word(&copied, 0u) == expected[0]);
  CHECK(cloned_word(&copied, 0u) == word(&source, 0u));
  if (count == 2) {
    CHECK(cloned_word(&copied, 1u) == expected[1]);
    CHECK(cloned_word(&copied, 1u) == word(&source, 1u));
  }
  else CHECK(copied.payload[8] == 0xa5u);
  for (size_t index = 0; index < sizeof(copied.guard); index++)
    CHECK(copied.guard[index] == 0xa5u);

  OK(et_g3c4_private_output_release_v1(output));
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  clone_bytes after_source = {0};
  after_source.length = 8 * count;
  OK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      second, count, &after_source, 8 + 8 * count));
  CHECK(cloned_word(&after_source, 0u) == word(&source, 0u));
  if (count == 2)
    CHECK(cloned_word(&after_source, 1u) == word(&source, 1u));
  OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first));
  OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(first));
  clone_alias_walkers(owner, control, 1);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      first, count, &copied, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_copy_v1(
      second, count, &control->logical_length, 8 + 8 * count) != 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(output) != 0);
  OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(second));
}

static void clone_failure_matrix(et_g3c4_model_owner_internal *owner) {
  et_g3c4_output_internal *output = published_terminal(owner, 1, NULL);
  const int64_t *storage = et_i64_tensor_test_data_storage_v1(output->ids);
  CHECK(storage != NULL);
  int64_t source_ids[2];
  memcpy(source_ids, storage, sizeof(source_ids));
  et_i64_tensor_error error;
  et_i64_tensor_borrow *held = NULL;
  OK(et_i64_tensor_borrow_begin_v1(output->ids, &held, &error));
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  OK(et_i64_tensor_borrow_end_v1(&held, &error));
  size_t live_before = clone_live_count();
  et_g3c4_context_allocation_limit =
      et_g3c4_context_successful_allocations;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  et_g3c4_context_allocation_limit = SIZE_MAX;
  CHECK(clone_live_count() == live_before);
  fail_terminal_copy = 1;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  CHECK(et_g3c4_private_last_error_domain_v1() == ET_G3C4_DOMAIN_I1);
  CHECK(et_g3c4_private_last_error_category_v1() ==
        ET_I64_TENSOR_ERROR_INTERNAL);
  CHECK(et_g3c4_private_last_error_code_v1() ==
        ET_I64_TENSOR_CODE_PROVIDER_REJECTED);
  fail_terminal_copy = 0;
  CHECK(clone_live_count() == live_before);
  void *retry = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
  CHECK(retry != NULL);
  OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(retry));
  tamper_clone_source_on_copy = (int64_t *)storage;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  CHECK(tamper_clone_source_on_copy == NULL);
  CHECK(clone_live_count() == live_before);
  ((int64_t *)storage)[0] = source_ids[0];
  int failures = 0;
  for (size_t cut = 0u; cut < 9u; cut++) {
    et_i64_tensor_test_fail_alloc_after_v1(cut);
    void *clone = et_g3c4_private_p2g2_terminal_ids_clone_v1(output);
    et_i64_tensor_test_reset_allocator_v1();
    if (clone == NULL) failures++;
    else OK(et_g3c4_private_p2g2_terminal_ids_clone_release_v1(clone));
    CHECK(clone_live_count() == live_before);
    CHECK(memcmp(storage, source_ids, sizeof(source_ids)) == 0);
  }
  CHECK(failures >= 3);
  int64_t *mutable_ids = (int64_t *)storage;
  mutable_ids[1] = 1;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  mutable_ids[1] = 0;
  mutable_ids[0] = 256;
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(output) == NULL);
  mutable_ids[0] = source_ids[0];
  CHECK(memcmp(storage, source_ids, sizeof(source_ids)) == 0);
  OK(et_g3c4_private_output_release_v1(output));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  prepared_output pending = make_prepared(owner, 0);
  CHECK(et_g3c4_private_p2g2_terminal_ids_clone_v1(pending.output) == NULL);
  discard_prepared(&pending);
  clone_case(owner, 1);
  clone_case(owner, 0);
  clone_failure_matrix(owner);
  printf("G3-C4 P2/G2 logical I1 clone PASS: checks=%zu\n", checks);
  return 0;
}
