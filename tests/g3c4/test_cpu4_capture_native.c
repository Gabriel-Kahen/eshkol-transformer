/* Future native gate: intentionally source-staged and NOT RUN in this slice. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_G3C4_NATIVE_OWNER_TESTING 1
#define ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE 1
#define ET_I64_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_F32_TENSOR_STORAGE_QUERY_PRIVATE 1
#define ET_G3C4_P2_G2_TERMINAL_TEST_MAIN cpu4_capture_terminal_predecessor_main
#include "test_p2_g2_native_terminal.c"
#undef ET_G3C4_P2_G2_TERMINAL_TEST_MAIN

static size_t capture_registry_count(void) {
  size_t n=0;for(cpu4_capture_provenance *p=cpu4_captures;p;p=p->next) ++n;
  return n;
}
static void capture_fault_controls(et_g3c4_model_owner_internal *owner) {
  et_g3c4_context_internal *c=seed_generator(owner,9);
  size_t count=capture_registry_count();uint32_t capacity;
  for(size_t cut=0;cut<3u;++cut) {
    et_g3c4_allocation_limit=et_g3c4_successful_allocations+cut;
    CHECK(et_cpu4_private_c4_capture_create_v1(c)==NULL);
    et_g3c4_allocation_limit=SIZE_MAX;
    CHECK(capture_registry_count()==count);
    CHECK(cpu4_history_cache_check(c,cpu4_history_auth(c),&capacity)==0);
    CHECK(et_g3c4_pins_idle(&c->pins) && c->owner->active==NULL);
  }
  for(size_t cut=0;cut<6u;++cut) {
    et_a2_kv_cache_test_fail_alloc_after_v1(cut);
    CHECK(et_cpu4_private_c4_capture_create_v1(c)==NULL);
    et_a2_kv_cache_test_reset_allocator_v1();
    CHECK(capture_registry_count()==count);
    CHECK(cpu4_history_cache_check(c,cpu4_history_auth(c),&capacity)==0);
    et_g3c4_model_pins_internal pins={0};et_f32_tensor_error e;
    OK(et_g3c4_model_pins_begin_internal(c->owner->parameters,c->owner->handles,&pins,&e));
    et_g3c4_model_pins_end_internal(&pins);
  }
  OK(et_g3c4_private_generator_close_v1(c));
}

int main(void) {
  et_g3c4_model_owner_internal *owner = create_owner();
  capture_fault_controls(owner);
  et_g3c4_context_internal *source = seed_generator(owner, 7);
  et_cpu4_c4_capture *capture =
      et_cpu4_private_c4_capture_create_v1(source);
  et_cpu4_c4_capture_record_v1 record, second;
  _Alignas(float) unsigned char values[4768], values_again[4768];
  CHECK(capture != NULL);
  memset(&record, 0xa5, sizeof(record));
  OK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &record, sizeof(record)));
  CHECK(record.size == 224u && record.major == 1u &&
        record.count == 0u && record.cache_capacity == 4u &&
        record.parameter_count == 14u && record.values_bytes == 4768u &&
        record.state == 1u);
  OK(et_cpu4_private_c4_capture_copy_parameters_v1(
      capture, values, sizeof(values)));
  et_g3c4_model_pins_internal pins = {0};
  et_f32_tensor_error pin_error;
  OK(et_g3c4_model_pins_begin_internal(
      owner->parameters, owner->handles, &pins, &pin_error));
  size_t offset = 0u;
  for (size_t i = 0u; i < 14u; ++i) {
    const et_kernel_tensor_view_v1 *view = &pins.views[i];
    CHECK(view->byte_length <= sizeof(values) - offset);
    CHECK(memcmp(values + offset, view->data, view->byte_length) == 0);
    offset += view->byte_length;
  }
  CHECK(offset == sizeof(values));
  OK(et_g3c4_model_pins_check_internal(&pins, &pin_error));
  et_g3c4_model_pins_end_internal(&pins);
  memset(&second, 0xa5, sizeof(second));
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &second, sizeof(second) - 1u) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(((const unsigned char *)&second)[0] == 0xa5u);
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, (et_cpu4_c4_capture_record_v1 *)&capture->magic, sizeof(record)) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(capture->magic == CPU4_CAPTURE_MAGIC);
  et_g3c4_error_set_internal(ET_G3C4_DOMAIN_C4, ET_G3C4_INVALID_STATE,
                              ET_G3C4_CODE_LIFECYCLE);
  et_g3c4_error_state_internal before_tls = et_g3c4_error_snapshot_internal();
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, (et_cpu4_c4_capture_record_v1 *)&et_g3c4_error_state,
      sizeof(record)) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(memcmp(&before_tls, &et_g3c4_error_state, sizeof(before_tls)) == 0);
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(NULL,
      (et_cpu4_c4_capture_record_v1 *)&et_g3c4_error_state,0u)==ET_G3C4_INVALID_ARGUMENT);
  CHECK(memcmp(&before_tls,&et_g3c4_error_state,sizeof(before_tls))==0);
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(NULL,
      (et_cpu4_c4_capture_record_v1 *)((uintptr_t)&et_g3c4_error_state-1u),0u)==ET_G3C4_INVALID_ARGUMENT);
  CHECK(memcmp(&before_tls,&et_g3c4_error_state,sizeof(before_tls))==0);
  CHECK(et_cpu4_private_c4_capture_copy_parameters_v1(NULL,
      &et_g3c4_error_state,UINT64_MAX)==ET_G3C4_INVALID_ARGUMENT);
  CHECK(memcmp(&before_tls,&et_g3c4_error_state,sizeof(before_tls))==0);

  cpu4_capture_provenance *node = cpu4_captures;
  CHECK(node != NULL && node->handle == capture);
  node->payload->values[0] ^= 1u;
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &second, sizeof(second)) == ET_G3C4_INTERNAL);
  node->payload->values[0] ^= 1u;
  OK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &second, sizeof(second)));

  OK(et_g3c4_private_generator_close_v1(source));
  OK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &second, sizeof(second)));
  OK(et_cpu4_private_c4_capture_copy_parameters_v1(
      capture, values_again, sizeof(values_again)));
  CHECK(memcmp(&record, &second, sizeof(record)) == 0);
  CHECK(memcmp(values, values_again, sizeof(values)) == 0);
  et_cpu4_c4_capture forged = *capture;
  CHECK(et_cpu4_private_c4_capture_release_v1(&forged) ==
        ET_G3C4_INVALID_ARGUMENT);
  capture->magic ^= 1u;
  CHECK(et_cpu4_private_c4_capture_release_v1(capture) == ET_G3C4_INTERNAL);
  capture->magic ^= 1u;
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  CHECK(node->payload == NULL && node->extent == 0u);
  capture->padding = 1u;
  CHECK(et_cpu4_private_c4_capture_release_v1(capture) == ET_G3C4_INTERNAL);
  capture->padding = 0u;
  OK(et_cpu4_private_c4_capture_release_v1(capture));
  CHECK(et_cpu4_private_c4_capture_copy_record_v1(
      capture, &second, sizeof(second)) == ET_G3C4_INVALID_STATE);

  /* A committed prompt binding must match capture's local pins at idle. */
  et_g3c4_context_internal *nonempty = seed_generator(owner, 8);
  int64_t prompt[3] = {3, 7, 11};
  float logits[256];
  OK(et_g3c4_private_call_acquire_v1(nonempty, 2, 1));
  OK(et_g3c4_private_prefill3_v1(nonempty, prompt, logits));
  OK(et_g3c4_private_call_abort_v1(nonempty));
  et_cpu4_c4_capture *prompt_capture =
      et_cpu4_private_c4_capture_create_v1(nonempty);
  CHECK(prompt_capture != NULL);
  OK(et_cpu4_private_c4_capture_copy_record_v1(
      prompt_capture, &second, sizeof(second)));
  CHECK(second.count == 3u && second.cache_capacity == 4u &&
        second.prompt_count == 3u && second.generated_count == 0u &&
        second.manual_count == 0u && second.commit_epoch == 1u);
  for (size_t i = 0u; i < 3u; ++i)
    CHECK(second.ids[i] == prompt[i] && second.origins[i] == 1u);
  OK(et_cpu4_private_c4_capture_release_v1(prompt_capture));
  OK(et_g3c4_private_generator_close_v1(nonempty));
  printf("CPU4 C4 capture future native fixture PASS: checks=%zu\n", checks);
  return 0;
}
