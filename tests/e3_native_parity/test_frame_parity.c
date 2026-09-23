#include "../../src/eshkol_transformer/e3_frame.c"

#include <inttypes.h>
#include <stdio.h>

#include "e3_native_parity_fixture.h"

static size_t checks;
#define CHECK(value) do { ++checks; if (!(value)) { \
  fprintf(stderr, "FAIL line %d: %s [domain=%lld category=%lld code=%lld]\n", \
      __LINE__, #value, (long long)e3_last_domain, \
      (long long)e3_last_category, (long long)e3_last_code); \
  exit(1); \
} } while (0)
#define OK(value) CHECK((value) == 0)

static const char *const role_names[21] = {
  "ET", "EP", "X", "N1", "QT", "KT", "VT", "QH", "KH", "VH", "AH",
  "AT", "AO", "R", "N2", "FU", "FG", "FD", "Y", "NF", "Z"
};
static unsigned char identities[14];
static et_f32_tensor_error tensor_error;

static owner *make_model(void) {
  initializer *initial = et_m3t_private_initializer_create_v1(1729);
  CHECK(initial != NULL);
  OK(et_m3t_private_initializer_seal_v1(initial));
  owner *model = et_m3t_private_owner_create_v1(initial);
  CHECK(model != NULL);
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_bind_v1(model, (int64_t)i, &identities[i]));
  for (size_t i = 0; i < 14; ++i)
    OK(et_m3t_private_owner_initialize_v1(model, (int64_t)i));
  OK(et_m3t_private_owner_seal_v1(model));
  return model;
}

static et_f32_tensor *scalar(void) {
  et_f32_tensor *tensor = NULL;
  OK(et_f32_tensor_create_v1(0, NULL, &tensor, &tensor_error));
  return tensor;
}

static void read_bits(et_f32_tensor *tensor, uint32_t *bits, size_t count) {
  OK(et_f32_tensor_copy_bits_to_v1(tensor, bits, count, &tensor_error));
}

static void write_scalar(et_f32_tensor *tensor, uint32_t bits) {
  OK(et_f32_tensor_copy_bits_from_v1(tensor, &bits, 1, &tensor_error));
}

static void stage(et_e3_frame_internal *frame,
                  const et_e3_parity_batch *batch) {
  et_kernel_tensor_view_v1 inputs = e3_native_view(
      (void *)batch->inputs, sizeof(batch->inputs), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 targets = e3_native_view(
      (void *)batch->targets, sizeof(batch->targets), "i64", 2, e3_shape_pair);
  et_kernel_tensor_view_v1 mask = e3_native_view(
      (void *)batch->mask, sizeof(batch->mask), "bool", 2, e3_shape_pair);
  OK(et_e3_private_stage_inputs_v1(frame, &inputs));
  OK(et_e3_private_stage_targets_v1(frame, &targets));
  OK(et_e3_private_stage_mask_v1(frame, &mask));
}

static void emit_words(const uint32_t *words, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    if (i) putchar(',');
    printf("\"%08" PRIx32 "\"", words[i]);
  }
}

static void observe_role(FILE *roles, et_e3_frame_internal *frame,
                         size_t ordinal, size_t role) {
  et_f32_scoped_guard_internal guard = {0};
  OK(et_f32_tensor_scoped_begin_internal(frame->forward[role], &guard,
                                          &tensor_error));
  CHECK(guard.view.rank >= 1 && guard.view.rank <= 4);
  CHECK(guard.view.byte_length % sizeof(uint32_t) == 0);
  const size_t count = guard.view.byte_length / sizeof(uint32_t);
  uint32_t words[512];
  CHECK(count <= sizeof(words) / sizeof(words[0]));
  memcpy(words, guard.view.data, guard.view.byte_length);
  fprintf(roles, "%zu\t%s\t", ordinal, role_names[role]);
  for (size_t i = 0; i < guard.view.rank; ++i) {
    if (i) fputc(',', roles);
    fprintf(roles, "%" PRIu64, guard.view.shape[i]);
  }
  fputc('\t', roles);
  for (size_t i = 0; i < count; ++i) {
    if (i) fputc(',', roles);
    fprintf(roles, "%08" PRIx32, words[i]);
  }
  fputc('\n', roles);
  OK(et_f32_tensor_scoped_end_internal(&guard));
}

static void check_empty_retry(et_e3_frame_internal *frame,
                              et_f32_tensor *destination[4]) {
  static const uint32_t sentinels[4] = {
    UINT32_C(0x3f000000), UINT32_C(0x40000000),
    UINT32_C(0x40400000), UINT32_C(0x40800000)
  };
  for (size_t i = 0; i < 4; ++i) write_scalar(destination[i], sentinels[i]);
  int64_t published_before[2];
  memcpy(published_before, frame->published_counts, sizeof(published_before));
  const uint32_t valid_before = frame->published_valid;
  OK(et_e3_private_acquire_v1(frame));
  CHECK(et_e3_private_finalize_v1(frame) == E3_INVALID_STATE);
  CHECK(e3_last_domain == 0 && e3_last_category == E3_INVALID_STATE &&
        e3_last_code == E3_CODE_PHASE);
  CHECK(frame->counts[0][0] == 0 && frame->counts[0][1] == 0 &&
        frame->counts[1][0] == 0 && frame->counts[1][1] == 0);
  OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);
  et_e3_private_finish_v1(frame);
  CHECK(frame->phase == E3_IDLE && frame->model->active == NULL);
  CHECK(frame->published_valid == valid_before &&
        memcmp(frame->published_counts, published_before,
               sizeof(published_before)) == 0);
  for (size_t i = 0; i < 4; ++i) {
    uint32_t actual;
    read_bits(destination[i], &actual, 1);
    CHECK(actual == sentinels[i]);
  }
}

static void run_case(const et_e3_parity_case *test_case, FILE *roles) {
  owner *model = make_model();
  et_f32_tensor *destination[4] = {scalar(), scalar(), scalar(), scalar()};
  et_e3_frame_internal *frame = et_e3_private_frame_create_v1(
      model, destination[0], destination[1], destination[2], destination[3]);
  CHECK(frame != NULL);
  check_empty_retry(frame, destination);
  OK(et_e3_private_acquire_v1(frame));

  printf("{\"batches\":[");
  for (size_t ordinal = 0; ordinal < test_case->batch_count; ++ordinal) {
    const et_e3_parity_batch *batch = &test_case->batches[ordinal];
    stage(frame, batch);
    for (size_t role = 0; role < 21; ++role) {
      OK(et_e3_private_forward_role_v1(frame, (int64_t)role));
      observe_role(roles, frame, ordinal, role);
    }
    uint32_t logits[512], ce[2], reduction[3], correct, state[3];
    read_bits(frame->forward[20], logits, 512);
    OK(et_e3_private_ce_v1(frame));
    read_bits(frame->ce, ce, 2);
    OK(et_e3_private_reduce_v1(frame));
    for (size_t i = 0; i < 3; ++i) read_bits(frame->batch[i], &reduction[i], 1);
    OK(et_e3_private_correct_v1(frame));
    read_bits(frame->batch[3], &correct, 1);
    CHECK(frame->active == (int64_t)batch->mask[0] + (int64_t)batch->mask[1]);
    OK(et_e3_private_accumulate_v1(frame));
    for (size_t i = 0; i < 3; ++i)
      read_bits(frame->sum[frame->sum_bank][i], &state[i], 1);
    const int64_t tokens = frame->counts[frame->sum_bank][0];
    const int64_t batches = frame->counts[frame->sum_bank][1];

    if (ordinal) putchar(',');
    printf("{\"ce_f32\":["); emit_words(ce, 2);
    printf("],\"correct_f32\":\"%08" PRIx32 "\",", correct);
    printf("\"counters\":[%" PRId64 ",%" PRId64 "],", tokens, batches);
    printf("\"inputs\":[%" PRId64 ",%" PRId64 "],",
           batch->inputs[0], batch->inputs[1]);
    printf("\"logits_f32\":["); emit_words(logits, 512);
    printf("],\"mask\":[%s,%s],\"ordinal\":%zu,",
           batch->mask[0] ? "true" : "false",
           batch->mask[1] ? "true" : "false", ordinal);
    printf("\"reduction_f32\":["); emit_words(reduction, 3);
    printf("],\"state_f32\":["); emit_words(state, 3);
    printf("],\"targets\":[%" PRId64 ",%" PRId64 "]}",
           batch->targets[0], batch->targets[1]);
  }

  OK(et_e3_private_finalize_v1(frame));
  OK(et_e3_private_cleanup_preflight_v1(frame));
  et_e3_private_drain_v1(frame);
  OK(et_e3_private_publish_v1(frame));
  et_e3_private_finish_v1(frame);
  uint32_t metrics[4];
  for (size_t i = 0; i < 4; ++i) read_bits(destination[i], &metrics[i], 1);
  const int64_t tokens = et_e3_private_counter_ref_v1(frame, 0);
  const int64_t batches = et_e3_private_counter_ref_v1(frame, 1);
  CHECK(tokens >= 1 && batches >= 1);
  printf("],\"final\":{\"counters\":[%" PRId64 ",%" PRId64
         "],\"metrics_f32\":[", tokens, batches);
  emit_words(metrics, 4);
  printf(" ]},\"kind\":\"observed-f32\","
         "\"profile\":\"e3-private-n1-t2-v256-d4-v1\"}\n");

  OK(et_e3_private_frame_destroy_preflight_v1(frame));
  et_e3_private_frame_destroy_v1(frame);
  for (size_t i = 0; i < 4; ++i)
    OK(et_f32_tensor_destroy_v1(&destination[i], &tensor_error));
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s CASE ROLE-OUTPUT\n", argv[0]);
    return 2;
  }
  const et_e3_parity_case *selected = NULL;
  for (size_t i = 0; i < ET_E3_PARITY_CASE_COUNT; ++i)
    if (strcmp(argv[1], e3_parity_cases[i].name) == 0) selected = &e3_parity_cases[i];
  if (!selected) {
    fprintf(stderr, "unknown parity case: %s\n", argv[1]);
    return 2;
  }
  FILE *roles = fopen(argv[2], "wx");
  if (!roles) {
    perror("open role output");
    return 2;
  }
  run_case(selected, roles);
  CHECK(fclose(roles) == 0);
  fprintf(stderr, "native case %s: %zu checks passed\n", selected->name, checks);
  return 0;
}
