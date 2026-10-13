/* The accepted composer aggregate, with the private holder enabled. */
#define ET_G3C4_P2_G2_HELD_HOLDER_PRIVATE 1
#define __wrap_et_g3c4_private_p2g2_terminal_commit_v1 \
  held_base_wrap_terminal_commit
#include "p2_g2_terminal_composer_native.c"
#undef __wrap_et_g3c4_private_p2g2_terminal_commit_v1

static int held_cut, held_kind, held_begins, held_first, held_ends, held_second;
static int64_t held_first_status, held_first_domain, held_first_category;
static int64_t held_first_code;

/* Cut 1 fails immediately after a genuine holder acquisition. Cut 2 also
 * makes the second abort appear busy; the Eshkol fail-stop must terminate. */
int64_t et_g3c4_p2g2_held_test_arm_v1(int64_t cut) {
  if (cut < 0 || cut > 3) return -1;
  held_cut = (int)cut;
  held_kind = 0;
  held_begins = held_first = held_ends = held_second = 0;
  held_first_status = held_first_domain = held_first_category = 0;
  held_first_code = 0;
  return 0;
}

int64_t et_g3c4_p2g2_held_test_observe_v1(int64_t field) {
  switch (field) {
    case 1: return held_begins;
    case 2: return held_first;
    case 3: return held_ends;
    case 4: return held_second;
    case 5: return held_first_status;
    case 6: return held_first_domain;
    case 7: return held_first_category;
    case 8: return held_first_code;
    case 9:
      return held_first_status == (held_kind == 1 ?
                 ET_I64_TENSOR_ERROR_INVALID_STATE :
                 ET_KERNEL_ERROR_INVALID_ARGUMENT) &&
             held_first_domain == (held_kind == 1 ?
                 ET_G3C4_DOMAIN_I1 : ET_G3C4_DOMAIN_K1) &&
             held_first_category == held_first_status &&
             held_first_code == (held_kind == 1 ?
                 ET_I64_TENSOR_CODE_ACTIVE_BORROW :
                 ET_KERNEL_CODE_PROVIDER_REJECTED);
    default: return -1;
  }
}

int64_t __real_et_g3c4_private_p2g2_holder_begin_capture_v1(
    void *, int64_t, void *, void *);
int64_t __wrap_et_g3c4_private_p2g2_holder_begin_capture_v1(
    void *holder, int64_t generation, void *context, void *output) {
  int64_t status = __real_et_g3c4_private_p2g2_holder_begin_capture_v1(
      holder, generation, context, output);
  if (status == 0) {
    et_g3c4_p2g2_holder *owner = et_g3c4_holder_find(holder, generation);
    if (owner == NULL) abort();
    held_kind = (int)owner->kind;
    held_begins++;
    if (held_cut == 1 || held_cut == 2) {
      held_cut += 4;
      return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
    }
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_holder_abort_attempt_v1(
    void *, int64_t, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_holder_abort_attempt_v1(
    void *holder, int64_t generation, void *context, void *output,
    int64_t ordinal) {
  int64_t status;
  if (ordinal == 2 && held_cut == 6) {
    held_second++;
    if (write(3, "R", 1) != 1) abort();
    return ET_G3C4_INVALID_STATE;
  }
  status = __real_et_g3c4_private_p2g2_holder_abort_attempt_v1(
      holder, generation, context, output, ordinal);
  if (ordinal == 1) {
    held_first++;
    held_first_status = status;
    held_first_domain = et_g3c4_private_last_error_domain_v1();
    held_first_category = et_g3c4_private_last_error_category_v1();
    held_first_code = et_g3c4_private_last_error_code_v1();
  } else if (ordinal == 2) {
    held_second++;
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_holder_end_after_busy_v1(
    void *, int64_t, void *, void *);
int64_t __wrap_et_g3c4_private_p2g2_holder_end_after_busy_v1(
    void *holder, int64_t generation, void *context, void *output) {
  int64_t status = __real_et_g3c4_private_p2g2_holder_end_after_busy_v1(
      holder, generation, context, output);
  if (status == 0) held_ends++;
  return status;
}

int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  int64_t status = held_base_wrap_terminal_commit(
      context, output, first, first_bytes, second, second_bytes);
  /* Every holder must be disposed before native zero, including first EOS. */
  if (status == 0 &&
      (et_g3c4_holder_token_count == 0 ||
       et_g3c4_holder_tokens[et_g3c4_holder_token_count - 1].live != NULL))
    abort();
  if (status == 0 && held_cut == 3) {
    if (write(3, "Z", 1) != 1) abort();
    return ET_G3C4_INVALID_STATE;
  }
  return status;
}
