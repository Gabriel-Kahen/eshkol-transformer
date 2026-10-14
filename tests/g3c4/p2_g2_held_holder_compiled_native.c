/* The accepted composer aggregate, with the private holder enabled. */
#define ET_G3C4_P2_G2_HELD_HOLDER_PRIVATE 1
#define __wrap_et_g3c4_private_p2g2_terminal_commit_v1 \
  held_base_wrap_terminal_commit
#include "p2_g2_terminal_composer_native.c"
#undef __wrap_et_g3c4_private_p2g2_terminal_commit_v1

static int held_cut, held_kind, held_begins, held_first, held_ends, held_second;
static int64_t held_first_status, held_first_domain, held_first_category;
static int64_t held_first_code;
static int held_trace_case;

/* Diagnostic only: fd 4 is a separate, unbuffered gate artifact.  A missing
 * or unwritable trace must not turn a crash into an apparent passing test. */
static void held_trace(const char *phase) {
  char line[128];
  int size = snprintf(line, sizeof(line), "case=%d phase=%s\n",
                      held_trace_case, phase);
  if (size < 0 || (size_t)size >= sizeof(line) ||
      write(4, line, (size_t)size) != size)
    abort();
}

int64_t et_g3c4_p2g2_held_test_phase_v1(int64_t phase) {
  const char *name = NULL;
  switch (phase) {
    case 1: name = "init.begin"; break;
    case 2: name = "init.tokenizer"; break;
    case 3: name = "init.model"; break;
    case 4: name = "init.input"; break;
    case 5: name = "init.mode"; break;
    case 10: held_trace_case++; name = "case.begin"; break;
    case 11: name = "fresh.begin"; break;
    case 12: name = "fresh.generator"; break;
    case 13: name = "fresh.native"; break;
    case 14: name = "fresh.holder"; break;
    case 15: name = "case.armed"; break;
    case 16: name = "generate.enter"; break;
    case 17: name = "generate.return"; break;
    case 18: name = "snapshot.return"; break;
    case 19: name = "postchecks.begin"; break;
    case 20: name = "terminal.released"; break;
    case 21: name = "generator.closed"; break;
    case 22: name = "case.done"; break;
    case 23: name = "reject.return"; break;
    default: abort();
  }
  held_trace(name);
  return 0;
}

/* Cuts 1/2 fail at the first HELD witness, after Eshkol records the lease.
 * Cut 2 also makes the second abort appear busy; Eshkol must fail-stop. */
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
  held_trace("native.begin.enter");
  int64_t status = __real_et_g3c4_private_p2g2_holder_begin_capture_v1(
      holder, generation, context, output);
  held_trace("native.begin.return");
  if (status == 0) {
    et_g3c4_p2g2_holder *owner = et_g3c4_holder_find(holder, generation);
    if (owner == NULL) abort();
    held_kind = (int)owner->kind;
    held_begins++;
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_holder_witness_v1(
    void *, int64_t, void *, void *);
int64_t __wrap_et_g3c4_private_p2g2_holder_witness_v1(
    void *holder, int64_t generation, void *context, void *output) {
  held_trace("native.witness.enter");
  int64_t status = __real_et_g3c4_private_p2g2_holder_witness_v1(
      holder, generation, context, output);
  held_trace("native.witness.return");
  if (status == 0 && (held_cut == 1 || held_cut == 2)) {
    held_cut += 4;
    return et_g3c4_fail(ET_G3C4_INVALID_STATE, ET_G3C4_CODE_LIFECYCLE);
  }
  return status;
}

int64_t __real_et_g3c4_private_p2g2_holder_abort_attempt_v1(
    void *, int64_t, void *, void *, int64_t);
int64_t __wrap_et_g3c4_private_p2g2_holder_abort_attempt_v1(
    void *holder, int64_t generation, void *context, void *output,
    int64_t ordinal) {
  int64_t status;
  held_trace("native.abort.enter");
  if (ordinal == 2 && held_cut == 6) {
    held_second++;
    if (write(3, "R", 1) != 1) abort();
    held_trace("native.abort.cut");
    return ET_G3C4_INVALID_STATE;
  }
  status = __real_et_g3c4_private_p2g2_holder_abort_attempt_v1(
      holder, generation, context, output, ordinal);
  held_trace("native.abort.return");
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
  held_trace("native.end.enter");
  int64_t status = __real_et_g3c4_private_p2g2_holder_end_after_busy_v1(
      holder, generation, context, output);
  held_trace("native.end.return");
  if (status == 0) held_ends++;
  return status;
}

int64_t __wrap_et_g3c4_private_p2g2_terminal_commit_v1(
    void *context, void *output, const void *first, int64_t first_bytes,
    const void *second, int64_t second_bytes) {
  held_trace("native.commit.enter");
  int64_t status = held_base_wrap_terminal_commit(
      context, output, first, first_bytes, second, second_bytes);
  held_trace("native.commit.return");
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
