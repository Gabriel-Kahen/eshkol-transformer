/* Serial, test-only observer for one E3 invocation.  No evaluator hook. */
#include <errno.h>
#include <fenv.h>
#include <stdint.h>
#include <string.h>
#include <xmmintrin.h>

typedef struct {
  int error_number;
  unsigned short x87_control;
  unsigned short x87_status;
  unsigned int mxcsr;
  int environment_valid;
  fenv_t environment;
} e3_fenv_image;

static struct {
  int64_t generation;
  int active;
  int outcome;
  unsigned int difference;
  e3_fenv_image before;
  e3_fenv_image after;
} e3_fenv_record;
#ifdef ET_E3_FENV_MXCSR_MUTANT
static int e3_fenv_mutant_armed;
#endif

int64_t et_e3_test_fenv_mutant_arm_v1(void) {
#ifdef ET_E3_FENV_MXCSR_MUTANT
  if (e3_fenv_record.active || e3_fenv_mutant_armed) return -1;
  e3_fenv_mutant_armed = 1;
  return 0;
#else
  return 0;
#endif
}

static void e3_fenv_capture(e3_fenv_image *image) {
  int saved_errno = errno;
  memset(&image->environment, 0, sizeof(image->environment));
  image->error_number = saved_errno;
  __asm__ __volatile__("fnstcw %0" : "=m"(image->x87_control));
  __asm__ __volatile__("fnstsw %0" : "=m"(image->x87_status));
  image->mxcsr = _mm_getcsr();
  image->environment_valid = fegetenv(&image->environment) == 0;
  errno = saved_errno;
}

int64_t et_e3_test_fenv_begin_v1(void) {
  if (e3_fenv_record.active || e3_fenv_record.generation == INT64_MAX)
    return -1;
  e3_fenv_record.generation++;
  e3_fenv_record.active = 1;
  e3_fenv_record.outcome = -1;
  e3_fenv_record.difference = 0;
  e3_fenv_capture(&e3_fenv_record.before);
  return e3_fenv_record.generation;
}

/* outcome: 0 returned, 1 caught exception.  The caller checks the report
 * after leaving the bracket, so the capture itself adds no assertions. */
int64_t et_e3_test_fenv_end_v1(int64_t token, int64_t outcome) {
  if (!e3_fenv_record.active || token != e3_fenv_record.generation ||
      (outcome != 0 && outcome != 1)) return -1;
#ifdef ET_E3_FENV_MXCSR_MUTANT
  /* Test-only causal mutant: the observation must see this control change. */
  if (e3_fenv_mutant_armed) {
    _mm_setcsr(_mm_getcsr() ^ 0x8000u);
    e3_fenv_mutant_armed = 0;
  }
#endif
  e3_fenv_capture(&e3_fenv_record.after);
  e3_fenv_record.active = 0;
  e3_fenv_record.outcome = (int)outcome;
  if (e3_fenv_record.before.error_number != e3_fenv_record.after.error_number)
    e3_fenv_record.difference |= 1u;
  if (e3_fenv_record.before.x87_control != e3_fenv_record.after.x87_control)
    e3_fenv_record.difference |= 2u;
  if (e3_fenv_record.before.x87_status != e3_fenv_record.after.x87_status)
    e3_fenv_record.difference |= 4u;
  if (e3_fenv_record.before.mxcsr != e3_fenv_record.after.mxcsr)
    e3_fenv_record.difference |= 8u;
  if (memcmp(&e3_fenv_record.before.environment,
             &e3_fenv_record.after.environment, sizeof(fenv_t)) != 0)
    e3_fenv_record.difference |= 16u;
  if (!e3_fenv_record.before.environment_valid ||
      !e3_fenv_record.after.environment_valid)
    e3_fenv_record.difference |= 32u;
  return 0;
}

/* Selectors expose exact words without copying an opaque fenv_t over FFI.
 * 0 generation, 1 active, 2 outcome, 3 difference, 4 fenv_t size,
 * 5/6 errno, 7/8 x87 CW, 9/10 x87 SW, 11/12 MXCSR. */
int64_t et_e3_test_fenv_ref_v1(int64_t selector) {
  switch (selector) {
    case 0: return e3_fenv_record.generation;
    case 1: return e3_fenv_record.active;
    case 2: return e3_fenv_record.outcome;
    case 3: return e3_fenv_record.difference;
    case 4: return (int64_t)sizeof(fenv_t);
    case 5: return e3_fenv_record.before.error_number;
    case 6: return e3_fenv_record.after.error_number;
    case 7: return e3_fenv_record.before.x87_control;
    case 8: return e3_fenv_record.after.x87_control;
    case 9: return e3_fenv_record.before.x87_status;
    case 10: return e3_fenv_record.after.x87_status;
    case 11: return e3_fenv_record.before.mxcsr;
    case 12: return e3_fenv_record.after.mxcsr;
    default: return -1;
  }
}

int64_t et_e3_test_fenv_env_byte_v1(int64_t side, int64_t index) {
  if (e3_fenv_record.active || (side != 0 && side != 1) ||
      index < 0 || (uint64_t)index >= sizeof(fenv_t)) return -1;
  const e3_fenv_image *image = side == 0 ? &e3_fenv_record.before :
                                          &e3_fenv_record.after;
  return ((const unsigned char *)&image->environment)[index];
}
