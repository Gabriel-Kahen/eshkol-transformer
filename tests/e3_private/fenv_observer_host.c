#include <assert.h>
#include <errno.h>
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <xmmintrin.h>

#include "fenv_observer.c"

static void bracket(int outcome, int mutate) {
#ifdef ET_E3_FENV_MXCSR_MUTANT
  if (mutate) assert(et_e3_test_fenv_mutant_arm_v1() == 0);
#else
  assert(!mutate);
#endif
  int64_t token = et_e3_test_fenv_begin_v1();
  assert(token > 0);
  assert(et_e3_test_fenv_begin_v1() == -1);
  assert(et_e3_test_fenv_end_v1(token + 1, outcome) == -1);
  assert(et_e3_test_fenv_end_v1(token, outcome) == 0);
  assert(et_e3_test_fenv_end_v1(token, outcome) == -1);
  assert(et_e3_test_fenv_ref_v1(0) == token);
  assert(et_e3_test_fenv_ref_v1(1) == 0);
  assert(et_e3_test_fenv_ref_v1(2) == outcome);
#ifdef ET_E3_FENV_MXCSR_MUTANT
  if (mutate) assert((et_e3_test_fenv_ref_v1(3) & 8) != 0);
  else assert(et_e3_test_fenv_ref_v1(3) == 0);
#else
  assert(et_e3_test_fenv_ref_v1(3) == 0);
#endif
  assert(et_e3_test_fenv_ref_v1(4) == sizeof(fenv_t));
  assert(et_e3_test_fenv_ref_v1(5) == 57);
  assert(et_e3_test_fenv_ref_v1(6) == 57);
  assert(et_e3_test_fenv_ref_v1(7) == et_e3_test_fenv_ref_v1(8));
  assert(et_e3_test_fenv_ref_v1(9) == et_e3_test_fenv_ref_v1(10));
  if (!mutate)
    assert(et_e3_test_fenv_ref_v1(11) == et_e3_test_fenv_ref_v1(12));
}

int main(void) {
  fenv_t ambient;
  assert(fegetenv(&ambient) == 0);
  assert(fesetround(FE_DOWNWARD) == 0);
  assert(feraiseexcept(FE_INEXACT) == 0);
  _mm_setcsr(_mm_getcsr() | 1u);  /* A nonzero MXCSR sticky flag. */
  assert((_mm_getcsr() & 0x3fu) != 0);
  errno = 57;
  bracket(0, 0);  /* no-op return calibration */
  bracket(1, 0);  /* no-op caught-path calibration */
#ifdef ET_E3_FENV_MXCSR_MUTANT
#ifdef ET_E3_FENV_EXPECT_KILL
  assert(et_e3_test_fenv_mutant_arm_v1() == 0);
  int64_t token = et_e3_test_fenv_begin_v1();
  assert(token > 0);
  assert(et_e3_test_fenv_end_v1(token, 0) == 0);
  if (et_e3_test_fenv_ref_v1(3) != 0) {
    fputs("E3-FENV-FAIL: ambient-state\n", stderr);
    return 42;
  }
  return 0;
#else
  bracket(0, 1);  /* detector itself sees its causal control-bit mutation */
  assert(et_e3_test_fenv_ref_v1(0) == 3);
#endif
#else
  assert(et_e3_test_fenv_ref_v1(0) == 2);
#endif
  assert(et_e3_test_fenv_ref_v1(99) == -1);
  assert(fesetenv(&ambient) == 0);
  puts("E3-FENV-OBSERVER-HOST-PASS");
  return 0;
}
