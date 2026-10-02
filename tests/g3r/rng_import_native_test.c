#include "g3t_transport.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int checks;
static void check(int ok) {
  if (!ok) abort();
  ++checks;
}

int main(void) {
  static const int64_t cases[][4] = {
      {1, 0, 0, 0},
      {1, INT64_MAX, INT64_MIN, INT64_MAX},
      {1, 1729, -1, 0},
      {1, 1729, -2, -1},
      {1, 1729, -1, -1},
      {1, INT64_MAX, INT64_MIN, INT64_MIN},
  };
  check(et_g3t_test_live_rng_clones_v1() == 0);
  for (size_t row = 0; row < sizeof(cases) / sizeof(cases[0]); ++row) {
    void *owner = et_g3t_private_rng_words_create_v1(
        cases[row][0], cases[row][1], cases[row][2], cases[row][3]);
    check(owner != NULL);
    check(et_g3t_test_rng_clone_state_v1(owner) == 1);
    check(et_g3t_test_live_rng_clones_v1() == 1);
    for (int i = 0; i < 4; ++i)
      check(et_g3t_test_rng_clone_word_v1(owner, i) == cases[row][i]);
    check(et_g3t_private_rng_release_v1(owner) == 0);
    check(et_g3t_private_rng_release_v1(owner) == 0);
    check(et_g3t_test_rng_clone_state_v1(owner) == -1);
    for (int i = 0; i < 4; ++i)
      check(et_g3t_test_rng_clone_word_v1(owner, i) == 0);
    check(et_g3t_test_live_rng_clones_v1() == 0);
  }
  check(et_g3t_private_rng_words_create_v1(0, 7, 0, 0) == NULL);
  check(et_g3t_private_last_error_category_v1() == 1);
  check(et_g3t_private_rng_words_create_v1(1, -1, 0, 0) == NULL);
  check(et_g3t_private_last_error_category_v1() == 1);
  check(et_g3t_test_live_rng_clones_v1() == 0);
  check(et_g3t_test_fail_alloc_after_v1(0) == 0);
  check(et_g3t_private_rng_words_create_v1(1, 7, 0, 0) == NULL);
  check(et_g3t_private_last_error_category_v1() == 5);
  check(et_g3t_test_live_rng_clones_v1() == 0);
  check(et_g3t_test_fail_alloc_after_v1(UINT64_MAX) == 0);
  puts("G3-R native typed RNG import PASS");
  return 0;
}
