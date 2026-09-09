#ifndef ET_N3K_TEST_FINITE_DIFFERENCE_H
#define ET_N3K_TEST_FINITE_DIFFERENCE_H

/* Development-only numerical differentiation of actual native forward calls.
 * The callback observes the supplied mutable primal through its context, writes
 * exactly output_count floats, and returns zero on successful native dispatch.
 * Neither provider schemas nor production derivative code are defined here. */
#include <math.h>
#include <stddef.h>
#include <stdio.h>

typedef int (*n3k_test_forward)(void *context, float *output);

static int n3k_test_central_vjp(
    const char *name, n3k_test_forward forward, void *context, float *primal,
    size_t primal_count, const float *upstream, size_t output_count,
    const float *analytic, float base_step, double absolute, double relative) {
  float positive[1024];
  float negative[1024];
  if (output_count == 0u || output_count > 1024u || primal_count == 0u ||
      !(base_step > 0.0f) || !isfinite(base_step)) return 0;
  for (size_t element = 0; element < primal_count; element++) {
    const float saved = primal[element];
    const float step = base_step * fmaxf(1.0f, fabsf(saved));
    const float plus = saved + step;
    const float minus = saved - step;
    double projected_difference = 0.0;
    if (!isfinite(plus) || !isfinite(minus) || !(plus > minus)) return 0;
    primal[element] = plus;
    int status = forward(context, positive);
    if (status == 0) {
      primal[element] = minus;
      status = forward(context, negative);
    }
    primal[element] = saved;
    if (status != 0) return 0;
    for (size_t index = 0; index < output_count; index++) {
      if (!isfinite(positive[index]) || !isfinite(negative[index]) ||
          !isfinite(upstream[index])) return 0;
      /* Project differences in binary64 to avoid subtracting two large scalar
       * objectives. Native operations and their inputs/outputs remain f32. */
      projected_difference += ((double)positive[index] - (double)negative[index]) *
                              (double)upstream[index];
    }
    const double measured = projected_difference / ((double)plus - (double)minus);
    const double expected = analytic[element];
    const double tolerance = absolute + relative * fmax(fabs(measured), fabs(expected));
    if (!isfinite(expected) || !isfinite(measured) ||
        fabs(measured - expected) > tolerance) {
      fprintf(stderr, "%s element %zu: native VJP %.17g, native central difference "
                      "%.17g, tolerance %.17g\n", name, element, expected, measured, tolerance);
      return 0;
    }
  }
  return 1;
}

#endif
