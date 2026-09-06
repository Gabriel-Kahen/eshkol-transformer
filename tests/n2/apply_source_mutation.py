#!/usr/bin/env python3
"""Apply one exact N2 negative-test mutation to a temporary provider source."""

from pathlib import Path
import sys


MUTATIONS = {
    "batch-index-collapse": (
        "return (n * s->t + t) * s->a + i;",
        "(void)n;\n  return t * s->a + i;",
    ),
    "reduction-order": (
        """for (size_t n = 0; n < s->n; n++) {
        for (size_t t = 0; t < s->t; t++) {
          const float product = upstream[index_linear_y(s, n, t, o)] *
                                x[index_linear_x(s, n, t, i)];""",
        """for (size_t t = 0; t < s->t; t++) {
        for (size_t n = 0; n < s->n; n++) {
          const float product = upstream[index_linear_y(s, n, t, o)] *
                                x[index_linear_x(s, n, t, i)];""",
    ),
    "philox-mask-inversion": (
        "if (uniform >= probability) {",
        "if (uniform < probability) {",
    ),
    "embedding-reduction-order": (
        """for (size_t n = 0; n < s->n; n++) {
        for (size_t t = 0; t < s->t; t++) {
          if ((size_t)ids[n * s->t + t] == v) {""",
        """for (size_t t = 0; t < s->t; t++) {
        for (size_t n = 0; n < s->n; n++) {
          if ((size_t)ids[n * s->t + t] == v) {""",
    ),
    "layer-norm-variance-formula": (
        "*rstd = variance_sum / dimension;",
        "*rstd = variance_sum * dimension;",
    ),
    "layer-norm-mean-order": (
        """const float dimension = (float)s->d;
  for (size_t d = 0; d < s->d; d++) {
    sum = sum + x[index3(s, n, t, d)];""",
        """const float dimension = (float)s->d;
  for (size_t offset = 0; offset < s->d; offset++) {
    const size_t d = s->d - 1u - offset;
    sum = sum + x[index3(s, n, t, d)];""",
    ),
    "gelu-inv-sqrt-two": (
        """static int gelu_one(float x, float *result) {
  const float half = 0x1p-1f;
  const float one = 0x1p+0f;
  const float inv_sqrt_2 = 0x1.6a09e6p-1f;""",
        """static int gelu_one(float x, float *result) {
  const float half = 0x1p-1f;
  const float one = 0x1p+0f;
  const float inv_sqrt_2 = 0x1.6a09e8p-1f;""",
    ),
    "gelu-slope-order": (
        "const float slope = cdf + x * density;",
        "const float slope = (cdf + x) * density;",
    ),
    "philox-lane-rotation": (
        "const size_t lane = index % 4u;",
        "const size_t lane = (index + 1u) % 4u;",
    ),
    "philox-key-halves": (
        "uint32_t k0 = (uint32_t)key;",
        "uint32_t k0 = (uint32_t)(key >> 32);",
    ),
    "philox-counter-offset": (
        "uint64_t block_low = state->counter_low;",
        "uint64_t block_low = state->counter_low + UINT64_C(1);",
    ),
    "dropout-sentinel-bypass": (
        """if ((state.counter_low == UINT64_MAX &&
           state.counter_high == UINT64_MAX) ||
          !advance_counter(&state, blocks, &next)) {""",
        "if ((void)blocks, (void)next, 0) {",
    ),
    "relu-zero-kink": (
        "value = x[index] > 0.0f",
        "value = x[index] >= 0.0f",
    ),
    "residual-second-edge-zero": (
        "memcpy(output_at(call, 1u)->data, input_at(call, 0u)->data, bytes);",
        "memset(output_at(call, 1u)->data, 0, bytes);",
    ),
}


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] not in MUTATIONS:
        raise SystemExit("usage: apply_source_mutation.py MUTATION INPUT OUTPUT")
    source = Path(sys.argv[2]).read_text(encoding="utf-8")
    old, new = MUTATIONS[sys.argv[1]]
    if source.count(old) != 1:
        raise SystemExit(f"mutation anchor count is {source.count(old)}, expected 1")
    Path(sys.argv[3]).write_text(source.replace(old, new), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
