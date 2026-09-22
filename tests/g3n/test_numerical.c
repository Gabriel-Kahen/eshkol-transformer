/* Independent scalar development oracles. No provider helper is included here. */
#include "test_common.h"
#include <fenv.h>
#include <float.h>

static float add32(float a, float b) { volatile float v = a + b; return v; }
static float mul32(float a, float b) { volatile float v = a * b; return v; }
static float sub32(float a, float b) { volatile float v = a - b; return v; }
static float div32(float a, float b) { volatile float v = a / b; return v; }

static float linear_literal(const float *x, const float *w, size_t n) {
  float a = 0.0f;
  for (size_t i = 0; i < n; ++i) a = add32(a, mul32(x[i], w[i]));
  return a;
}

static void norm_literal(const g3n_fixture *f, float out[4]) {
  const float *x = f->data[0];
  float mean = 0.0f, variance = 0.0f;
  for (size_t i = 0; i < 4u; ++i) mean = add32(mean, x[i]);
  mean = div32(mean, 4.0f);
  for (size_t i = 0; i < 4u; ++i) {
    float delta = sub32(x[i], mean);
    variance = add32(variance, mul32(delta, delta));
  }
  variance = div32(variance, 4.0f);
  float shifted = add32(variance, f->data[3][0]);
  float root = sqrtf(shifted), reciprocal = div32(1.0f, root);
  for (size_t i = 0; i < 4u; ++i) {
    float delta = sub32(x[i], mean);
    float normalized = mul32(delta, reciprocal);
    out[i] = add32(mul32(normalized, f->data[1][i]), f->data[2][i]);
  }
}

/* Algebraic long-double oracle intentionally does not mirror f32 evaluation. */
static void math_reference(const g3n_fixture *f, size_t row, float out[256]) {
  if (row < 2u) {
    memcpy(out, f->data[1] + 4u * (size_t)f->positions[0], 4u * sizeof(float));
  } else if (row < 6u) {
    size_t width = (size_t)f->shape[2], columns = (size_t)f->shape[3];
    for (size_t o = 0; o < columns; ++o) {
      long double a = 0.0L;
      for (size_t d = 0; d < width; ++d)
        a += (long double)f->data[0][d] * (long double)f->data[1][o * width + d];
      out[o] = (float)a;
    }
  } else if (row == 6u) {
    long double sum = 0.0L, squares = 0.0L;
    for (size_t i = 0; i < 4u; ++i) sum += (long double)f->data[0][i];
    long double mean = sum / 4.0L;
    for (size_t i = 0; i < 4u; ++i) {
      long double delta = (long double)f->data[0][i] - mean;
      squares += delta * delta;
    }
    long double denominator = sqrtl(squares / 4.0L + (long double)f->data[3][0]);
    for (size_t i = 0; i < 4u; ++i)
      out[i] = (float)(((long double)f->data[0][i] - mean) / denominator *
                      (long double)f->data[1][i] + (long double)f->data[2][i]);
  } else if (row == 7u) {
    for (size_t i = 0; i < 4u; ++i)
      out[i] = (float)((long double)f->data[0][i] + (long double)f->data[1][i]);
  } else if (row < 10u) {
    /* T=1: both separately checked rank directions have identity flat layout. */
    memcpy(out, f->data[0], 4u * sizeof(float));
  } else {
    for (size_t h = 0; h < 2u; ++h) {
      long double probability = 0.0L;
      if (f->keep && f->positions[1] <= f->positions[0]) {
        long double score = ((long double)f->data[0][h * 2u] * f->data[1][h * 2u] +
                             (long double)f->data[0][h * 2u + 1u] * f->data[1][h * 2u + 1u]) / sqrtl(2.0L);
        long double mass = expl(score - score);
        probability = mass / mass;
      }
      for (size_t d = 0; d < 2u; ++d)
        out[h * 2u + d] = probability == 0.0L ? 0.0f :
                          (float)(probability * (long double)f->data[2][h * 2u + d]);
    }
  }
}

static void exact_metadata(void) {
  static const char *const names[] = {
    "g3n.causal-attention-forward", "g3n.embedding-forward", "g3n.head-layout-forward",
    "g3n.layer-norm-forward", "g3n.linear-forward", "g3n.residual-forward"};
  static const size_t row_counts[] = {1,2,1,1,4,1};
  static const size_t row_indices[][4] = {{10,0,0,0},{0,1,0,0},{9,0,0,0},{6,0,0,0},{2,3,4,5},{7,0,0,0}};
  const et_kernel_provider_v1 *p = et_g3n_kernel_provider_v1();
  CHECK(p != NULL); CHECK(p == et_g3n_kernel_provider_v1());
  CHECK(p->struct_size == ET_KERNEL_PROVIDER_V1_0_SIZE);
  CHECK(p->abi_major == 1u); CHECK(p->abi_minor == 0u); CHECK(p->required_features == 0u);
  CHECK(strcmp(p->name, "g3n.cpu-f32.serial") == 0);
  CHECK(strcmp(p->version, "1.0") == 0);
  CHECK(strcmp(p->evidence, "G3-N:cpu-f32-c2-forward-v1") == 0);
  CHECK(p->capability_count == 6u);
  CHECK(p->capability_stride == ET_KERNEL_CAPABILITY_V1_0_SIZE);
  CHECK(p->capability_bytes == 6u * ET_KERNEL_CAPABILITY_V1_0_SIZE);
  CHECK(p->capabilities != NULL); CHECK(p->validate_call != NULL); CHECK(p->invoke_call != NULL);
  const et_kernel_capability_v1 *caps = p->capabilities;
  size_t pairs = 0u;
  for (size_t ci = 0; ci < 6u; ++ci) {
    const et_kernel_capability_v1 *c = &caps[ci];
    CHECK(c->struct_size == ET_KERNEL_CAPABILITY_V1_0_SIZE);
    CHECK(strcmp(c->name, names[ci]) == 0);
    CHECK(c->status == ET_KERNEL_CAPABILITY_VERIFIED);
    CHECK(strcmp(c->implementation, "g3n.cpu-f32.serial") == 0);
    CHECK(strcmp(c->version, "1.0") == 0);
    CHECK(strcmp(c->evidence, "G3-N:cpu-f32-c2-forward-v1") == 0);
    CHECK(c->deterministic == 1u);
    for (size_t i = 0; i < sizeof(c->reserved); ++i) CHECK(c->reserved[i] == 0u);
    CHECK(c->operation_count == (ci == 2u ? 2u : 1u)); CHECK(c->operations != NULL);
    CHECK(c->dtype_count == 1u); CHECK(c->dtypes != NULL); CHECK(strcmp(c->dtypes[0], "f32") == 0);
    CHECK(c->device_count == 1u); CHECK(c->devices != NULL); CHECK(strcmp(c->devices[0], "cpu") == 0);
    CHECK(c->shape_range_count == row_counts[ci]); CHECK(c->shape_ranges != NULL);
    g3n_fixture f; fixture_init(&f, row_indices[ci][0]);
    CHECK(strcmp(c->operations[0], f.request.operation) == 0);
    if (ci == 2u) CHECK(strcmp(c->operations[1], "g3n.heads.split.forward") == 0);
    for (size_t ri = 0; ri < row_counts[ci]; ++ri) {
      fixture_init(&f, row_indices[ci][ri]);
      const et_kernel_shape_range_v1 *row = &c->shape_ranges[ri];
      CHECK(row->rank == f.request.rank); CHECK(row->dimensions != NULL);
      for (size_t di = 0; di < row->rank; ++di) {
        const et_kernel_dimension_range_v1 *d = &row->dimensions[di];
        CHECK(d->minimum == f.shape[di]); CHECK(d->maximum == f.shape[di]);
        CHECK(d->maximum_unbounded == 0u);
        for (size_t i = 0; i < sizeof(d->reserved); ++i) CHECK(d->reserved[i] == 0u);
      }
    }
    pairs += c->operation_count * c->shape_range_count;
  }
  CHECK(pairs == 11u);
}

static void mathematical_all_rows(et_kernel_runtime *runtime) {
  for (size_t row = 0; row < G3N_ROWS; ++row) {
    g3n_fixture f;
    fixture_init(&f, row);
    float expected[256] = {0}, literal[256] = {0};
    math_reference(&f, row, expected);
    run(runtime, &f.call);
    size_t n = f.outputs[0].byte_length / sizeof(float);
    expect_floats(f.output, expected, n, 2e-6f);
    memcpy(literal, expected, sizeof(literal));
    if (row >= 2u && row < 6u)
      for (size_t o = 0; o < n; ++o)
        literal[o] = linear_literal(f.data[0], f.data[1] + o * (size_t)f.shape[2], (size_t)f.shape[2]);
    if (row == 6u) norm_literal(&f, literal);
    CHECK(memcmp(f.output, literal, n * sizeof(float)) == 0);
    float first[256]; memcpy(first, f.output, sizeof(first));
    run(runtime, &f.call);
    CHECK(memcmp(first, f.output, n * sizeof(float)) == 0);
    f.request.deterministic = 0u;
    run(runtime, &f.call);
    CHECK(memcmp(first, f.output, n * sizeof(float)) == 0);
  }
}

static void rejected_value(et_kernel_runtime *runtime, g3n_fixture *f) {
  g3n_fixture saved;
  memcpy(&saved, f, sizeof(saved));
  reject(runtime, &f->call, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  CHECK(memcmp(&saved, f, sizeof(saved)) == 0);
}

static void embedding(et_kernel_runtime *runtime) {
  for (size_t row = 0; row < 2u; ++row) {
    g3n_fixture f; fixture_init(&f, row);
    size_t vocab = (size_t)f.shape[2];
    for (size_t id = 0; id < vocab; ++id) {
      f.positions[0] = (int64_t)id;
      f.data[1][4u * id] = -0.0f;
      f.data[1][4u * id + 1u] = FLT_TRUE_MIN;
      run(runtime, &f.call);
      CHECK(memcmp(f.output, f.data[1] + 4u * id, sizeof(float) * 4u) == 0);
    }
    f.positions[0] = -1; rejected_value(runtime, &f);
    f.positions[0] = (int64_t)vocab; rejected_value(runtime, &f);
    f.positions[0] = INT64_MAX; rejected_value(runtime, &f);
    f.positions[0] = 0; f.data[1][4u] = NAN; rejected_value(runtime, &f);
    f.data[1][4u] = INFINITY; rejected_value(runtime, &f);
  }
}

static void linear(et_kernel_runtime *runtime) {
  for (size_t row = 2u; row < 6u; ++row) {
    g3n_fixture f; fixture_init(&f, row);
    size_t width = (size_t)f.shape[2], columns = (size_t)f.shape[3];
    /* Every matrix row and every input channel has an independent address. */
    for (size_t selected = 0; selected < width; ++selected) {
      memset(f.data[0], 0, width * sizeof(float)); f.data[0][selected] = 1.0f;
      for (size_t i = 0; i < width * columns; ++i) f.data[1][i] = (float)(i + 1u) / 1024.0f;
      run(runtime, &f.call);
      for (size_t o = 0; o < columns; ++o) CHECK(f.output[o] == f.data[1][o * width + selected]);
      /* Deliberate [Din,Dout] indexing mutant is observably different. */
      if (selected == 0u) CHECK(f.output[1] != f.data[1][selected * columns + 1u]);
    }
    memset(f.data[0], 0, width * sizeof(float));
    for (size_t i = 0; i < width * columns; ++i) f.data[1][i] = 1.0f;
    f.data[0][0] = 0x1p24f; f.data[0][1] = 1.0f;
    f.data[0][2] = -0x1p24f; f.data[0][3] = 1.0f;
    run(runtime, &f.call);
    for (size_t o = 0; o < columns; ++o) CHECK(fbits(f.output[o]) == fbits(1.0f));
    CHECK((float)((double)f.data[0][0] + f.data[0][1] + f.data[0][2] + f.data[0][3]) == 2.0f);
    float reversed = 0.0f;
    for (size_t d = width; d > 0u; --d) reversed = add32(reversed, f.data[0][d - 1u]);
    CHECK(reversed != f.output[0]);
    memset(f.data[0], 0, width * sizeof(float));
    memset(f.data[1], 0, width * columns * sizeof(float));
    f.data[0][0] = 1.0f; f.data[0][1] = 0x1.000002p0f;
    for (size_t o = 0; o < columns; ++o) {
      f.data[1][o * width] = -1.0f; f.data[1][o * width + 1u] = 0x1.fffffep-1f;
    }
    run(runtime, &f.call);
    for (size_t o = 0; o < columns; ++o) CHECK(fbits(f.output[o]) == 0u);
    CHECK(fmaf(f.data[0][1], f.data[1][1], -1.0f) != f.output[0]);
    f.data[0][0] = FLT_MAX; f.data[1][0] = 2.0f; rejected_value(runtime, &f);
    f.data[1][0] = 1.0f; f.data[0][1] = FLT_MAX; f.data[1][1] = 1.0f;
    rejected_value(runtime, &f);
  }
}

static void norm_case(et_kernel_runtime *runtime, g3n_fixture *f, int mathematical) {
  float expected[256], literal[4];
  norm_literal(f, literal); run(runtime, &f->call);
  CHECK(memcmp(literal, f->output, sizeof(literal)) == 0);
  if (mathematical) {
    math_reference(f, 6u, expected);
    expect_floats(f->output, expected, 4u, 2e-6f);
  }
}

static void layer_norm(et_kernel_runtime *runtime) {
  g3n_fixture f; fixture_init(&f, 6u);
  norm_case(runtime, &f, 1);
  /* Biased variance, channel parameters, and epsilon are all observable. */
  float correct[4]; norm_literal(&f, correct);
  float mean = 0.0f, squared = 0.0f;
  for (size_t i = 0; i < 4u; ++i) mean += f.data[0][i] / 4.0f;
  for (size_t i = 0; i < 4u; ++i) squared += (f.data[0][i] - mean) * (f.data[0][i] - mean);
  float unbiased = (f.data[0][0] - mean) / sqrtf(squared / 3.0f + f.data[3][0]) * f.data[1][0] + f.data[2][0];
  CHECK(fabsf(unbiased - correct[0]) > 1e-4f);
  for (size_t i = 0; i < 4u; ++i) f.data[0][i] = 2.0f;
  norm_case(runtime, &f, 1);
  CHECK(memcmp(f.output, f.data[2], 4u * sizeof(float)) == 0);
  f.data[3][0] = FLT_TRUE_MIN;
  norm_case(runtime, &f, 1);
  for (size_t i = 0; i < 4u; ++i) f.data[0][i] = (float)(i + 1u) * FLT_TRUE_MIN;
  norm_case(runtime, &f, 1);
  fixture_init(&f, 6u); f.data[3][0] = 0.125f; norm_case(runtime, &f, 1);
  /* Deterministic binary32 mantissas expose reduction order and affine FMA. */
  uint32_t state = UINT32_C(0x1739ac51);
  for (size_t trial = 0; trial < 64u; ++trial) {
    fixture_init(&f, 6u);
    for (size_t operand = 0; operand < 3u; ++operand)
      for (size_t i = 0; i < 4u; ++i) {
        state = state * UINT32_C(1664525) + UINT32_C(1013904223);
        uint32_t word = UINT32_C(0x3f000000) | (state & UINT32_C(0x007fffff));
        if (state & UINT32_C(0x80000000)) word |= UINT32_C(0x80000000);
        memcpy(&f.data[operand][i], &word, sizeof(word));
      }
    f.data[3][0] = trial % 2u ? 0.125f : FLT_TRUE_MIN;
    norm_case(runtime, &f, 1);
  }
  fixture_init(&f, 6u);
  for (size_t i = 0; i < 4u; ++i) f.data[0][i] = (float)((int)i - 2) * 1e17f;
  norm_case(runtime, &f, 1);
  static const float bad_epsilon[] = {0.0f, -0.0f, -1.0f, NAN, INFINITY, -INFINITY};
  for (size_t i = 0; i < COUNT(bad_epsilon); ++i) {
    fixture_init(&f, 6u); f.data[3][0] = bad_epsilon[i]; rejected_value(runtime, &f);
  }
  /* Reachable failure stages: sum, delta, square, variance accumulation,
     variance+epsilon, affine multiply, affine add. sqrt/reciprocal and
     normalized intermediates cannot overflow after prior finite admission. */
  fixture_init(&f, 6u); f.data[0][0] = FLT_MAX; f.data[0][1] = FLT_MAX; rejected_value(runtime, &f);
  fixture_init(&f, 6u);
  f.data[0][0] = FLT_MAX; f.data[0][1] = -FLT_MAX;
  f.data[0][2] = -FLT_MAX; f.data[0][3] = FLT_MAX / 2.0f; rejected_value(runtime, &f);
  fixture_init(&f, 6u); f.data[0][0] = 1e20f; rejected_value(runtime, &f);
  fixture_init(&f, 6u);
  for (size_t i = 0; i < 4u; ++i) f.data[0][i] = i % 2u ? -1e19f : 1e19f;
  rejected_value(runtime, &f);
  fixture_init(&f, 6u);
  for (size_t i = 0; i < 4u; ++i) f.data[0][i] = i % 2u ? -1e18f : 1e18f;
  f.data[3][0] = FLT_MAX; rejected_value(runtime, &f);
  fixture_init(&f, 6u); memset(f.data[0], 0, 4u * sizeof(float)); f.data[0][3] = 1.0f;
  f.data[1][3] = FLT_MAX; rejected_value(runtime, &f);
  f.data[1][3] = FLT_MAX / 2.0f; f.data[2][3] = FLT_MAX; rejected_value(runtime, &f);
}

static void copies_and_residual(et_kernel_runtime *runtime) {
  const float values[] = {-0.0f, 0.0f, FLT_TRUE_MIN, -FLT_TRUE_MIN};
  for (size_t row = 7u; row < 10u; ++row) {
    g3n_fixture f; fixture_init(&f, row);
    memcpy(f.data[0], values, sizeof(values));
    if (row == 7u) memcpy(f.data[1], values, sizeof(values));
    run(runtime, &f.call);
    for (size_t i = 0; i < 4u; ++i)
      CHECK(fbits(f.output[i]) == fbits(row == 7u ? add32(values[i], values[i]) : values[i]));
    if (row >= 8u) {
      CHECK(f.inputs[0].rank == (row == 8u ? 3u : 4u));
      CHECK(f.outputs[0].rank == (row == 8u ? 4u : 3u));
      f.inputs[0].rank = row == 8u ? 4u : 3u;
      reject(runtime, &f.call, ET_KERNEL_ERROR_SHAPE_MISMATCH, UINT32_MAX);
    } else {
      f.data[0][0] = FLT_MAX; f.data[1][0] = FLT_MAX; rejected_value(runtime, &f);
    }
  }
}

static void attention(et_kernel_runtime *runtime) {
  static const int64_t pairs[][2] = {{0,0},{1,0},{0,1},{16777215,16777215},{16777215,0},{0,16777215}};
  g3n_fixture f;
  for (size_t p = 0; p < COUNT(pairs); ++p) for (unsigned mask = 0; mask < 2u; ++mask) {
    fixture_init(&f, 10u); f.positions[0] = pairs[p][0]; f.positions[1] = pairs[p][1]; f.keep = (uint8_t)mask;
    float expected[256]; math_reference(&f, 10u, expected); run(runtime, &f.call);
    CHECK(memcmp(f.output, expected, 4u * sizeof(float)) == 0);
    int admitted = mask && pairs[p][1] <= pairs[p][0];
    for (size_t i = 0; i < 4u; ++i) CHECK(fbits(f.output[i]) == (admitted ? fbits(f.data[2][i]) : 0u));
  }
  for (size_t head = 0; head < 2u; ++head) {
    fixture_init(&f, 10u); f.data[0][2u * head] = FLT_MAX; f.data[1][2u * head] = 2.0f;
    rejected_value(runtime, &f); /* score must be evaluated even at probability one. */
    f.keep = 0u; run(runtime, &f.call);
    for (size_t i = 0; i < 4u; ++i) CHECK(fbits(f.output[i]) == 0u);
    f.keep = 1u; f.positions[1] = 1; run(runtime, &f.call);
    for (size_t i = 0; i < 4u; ++i) CHECK(fbits(f.output[i]) == 0u);
    f.positions[0] = 1; f.data[1][2u * head] = 1.0f;
    f.data[0][2u * head + 1u] = FLT_MAX; f.data[1][2u * head + 1u] = 1.0f;
    rejected_value(runtime, &f);
  }
  for (size_t operand = 0; operand < 3u; ++operand) for (size_t i = 0; i < 4u; ++i) {
    fixture_init(&f, 10u); f.keep = 0u; f.data[operand][i] = NAN; rejected_value(runtime, &f);
    f.data[operand][i] = INFINITY; rejected_value(runtime, &f);
  }
  for (size_t position = 0; position < 2u; ++position) {
    fixture_init(&f, 10u); f.positions[position] = -1; rejected_value(runtime, &f);
    f.positions[position] = 16777216; rejected_value(runtime, &f);
    f.positions[position] = INT64_MAX; rejected_value(runtime, &f);
  }
  fixture_init(&f, 10u); f.keep = 2; rejected_value(runtime, &f);
  f.keep = 255; rejected_value(runtime, &f);
  fixture_init(&f, 10u); f.data[2][0] = -0.0f; run(runtime, &f.call);
  /* A2 weighted sum starts at +0: +0 + (1 * -0) is +0. */
  CHECK(fbits(f.output[0]) == 0u);
}

int main(void) {
  CHECK(fesetround(FE_TONEAREST) == 0);
  exact_metadata();
  et_kernel_runtime *runtime = runtime_new();
  mathematical_all_rows(runtime); embedding(runtime); linear(runtime);
  layer_norm(runtime); copies_and_residual(runtime); attention(runtime);
  et_kernel_runtime_destroy(runtime);
  printf("G3N numerical: %zu checks; all eleven rows, independent math/literal and discriminating mutations PASS\n", checks);
  return 0;
}
