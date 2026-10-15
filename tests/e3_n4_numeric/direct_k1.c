#include "eshkol_transformer/e3_evaluation_metrics_abi.h"
#include "eshkol_transformer/l3s_masked_objective_abi.h"

#include <errno.h>
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define COUNT(x) (sizeof(x) / sizeof((x)[0]))

enum operation { CORRECT, ACCUMULATE, FINALIZE, REDUCE, SEED };
static const char *const names[] = {
  "e3.evaluation-metrics.correct.bool", "e3.evaluation-metrics.accumulate",
  "e3.evaluation-metrics.finalize", "l3s.masked-objective.reduce.f32",
  "l3s.masked-objective.numerator-seed.f32"
};
typedef union store { uint64_t align; unsigned char bytes[2064]; } store;
typedef struct fixture {
  et_kernel_call_v1 call;
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 in[9], out[5];
  uint64_t request_shape[3], in_shape[9][3], out_shape[5][2];
  char capability[48], operation[64];
  store input[9], output[5];
} fixture;

static void put32(void *p, size_t i, uint32_t bits) {
  memcpy((unsigned char *)p + i * 4u, &bits, 4u);
}
static uint32_t get32(const void *p, size_t i) {
  uint32_t bits; memcpy(&bits, (const unsigned char *)p + i * 4u, 4u); return bits;
}
static void put64(void *p, size_t i, int64_t value) {
  memcpy((unsigned char *)p + i * 8u, &value, 8u);
}
static int64_t get64(const void *p) {
  int64_t value; memcpy(&value, p, 8u); return value;
}
static void *data(store *s) { return s->bytes + 8u; }
static et_kernel_tensor_view_v1 view(void *p, size_t bytes, const char *dtype,
                                     size_t rank, const uint64_t *shape) {
  return (et_kernel_tensor_view_v1){sizeof(et_kernel_tensor_view_v1), p, bytes,
    dtype, "cpu", ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR, 0u, rank, shape};
}
static void setup(fixture *f, enum operation op) {
  memset(f, 0, sizeof(*f));
  const int l3s = op >= REDUCE;
  strcpy(f->capability, l3s ? "l3s.masked-objective" : "e3.evaluation-metrics");
  strcpy(f->operation, names[op]);
  f->request_shape[0] = 1; f->request_shape[1] = 2;
  f->request_shape[2] = l3s ? 0 : 256;
  f->request = (et_kernel_request_v1){sizeof(f->request), f->operation, "f32",
    "cpu", l3s ? 2u : 3u, f->request_shape, 1u, {0}};
  const size_t ni = op == ACCUMULATE ? 9u : op == SEED ? 1u : op == REDUCE ? 2u : 3u;
  const size_t no = op == ACCUMULATE ? 5u : op == FINALIZE ? 4u :
    op == CORRECT ? 2u : op == REDUCE ? 3u : 1u;
  for (size_t i = 0; i < 9u; ++i) {
    memset(f->input[i].bytes, 0xa5, sizeof(store));
    f->in_shape[i][0] = 1; f->in_shape[i][1] = 2; f->in_shape[i][2] = 256;
    const int integer = (op == ACCUMULATE && (i == 3u || i == 4u || i == 8u)) ||
      (op == CORRECT && i == 1u);
    const size_t rank = op == CORRECT ? (i == 0u ? 3u : 2u) : l3s ? 2u : 0u;
    const size_t bytes = op == CORRECT ? (i == 0u ? 2048u : i == 1u ? 16u : 2u) :
      l3s ? 8u : integer ? 8u : 4u;
    f->in[i] = view(data(&f->input[i]), bytes, integer ? "i64" :
      op == CORRECT && i == 2u ? "bool" : "f32", rank,
      rank ? f->in_shape[i] : NULL);
    memset(f->in[i].data, 0, bytes);
  }
  for (size_t i = 0; i < 5u; ++i) {
    memset(f->output[i].bytes, 0x5a, sizeof(store));
    f->out_shape[i][0] = 1; f->out_shape[i][1] = 2;
    const int integer = (op == ACCUMULATE && i >= 3u) || (op == CORRECT && i == 1u);
    f->out[i] = view(data(&f->output[i]), integer ? 8u : op == SEED ? 8u : 4u,
      integer ? "i64" : "f32", op == SEED ? 2u : 0u,
      op == SEED ? f->out_shape[i] : NULL);
  }
  if (op == CORRECT) {
    put64(f->in[1].data, 0, 0); put64(f->in[1].data, 1, 0);
    memset(f->in[2].data, 1, 2u);
  } else if (op == ACCUMULATE) {
    put32(f->in[6].data, 0, 0x3f800000u);
    put64(f->in[8].data, 0, 1);
  } else if (op == FINALIZE) {
    put32(f->in[1].data, 0, 0x3f800000u);
  } else if (op == REDUCE) {
    put32(f->in[0].data, 0, 0x3f800000u);
    put32(f->in[1].data, 0, 0x3f800000u);
    put32(f->in[1].data, 1, 0x3f800000u);
  } else {
    put32(f->in[0].data, 0, 0x3f800000u);
    put32(f->in[0].data, 1, 0x3f800000u);
  }
  f->call = (et_kernel_call_v1){sizeof(f->call), f->capability, &f->request,
    ni, sizeof(f->in[0]), ni * sizeof(f->in[0]), f->in,
    no, sizeof(f->out[0]), no * sizeof(f->out[0]), f->out};
}
static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  CHECK(strcmp(symbol, et_kernel_provider_symbol()) == 0);
  return context;
}
static unsigned short control(void) {
  unsigned short v; __asm__ volatile("fnstcw %0" : "=m"(v)); return v;
}
static unsigned short status_word(void) {
  unsigned short v; __asm__ volatile("fnstsw %0" : "=am"(v)); return v;
}
static void set_control(unsigned short v) { __asm__ volatile("fldcw %0" :: "m"(v)); }
static void sticky(void) {
  __asm__ volatile("fldz\n\tfldz\n\tfdivp\n\tfstp %%st(0)" ::: "st");
  _mm_setcsr(_mm_getcsr() | 0x05u);
}

/* K1 error storage is disjoint from every view. The fixture comparison covers
 * input payloads, descriptors, all metadata, and both sides of every output. */
static void dispatch(et_kernel_runtime *r, fixture *f, uint32_t category,
                     uint32_t code, const char *operation) {
  fixture before; memcpy(&before, f, sizeof(before));
  const unsigned short cw = control(), sw = status_word();
  const unsigned mx = _mm_getcsr();
  et_kernel_error error; memset(&error, 0x71, sizeof(error));
  errno = E2BIG;
  const int32_t rc = et_kernel_runtime_dispatch(r, &f->call, &error);
  if (rc != (int32_t)category || error.code != code)
    fprintf(stderr, "%s: got %d/%u expected %u/%u (%s)\n",
      f->operation, rc, error.code, category, code, error.message);
  CHECK(rc == (int32_t)category);
  CHECK(error.category == category && error.code == code);
  CHECK(strcmp(error.operation, operation) == 0);
  const char *message = category == ET_KERNEL_ERROR_NONE ? "" :
    code == ET_KERNEL_CODE_INVALID_SHAPE ? "target index is outside vocabulary" :
    category == ET_KERNEL_ERROR_UNSUPPORTED ? "unsupported x87 or MXCSR controls" :
    strncmp(f->operation, "e3.", 3u) == 0 ?
      "invalid operand, intermediate or result" :
      "invalid operand, weight, intermediate or result";
  CHECK(strcmp(error.message, message) == 0);
  CHECK(errno == E2BIG);
  CHECK(control() == cw);
  CHECK((_mm_getcsr() & ~0x3fu) == (mx & ~0x3fu));
  if (category != ET_KERNEL_ERROR_NONE) {
    CHECK(status_word() == sw && _mm_getcsr() == mx);
    CHECK(memcmp(f, &before, sizeof(before)) == 0);
    return;
  }
  CHECK((status_word() & sw & 0x3fu) == (sw & 0x3fu));
  CHECK((_mm_getcsr() & mx & 0x3fu) == (mx & 0x3fu));
  for (size_t i = 0; i < f->call.output_count; ++i) {
    const size_t offset = (size_t)((unsigned char *)f->out[i].data - (unsigned char *)f);
    CHECK(offset <= sizeof(*f) - f->out[i].byte_length);
    memcpy((unsigned char *)&before + offset, f->out[i].data, f->out[i].byte_length);
  }
  CHECK(memcmp(f, &before, sizeof(before)) == 0);
}
static void ok(et_kernel_runtime *r, fixture *f) {
  dispatch(r, f, ET_KERNEL_ERROR_NONE, ET_KERNEL_CODE_OK, "");
}
static void domain(et_kernel_runtime *r, fixture *f) {
  dispatch(r, f, ET_KERNEL_ERROR_INVALID_ARGUMENT,
           ET_KERNEL_CODE_PROVIDER_REJECTED, f->operation);
}
static void unsupported(et_kernel_runtime *r, fixture *f) {
  dispatch(r, f, ET_KERNEL_ERROR_UNSUPPORTED,
           ET_KERNEL_CODE_PROVIDER_REJECTED, f->operation);
}

static void bit_cases(et_kernel_runtime *e3, et_kernel_runtime *l3s) {
  static const uint32_t positive[] = {
    0x00000000u, 0x00000001u, 0x007fffffu, 0x00800000u,
    0x00800001u, 0x3f7fffffu, 0x3f800000u, 0x3f800001u
  };
  fixture f;
  for (size_t i = 0; i < COUNT(positive) + 1u; ++i) {
    const uint32_t bits = i == COUNT(positive) ? 0x80000000u : positive[i];
    const uint32_t expected = bits == 0x80000000u ? 0u : bits;
    setup(&f, ACCUMULATE); put32(f.in[5].data, 0, bits); ok(e3, &f);
    CHECK(get32(f.out[0].data, 0) == expected);
    CHECK(get32(f.out[1].data, 0) == 0x3f800000u);
    CHECK(get64(f.out[3].data) == 1 && get64(f.out[4].data) == 1);
    setup(&f, FINALIZE); put32(f.in[0].data, 0, bits); ok(e3, &f);
    CHECK(get32(f.out[0].data, 0) == expected);
    CHECK(get32(f.out[1].data, 0) == 0x3f800000u);
    if (bits <= 0x00800001u || bits == 0x80000000u)
      CHECK(get32(f.out[2].data, 0) == 0x3f800000u);
  }
  for (size_t i = 0; i < 5u; ++i) {
    setup(&f, REDUCE);
    put32(f.in[1].data, 0, positive[i]);
    put32(f.in[1].data, 1, 0x3f800000u);
    ok(l3s, &f);
    CHECK(get32(f.out[0].data, 0) == positive[i]);
    CHECK(get32(f.out[1].data, 0) == 0x3f800000u);
    CHECK(get32(f.out[2].data, 0) == positive[i]);
  }
  setup(&f, SEED); put32(f.in[0].data, 0, 0x80000000u);
  ok(l3s, &f); CHECK(get32(f.out[0].data, 0) == 0u);
  CHECK(get32(f.out[0].data, 1) == 0x3f800000u);

  static const struct { uint32_t bits; uint32_t correct; } logits[] = {
    {0x00000001u, 0u}, {0x007fffffu, 0u}, {0x00800000u, 0u},
    {0x00800001u, 0u}, {0x3f7fffffu, 0u}, {0x3f800000u, 0u},
    {0x3f800001u, 0u}, {0x80000001u, 0x3f800000u},
    {0x80000000u, 0x3f800000u}
  };
  for (size_t i = 0; i < COUNT(logits); ++i) {
    setup(&f, CORRECT);
    ((unsigned char *)f.in[2].data)[1] = 0;
    put32(f.in[0].data, 0, 0x80000000u);
    put32(f.in[0].data, 1, logits[i].bits);
    ok(e3, &f);
    CHECK(get32(f.out[0].data, 0) == logits[i].correct);
    CHECK(get64(f.out[1].data) == 1);
  }
}

static void precedence_cases(et_kernel_runtime *e3, et_kernel_runtime *l3s) {
  static const uint32_t bad[] = {0x7fc00001u, 0x7f800001u,
    0xff800001u, 0x7f800000u, 0xff800000u};
  fixture f;
  for (size_t i = 0; i < COUNT(bad); ++i) {
    setup(&f, CORRECT); ((unsigned char *)f.in[2].data)[1] = 0;
    put32(f.in[0].data, 256, bad[i]); domain(e3, &f);
    setup(&f, ACCUMULATE); put32(f.in[5].data, 0, bad[i]); domain(e3, &f);
    setup(&f, FINALIZE); put32(f.in[0].data, 0, bad[i]); domain(e3, &f);
    setup(&f, REDUCE); put32(f.in[1].data, 0, bad[i]); domain(l3s, &f);
    setup(&f, SEED); put32(f.in[0].data, 0, bad[i]); domain(l3s, &f);
  }
  setup(&f, CORRECT); put64(f.in[1].data, 0, 256);
  put32(f.in[0].data, 0, 0x7fc00001u);
  ((unsigned char *)f.in[2].data)[0] = 2;
  dispatch(e3, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH,
           ET_KERNEL_CODE_INVALID_SHAPE, f.operation);
  put64(f.in[1].data, 0, 0); domain(e3, &f);
  put32(f.in[0].data, 0, 0); domain(e3, &f);
  ((unsigned char *)f.in[2].data)[0] = 1; ok(e3, &f);
  ((unsigned char *)f.in[2].data)[0] = 0;
  ((unsigned char *)f.in[2].data)[1] = 0;
  domain(e3, &f); /* 00 */
  ((unsigned char *)f.in[2].data)[1] = 1; ok(e3, &f); /* 01: provider only */

  setup(&f, REDUCE);
  put32(f.in[1].data, 0, 0x80000001u); domain(l3s, &f);
  setup(&f, REDUCE);
  put32(f.in[0].data, 1, 0x7fc00001u);
  put32(f.in[1].data, 1, 0u); domain(l3s, &f); /* masked CE still validated */
}

static void environment_cases(et_kernel_runtime *e3, et_kernel_runtime *l3s) {
  fixture f;
  fenv_t baseline; CHECK(fegetenv(&baseline) == 0);
  const unsigned short cw = control();
  const unsigned mx = _mm_getcsr();
  et_kernel_runtime *r[] = {e3, l3s};
  for (size_t p = 0; p < COUNT(r); ++p) {
    const enum operation op = p ? REDUCE : CORRECT;
    setup(&f, op);
    if (p) put32(f.in[1].data, 0, 0x7f800001u);
    else put32(f.in[0].data, 0, 0x7f800001u);
    sticky();
    domain(r[p], &f);
    CHECK(fesetenv(&baseline) == 0);
    if (p) put32(f.in[1].data, 0, 0x3f800000u);
    else put32(f.in[0].data, 0, 0u);
    ok(r[p], &f); /* same descriptors and storage after rejection */
    CHECK(fesetenv(&baseline) == 0);

    static const unsigned mx_bits[] = {0x0040u, 0x8000u, 0x2000u};
    for (size_t j = 0; j < COUNT(mx_bits); ++j) {
      setup(&f, op);
      if (p) put32(f.in[1].data, 0, 0x7fc00001u);
      else put32(f.in[0].data, 0, 0x7fc00001u);
      _mm_setcsr(mx | mx_bits[j]);
      unsupported(r[p], &f); /* control rejection precedes operand rejection */
      CHECK(fesetenv(&baseline) == 0);
      domain(r[p], &f);
      if (p) put32(f.in[1].data, 0, 0x3f800000u);
      else put32(f.in[0].data, 0, 0u);
      ok(r[p], &f);
      CHECK(fesetenv(&baseline) == 0);
    }
    setup(&f, op);
    if (p) put32(f.in[1].data, 0, 0x7fc00001u);
    else put32(f.in[0].data, 0, 0x7fc00001u);
    set_control((unsigned short)(cw | 0x0400u));
    unsupported(r[p], &f);
    CHECK(fesetenv(&baseline) == 0);
    domain(r[p], &f);
    if (p) put32(f.in[1].data, 0, 0x3f800000u);
    else put32(f.in[0].data, 0, 0u);
    ok(r[p], &f);
    CHECK(fesetenv(&baseline) == 0);
  }
}

int main(int argc, char **argv) {
  CHECK(argc == 2);
  fenv_t saved; CHECK(fegetenv(&saved) == 0);
  CHECK(fesetenv(FE_DFL_ENV) == 0);
  et_kernel_runtime *e3 = NULL, *l3s = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve,
    (void *)et_e3_metrics_kernel_provider_v1(), &e3, &error) == 0);
  CHECK(et_kernel_runtime_discover(resolve,
    (void *)et_l3s_kernel_provider_v1(), &l3s, &error) == 0);
  if (strcmp(argv[1], "bits") == 0) bit_cases(e3, l3s);
  else if (strcmp(argv[1], "first") == 0) precedence_cases(e3, l3s);
  else if (strcmp(argv[1], "fenv") == 0) environment_cases(e3, l3s);
  else CHECK(0);
  et_kernel_runtime_destroy(l3s);
  et_kernel_runtime_destroy(e3);
  CHECK(fesetenv(&saved) == 0);
  printf("E3 N4 direct K1 %s PASS\n", argv[1]);
  return 0;
}
