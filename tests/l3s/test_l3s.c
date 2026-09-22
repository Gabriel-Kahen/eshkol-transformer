#include "eshkol_transformer/l3s_masked_objective_abi.h"

#include <fenv.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h>

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
static size_t checks;
static unsigned short x87_control(void);
static unsigned short x87_status(void);
#define CHECK(c) do { ++checks; if (!(c)) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); \
} } while (0)

static const char *const operations[] = {
  "l3s.masked-objective.reduce.bool", "l3s.masked-objective.reduce.f32",
  "l3s.masked-objective.numerator-seed.bool",
  "l3s.masked-objective.numerator-seed.f32",
  "l3s.masked-objective.mean-seed.bool",
  "l3s.masked-objective.mean-seed.f32"
};

typedef union storage { uint64_t alignment; unsigned char bytes[40]; } storage;
typedef struct fixture {
  et_kernel_call_v1 call;
  et_kernel_request_v1 request;
  et_kernel_tensor_view_v1 inputs[2], outputs[3];
  uint64_t request_shape[2], input_shapes[2][2], output_shape[2];
  _Alignas(8) char capability[128], operation[128], f32[128], boolean[128], cpu[128];
  storage input_data[2], output_data[3];
} fixture;

static void put_bits(void *data, size_t index, uint32_t bits) {
  memcpy((unsigned char *)data + index * 4u, &bits, 4u);
}
static uint32_t get_bits(const void *data, size_t index) {
  uint32_t bits;
  memcpy(&bits, (const unsigned char *)data + index * 4u, 4u);
  return bits;
}
static void *payload(storage *s) { return s->bytes + 8u; }
static et_kernel_tensor_view_v1 view(fixture *f, void *data, int boolean,
                                     size_t rank, const uint64_t *shape) {
  et_kernel_tensor_view_v1 v = {0};
  v.struct_size = sizeof(v); v.data = data;
  v.byte_length = rank ? (boolean ? 2u : 8u) : 4u;
  v.dtype = boolean ? f->boolean : f->f32; v.device = f->cpu;
  v.layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR; v.rank = rank; v.shape = shape;
  return v;
}
static void setup(fixture *f, size_t operation) {
  memset(f, 0, sizeof(*f));
  strcpy(f->capability, "l3s.masked-objective");
  strcpy(f->operation, operations[operation]);
  strcpy(f->f32, "f32"); strcpy(f->boolean, "bool"); strcpy(f->cpu, "cpu");
  const int reduce = operation < 2u, boolean = operation % 2u == 0u;
  f->request_shape[0] = 1u; f->request_shape[1] = 2u;
  f->output_shape[0] = 1u; f->output_shape[1] = 2u;
  f->request = (et_kernel_request_v1){sizeof(f->request), f->operation,
    f->f32, f->cpu, 2u, f->request_shape, 1u, {0}};
  for (size_t i = 0; i < 2u; ++i) {
    memset(f->input_data[i].bytes, 0xa5, sizeof(storage));
    f->input_shapes[i][0] = 1u; f->input_shapes[i][1] = 2u;
    f->inputs[i] = view(f, payload(&f->input_data[i]),
      boolean && i == (reduce ? 1u : 0u), 2u, f->input_shapes[i]);
  }
  if (reduce) {
    put_bits(f->inputs[0].data, 0, UINT32_C(0x40000000));
    put_bits(f->inputs[0].data, 1, UINT32_C(0x41200000));
  }
  void *mask = f->inputs[reduce ? 1u : 0u].data;
  if (boolean) memset(mask, 1, 2u);
  else { put_bits(mask, 0, UINT32_C(0x3f800000)); put_bits(mask, 1, UINT32_C(0x3f800000)); }
  for (size_t i = 0; i < 3u; ++i) {
    memset(f->output_data[i].bytes, 0x5a, sizeof(storage));
    f->outputs[i] = view(f, payload(&f->output_data[i]), 0,
      reduce ? 0u : 2u, reduce ? NULL : f->output_shape);
  }
  f->call = (et_kernel_call_v1){sizeof(f->call), f->capability, &f->request,
    reduce ? 2u : 1u, sizeof(f->inputs[0]),
    (reduce ? 2u : 1u) * sizeof(f->inputs[0]), f->inputs,
    reduce ? 3u : 1u, sizeof(f->outputs[0]),
    (reduce ? 3u : 1u) * sizeof(f->outputs[0]), f->outputs};
}
static void *mask_data(fixture *f) {
  return f->inputs[f->call.input_count - 1u].data;
}
static const et_kernel_provider_v1 *resolve(void *context, const char *symbol) {
  CHECK(context == NULL);
  CHECK(strcmp(symbol, et_kernel_provider_symbol()) == 0);
  return et_l3s_kernel_provider_v1();
}

/* The error record is separate trusted K1 control storage. Never exercise an
 * error/data alias as a recoverable provider rejection: K1 writes it first. */
static void reject(et_kernel_runtime *runtime, fixture *f, uint32_t category,
                   uint32_t code) {
  fixture before; memcpy(&before, f, sizeof(before));
  et_kernel_error error;
  memset(&error, 0x71, sizeof(error));
  const unsigned short control = x87_control(), status_word = x87_status();
  const unsigned mxcsr = _mm_getcsr();
  const int32_t status = et_kernel_runtime_dispatch(runtime, &f->call, &error);
  CHECK(x87_control() == control); CHECK(x87_status() == status_word);
  CHECK(_mm_getcsr() == mxcsr);
  if (status != (int32_t)category || (code != UINT32_MAX && error.code != code))
    fprintf(stderr, "%s: got %d/%u expected %u/%u (%s)\n", f->operation,
      status, error.code, category, code, error.message);
  CHECK(status == (int32_t)category); CHECK(error.category == category);
  CHECK(error.code != ET_KERNEL_CODE_OK);
  if (code != UINT32_MAX) CHECK(error.code == code);
  CHECK(memcmp(f, &before, sizeof(before)) == 0);
}
static void succeed(et_kernel_runtime *runtime, fixture *f) {
  fixture before; memcpy(&before, f, sizeof(before));
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, &f->call, &error) == 0);
  CHECK(error.category == ET_KERNEL_ERROR_NONE && error.code == ET_KERNEL_CODE_OK);
  /* Exempt only exact declared output spans; metadata, inputs and surrounding
   * guard bytes remain covered, including when outputs share adjacent backing. */
  for (size_t i = 0; i < f->call.output_count; ++i) {
    const uintptr_t offset = (uintptr_t)f->outputs[i].data - (uintptr_t)f;
    CHECK(offset <= sizeof(*f) - f->outputs[i].byte_length);
    memcpy((unsigned char *)&before + offset, f->outputs[i].data,
           f->outputs[i].byte_length);
  }
  CHECK(memcmp(f, &before, sizeof(before)) == 0);
}
static void output_is(fixture *f, size_t output, size_t index, uint32_t bits) {
  CHECK(get_bits(f->outputs[output].data, index) == bits);
}

static void capability_tests(et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *p = et_l3s_kernel_provider_v1();
  CHECK(p == et_l3s_kernel_provider_v1()); CHECK(p->capability_count == 1u);
  CHECK(strcmp(p->name, "l3s.cpu-f32.serial") == 0);
  CHECK(strcmp(p->version, "1.0") == 0);
  CHECK(strcmp(p->evidence, "L3S:cpu-f32-masked-objective-v1") == 0);
  CHECK(p->abi_major == 1u && p->abi_minor == 0u && p->required_features == 0u);
  const et_kernel_capability_v1 *c =
    et_kernel_runtime_capability_find(runtime, "l3s.masked-objective");
  CHECK(c != NULL); CHECK(c->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(strcmp(c->implementation, p->name) == 0);
  CHECK(strcmp(c->version, p->version) == 0 && strcmp(c->evidence, p->evidence) == 0);
  CHECK(c->deterministic == 1u && c->operation_count == COUNT(operations));
  CHECK(c->dtype_count == 1u && strcmp(c->dtypes[0], "f32") == 0);
  CHECK(c->device_count == 1u && strcmp(c->devices[0], "cpu") == 0);
  CHECK(c->shape_range_count == 1u && c->shape_ranges[0].rank == 2u);
  for (size_t d = 0; d < 2u; ++d) {
    const et_kernel_dimension_range_v1 *range = &c->shape_ranges[0].dimensions[d];
    CHECK(range->minimum == d + 1u && range->maximum == d + 1u);
    CHECK(range->maximum_unbounded == 0u);
  }
  size_t verified = 0;
  for (size_t i = 0; i < et_kernel_runtime_capability_count(runtime); ++i)
    verified += et_kernel_runtime_capability_at(runtime, i)->status == ET_KERNEL_CAPABILITY_VERIFIED;
  CHECK(verified == 1u);
  et_kernel_runtime *baseline = NULL; et_kernel_error error;
  CHECK(et_kernel_runtime_baseline(&baseline, &error) == 0);
  for (size_t op = 0; op < COUNT(operations); ++op) {
    size_t matches = 0;
    for (size_t i = 0; i < c->operation_count; ++i)
      matches += strcmp(c->operations[i], operations[op]) == 0;
    CHECK(matches == 1u);
    fixture f; setup(&f, op);
    reject(baseline, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    for (unsigned deterministic = 0; deterministic < 2u; ++deterministic) {
      f.request.deterministic = (uint8_t)deterministic;
      CHECK(et_kernel_runtime_capability_require(runtime, f.capability, &f.request, NULL, &error) == 0);
    }
    for (size_t d = 0; d < 2u; ++d) for (size_t side = 0; side < 2u; ++side) {
      setup(&f, op); f.request_shape[d] = side ? d + 2u : d;
      reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    }
    setup(&f, op); f.request.rank = 1u;
    reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f, op); f.request.dtype = f.boolean;
    reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f, op); strcpy(f.operation, "l3s.masked-objective.unknown");
    reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
    setup(&f, op); strcpy(f.cpu, "cuda:0");
    reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  }
  et_kernel_runtime_destroy(baseline);
}

static void positive_tests(et_kernel_runtime *runtime) {
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f; setup(&f, op); succeed(runtime, &f);
    if (op < 2u) {
      output_is(&f, 0, 0, 0x41400000u); output_is(&f, 1, 0, 0x40000000u);
      output_is(&f, 2, 0, 0x40c00000u);
    } else for (size_t t = 0; t < 2u; ++t)
      output_is(&f, 0, t, op < 4u ? 0x3f800000u : 0x3f000000u);
    storage expected[3]; memcpy(expected, f.output_data, sizeof(expected));
    for (size_t repeat = 0; repeat < 64u; ++repeat) {
      succeed(runtime, &f); CHECK(memcmp(expected, f.output_data, sizeof(expected)) == 0);
    }
    /* Old output contents have no numerical input role. */
    setup(&f, op);
    for (size_t o = 0; o < f.call.output_count; ++o)
      for (size_t t = 0; t < f.outputs[o].byte_length / 4u; ++t)
        put_bits(f.outputs[o].data, t, 0x7f800001u);
    succeed(runtime, &f);
    if (op % 2u == 0u) {
      setup(&f, op);
      et_kernel_tensor_view_v1 *mask = &f.inputs[f.call.input_count - 1u];
      mask->data = (unsigned char *)mask->data + 1u;
      memset(mask->data, 1, 2u); /* bool alignment is one byte */
      succeed(runtime, &f);
    }
    for (size_t zero = 0; zero < 2u; ++zero) {
      setup(&f, op);
      if (op % 2u == 0u) ((unsigned char *)mask_data(&f))[zero] = 0;
      else put_bits(mask_data(&f), zero, 0x80000000u);
      succeed(runtime, &f);
      if (op < 2u) {
        const uint32_t selected = zero ? 0x40000000u : 0x41200000u;
        output_is(&f, 0, 0, selected); output_is(&f, 1, 0, 0x3f800000u);
        output_is(&f, 2, 0, selected);
      } else { output_is(&f, 0, zero, 0u); output_is(&f, 0, 1u-zero, 0x3f800000u); }
    }
    if (op < 2u) {
      setup(&f, op); put_bits(f.inputs[0].data, 0, 0x80000000u); put_bits(f.inputs[0].data, 1, 0u);
      succeed(runtime, &f); output_is(&f, 0, 0, 0u); output_is(&f, 2, 0, 0u);
      /* Distinct scalar views may occupy adjacent words of one caller buffer. */
      setup(&f, op);
      for (size_t i = 0; i < 3u; ++i) f.outputs[i].data = f.output_data[0].bytes + 8u + 4u*i;
      succeed(runtime, &f);
    }
    if (op % 2u != 0u) {
      setup(&f, op); put_bits(mask_data(&f), 0, 0x3f000000u); put_bits(mask_data(&f), 1, 0x40000000u);
      succeed(runtime, &f);
      if (op == 1u) {
        output_is(&f, 0, 0, 0x41a80000u); output_is(&f, 1, 0, 0x40200000u);
        output_is(&f, 2, 0, 0x41066666u);
      } else {
        output_is(&f, 0, 0, op == 3u ? 0x3f000000u : 0x3e4ccccdu);
        output_is(&f, 0, 1, op == 3u ? 0x40000000u : 0x3f4ccccdu);
      }
      setup(&f, op); put_bits(mask_data(&f), 0, 1u); put_bits(mask_data(&f), 1, 1u);
      if (op == 1u) { put_bits(f.inputs[0].data, 0, 0u); put_bits(f.inputs[0].data, 1, 0u); }
      succeed(runtime, &f);
      if (op == 1u) { output_is(&f, 0, 0, 0u); output_is(&f, 1, 0, 2u); output_is(&f, 2, 0, 0u); }
      else for (size_t t = 0; t < 2u; ++t) output_is(&f, 0, t, op == 3u ? 1u : 0x3f000000u);
      setup(&f, op); put_bits(mask_data(&f), 0, 1u); put_bits(mask_data(&f), 1, 0x7f7fffffu);
      if (op == 1u) { put_bits(f.inputs[0].data, 0, 0u); put_bits(f.inputs[0].data, 1, 0u); }
      succeed(runtime, &f);
      if (op == 5u) { output_is(&f, 0, 0, 0u); output_is(&f, 0, 1, 0x3f800000u); }
    }
  }
  fixture f;
  const uint32_t boundaries[] = {1u, 0x007fffffu, 0x00800000u, 0x7f7fffffu};
  for (size_t i = 0; i < COUNT(boundaries); ++i) {
    setup(&f, 1); put_bits(f.inputs[0].data, 0, boundaries[i]); put_bits(f.inputs[0].data, 1, 0u);
    put_bits(mask_data(&f), 1, 0u); succeed(runtime, &f);
    output_is(&f, 0, 0, boundaries[i]); output_is(&f, 2, 0, boundaries[i]);
    for (size_t op = 1u; op < COUNT(operations); op += 2u) {
      setup(&f, op); put_bits(mask_data(&f), 0, boundaries[i]); put_bits(mask_data(&f), 1, 0u);
      if (op == 1u) { put_bits(f.inputs[0].data, 0, 0u); put_bits(f.inputs[0].data, 1, 0u); }
      succeed(runtime, &f);
      if (op == 1u) { output_is(&f, 0, 0, 0u); output_is(&f, 1, 0, boundaries[i]); output_is(&f, 2, 0, 0u); }
      else { output_is(&f, 0, 0, op == 3u ? boundaries[i] : 0x3f800000u); output_is(&f, 0, 1, 0u); }
    }
  }
  /* Gradual underflow, including exact halfway rounding to positive zero. */
  for (uint32_t small = 1u; small <= 3u; small += 2u) {
    setup(&f, 1); put_bits(f.inputs[0].data, 0, small); put_bits(f.inputs[0].data, 1, 0u);
    put_bits(mask_data(&f), 0, 0x3f000000u); put_bits(mask_data(&f), 1, 0u);
    succeed(runtime, &f); output_is(&f, 0, 0, small == 1u ? 0u : 2u);
    output_is(&f, 2, 0, small == 1u ? 0u : 4u);
  }
  setup(&f, 5); put_bits(mask_data(&f), 0, 0x40400000u); put_bits(mask_data(&f), 1, 0x40800000u);
  succeed(runtime, &f); output_is(&f, 0, 0, 0x3edb6db7u); /* reciprocal-first differs by one ULP */
  setup(&f, 1); put_bits(f.inputs[0].data, 0, 0x33800000u); put_bits(f.inputs[0].data, 1, 0x3f800001u);
  put_bits(mask_data(&f), 1, 0x3f800001u);
  succeed(runtime, &f); output_is(&f, 0, 0, 0x3f800002u); /* contracted result is 0x3f800003 */
}

static void value_rejections(et_kernel_runtime *runtime) {
  const uint32_t invalid[] = {0x7fc00001u, 0x7f800001u, 0xff800001u,
    0x7f800000u, 0xff800000u, 0xbf800000u, 0x80000001u};
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f; setup(&f, op);
    for (size_t input = 0; input < f.call.input_count; ++input) {
      if (strcmp(f.inputs[input].dtype, "f32") != 0) continue;
      for (size_t pattern = 0; pattern < COUNT(invalid); ++pattern)
        for (size_t token = 0; token < 2u; ++token) {
          setup(&f, op); put_bits(f.inputs[input].data, token, invalid[pattern]);
          reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
          if (op < 2u && input == 0u) {
            if (op == 0u) ((unsigned char *)mask_data(&f))[token] = 0;
            else put_bits(mask_data(&f), token, 0u);
            reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
          }
        }
    }
    if (op % 2u == 0u) for (unsigned bad = 2; bad <= 255u; bad += 253u)
      for (size_t token = 0; token < 2u; ++token) {
        setup(&f, op); ((unsigned char *)mask_data(&f))[token] = (unsigned char)bad;
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
      }
    setup(&f, op); memset(mask_data(&f), 0, op % 2u == 0u ? 2u : 8u);
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    if (op % 2u != 0u) {
      put_bits(mask_data(&f), 0, 0x80000000u); put_bits(mask_data(&f), 1, 0x80000000u);
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
      setup(&f, op); put_bits(mask_data(&f), 0, 0x7f7fffffu); put_bits(mask_data(&f), 1, 0x7f7fffffu);
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
    }
  }
  fixture f;
  for (size_t token = 0; token < 2u; ++token) {
    setup(&f, 1); put_bits(f.inputs[0].data, token, 0x7f7fffffu); put_bits(mask_data(&f), token, 0x40000000u);
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  for (size_t op = 0; op < 2u; ++op) {
    setup(&f, op); put_bits(f.inputs[0].data, 0, 0x7f7fffffu); put_bits(f.inputs[0].data, 1, 0x7f7fffffu);
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
  }
  /* Independent rational witness: products 0x7df95289/0x7e1c9ee3,
   * W=0x3e8ca414 and N=0x7e8ca414 are finite; N/W alone is 2^128.
   * Failure must not publish the otherwise valid numerator or denominator. */
  setup(&f, 1); put_bits(f.inputs[0].data, 0, 0x7f7fffffu); put_bits(f.inputs[0].data, 1, 0x7f7fffffu);
  put_bits(mask_data(&f), 0, 0x3df9528au); put_bits(mask_data(&f), 1, 0x3e1c9ee4u);
  reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
}

static void schema_rejections(et_kernel_runtime *runtime) {
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f; setup(&f, op);
    const size_t ni = f.call.input_count, no = f.call.output_count;
    for (size_t side = 0; side < 2u; ++side) for (size_t index = 0; index < (side ? no : ni); ++index) {
#define RESET_VIEW() setup(&f, op); v = side ? &f.outputs[index] : &f.inputs[index]
      et_kernel_tensor_view_v1 *v;
      RESET_VIEW(); v->struct_size = sizeof(*v) - 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      RESET_VIEW(); v->struct_size = sizeof(*v) + 8u;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      RESET_VIEW(); v->byte_length--;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->byte_length++;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->data = NULL;
      reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->data = (void *)(UINTPTR_MAX - 1u);
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
      RESET_VIEW(); v->layout = 0u;
      reject(runtime, &f, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->offset_bytes = 4u;
      reject(runtime, &f, ET_KERNEL_ERROR_NONCONTIGUOUS, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW(); v->dtype = "f16"; v->byte_length = v->rank ? 4u : 2u;
      reject(runtime, &f, ET_KERNEL_ERROR_DTYPE_MISMATCH, ET_KERNEL_CODE_INVALID_TEXT);
      RESET_VIEW(); v->device = "cuda:0";
      reject(runtime, &f, ET_KERNEL_ERROR_DEVICE_MISMATCH, ET_KERNEL_CODE_INVALID_BUFFER);
      RESET_VIEW();
      if (strcmp(v->dtype, "f32") == 0) {
        v->data = (unsigned char *)v->data + 1u;
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      }
      RESET_VIEW();
      if (v->rank != 0u) {
        uint64_t *shape = side ? f.output_shape : f.input_shapes[index];
        shape[0] = 2u; shape[1] = 1u;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->rank = 1u; v->shape = &f.request_shape[1];
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->shape = NULL;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW();
        (side ? f.output_shape : f.input_shapes[index])[0] = UINT64_MAX;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INTEGER_OVERFLOW);
        RESET_VIEW(); v->shape = (const uint64_t *)((const unsigned char *)v->shape + 1u);
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      } else {
        v->shape = f.request_shape;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
        RESET_VIEW(); v->rank = 1u; v->shape = f.request_shape;
        reject(runtime, &f, ET_KERNEL_ERROR_SHAPE_MISMATCH, ET_KERNEL_CODE_INVALID_SHAPE);
      }
#undef RESET_VIEW
    }
    setup(&f, op); f.call.struct_size--;
    reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
    setup(&f, op); f.request.struct_size--;
    reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
    setup(&f, op); f.request.deterministic = 2u;
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_TEXT);
    for (size_t side = 0; side < 2u; ++side) {
      setup(&f, op);
      if (side) --f.call.output_stride; else --f.call.input_stride;
      reject(runtime, &f, ET_KERNEL_ERROR_VERSION_MISMATCH, ET_KERNEL_CODE_INVALID_STRUCT_SIZE);
      setup(&f, op);
      if (side) --f.call.output_bytes; else --f.call.input_bytes;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) f.call.outputs = (unsigned char *)f.outputs + 1u;
      else f.call.inputs = (unsigned char *)f.inputs + 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
      setup(&f, op);
      if (side) f.call.output_count = SIZE_MAX; else f.call.input_count = SIZE_MAX;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) f.call.output_stride = SIZE_MAX - 7u; else f.call.input_stride = SIZE_MAX - 7u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INTEGER_OVERFLOW);
      setup(&f, op);
      if (side) { f.call.output_count = 0u; f.call.output_stride = 0u; f.call.output_bytes = 0u; f.call.outputs = NULL; }
      else { f.call.input_count = 0u; f.call.input_stride = 0u; f.call.input_bytes = 0u; f.call.inputs = NULL; }
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_INVALID_BUFFER);
    }
  }
}

static void alias_rejections(et_kernel_runtime *runtime) {
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f; setup(&f, op);
    const size_t ni = f.call.input_count, no = f.call.output_count;
    for (size_t o = 0; o < no; ++o) for (size_t i = 0; i < ni; ++i) {
      setup(&f, op); f.outputs[o].data = f.inputs[i].data;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
      setup(&f, op); f.outputs[o].data = (unsigned char *)f.inputs[i].data + 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    for (size_t a = 0; a < no; ++a) for (size_t b = a+1u; b < no; ++b) {
      setup(&f, op); f.outputs[b].data = f.outputs[a].data;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
      setup(&f, op); f.outputs[b].data = (unsigned char *)f.outputs[a].data + 1u;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    if (ni == 2u) for (size_t offset = 0; offset <= 4u; offset += 4u) {
      setup(&f, op); f.inputs[1].data = (unsigned char *)f.inputs[0].data + offset;
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    for (size_t o = 0; o < no; ++o) for (size_t metadata = 0; metadata < 12u; ++metadata) {
      setup(&f, op);
      void *spans[] = {&f.call, &f.request, f.inputs, f.outputs,
        f.request_shape, f.input_shapes[0], ni == 2u ? f.input_shapes[1] : f.output_shape,
        f.capability, f.operation, f.f32, f.cpu,
        op % 2u == 0u ? (void *)f.boolean : (void *)f.operation};
      f.outputs[o].data = spans[metadata];
      reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    }
    /* Exact endpoint adjacency is allowed, including shared input backing. */
    setup(&f, op); f.outputs[0].data = (unsigned char *)f.inputs[0].data +
      (f.inputs[0].byte_length + 3u) / 4u * 4u;
    succeed(runtime, &f);
    if (ni == 2u) {
      setup(&f, op); f.inputs[1].data = (unsigned char *)f.inputs[0].data + 8u;
      if (op == 0u) memset(f.inputs[1].data, 1, 2u);
      else { put_bits(f.inputs[1].data, 0, 0x3f800000u); put_bits(f.inputs[1].data, 1, 0x3f800000u); }
      succeed(runtime, &f);
    }
  }
}

static void compatible_table_tests(et_kernel_runtime *runtime) {
  typedef struct extended_view {
    et_kernel_tensor_view_v1 prefix;
    unsigned char extension[8];
  } extended_view;
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f; setup(&f, op);
    struct { et_kernel_request_v1 prefix; unsigned char extension[8]; } request;
    memset(&request, 0xe9, sizeof(request)); request.prefix = f.request;
    request.prefix.struct_size = sizeof(request); f.call.request = &request.prefix;
    extended_view inputs[2], outputs[3], saved_inputs[2], saved_outputs[3];
    memset(inputs, 0xc7, sizeof(inputs)); memset(outputs, 0xd8, sizeof(outputs));
    for (size_t i = 0; i < f.call.input_count; ++i) {
      inputs[i].prefix = f.inputs[i]; inputs[i].prefix.struct_size = sizeof(inputs[i]);
    }
    for (size_t i = 0; i < f.call.output_count; ++i) {
      outputs[i].prefix = f.outputs[i]; outputs[i].prefix.struct_size = sizeof(outputs[i]);
    }
    f.call.inputs = inputs; f.call.input_stride = sizeof(inputs[0]);
    f.call.input_bytes = f.call.input_count * sizeof(inputs[0]);
    f.call.outputs = outputs; f.call.output_stride = sizeof(outputs[0]);
    f.call.output_bytes = f.call.output_count * sizeof(outputs[0]);
    memcpy(saved_inputs, inputs, sizeof(inputs)); memcpy(saved_outputs, outputs, sizeof(outputs));
    unsigned char saved_request[sizeof(request)]; memcpy(saved_request, &request, sizeof(request));
    succeed(runtime, &f);
    CHECK(memcmp(saved_inputs, inputs, sizeof(inputs)) == 0);
    CHECK(memcmp(saved_outputs, outputs, sizeof(outputs)) == 0);
    CHECK(memcmp(saved_request, &request, sizeof(request)) == 0);
    /* Metadata exclusions cover the full strided table, including extensions. */
    outputs[0].prefix.data = inputs[f.call.input_count - 1u].extension;
    memcpy(saved_outputs, outputs, sizeof(outputs));
    reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_ALIASING_OUTPUT);
    CHECK(memcmp(saved_inputs, inputs, sizeof(inputs)) == 0);
    CHECK(memcmp(saved_outputs, outputs, sizeof(outputs)) == 0);
    CHECK(memcmp(saved_request, &request, sizeof(request)) == 0);
  }
}

static unsigned short x87_control(void) {
  unsigned short value; __asm__ volatile("fnstcw %0" : "=m"(value)); return value;
}
static unsigned short x87_status(void) {
  unsigned short value; __asm__ volatile("fnstsw %0" : "=am"(value)); return value;
}
static void set_x87_control(unsigned short value) { __asm__ volatile("fldcw %0" :: "m"(value)); }
static void invalid_value(fixture *f, size_t op) {
  if (op < 2u) put_bits(f->inputs[0].data, 1u, 0x7f800001u);
  else if (op % 2u != 0u) put_bits(mask_data(f), 1u, 0x7f800001u);
  else ((unsigned char *)mask_data(f))[1] = 2u;
}
static void environment_tests(et_kernel_runtime *runtime) {
  fenv_t saved; CHECK(fegetenv(&saved) == 0);
  CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
  const unsigned short cw = x87_control(); const unsigned mx = _mm_getcsr();
  for (size_t op = 0; op < COUNT(operations); ++op) {
    fixture f;
    const unsigned mx_bits[] = {0x0040u, 0x8000u, 0x2000u, 0x4000u, 0x6000u};
    for (size_t b = 0; b < COUNT(mx_bits); ++b) {
      setup(&f, op); invalid_value(&f, op); _mm_setcsr(mx | mx_bits[b]);
      reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(_mm_getcsr() == (mx | mx_bits[b])); CHECK(x87_control() == cw); _mm_setcsr(mx);
    }
    for (size_t b = 0; b < 6u; ++b) {
      setup(&f, op); invalid_value(&f, op);
      const unsigned changed = mx & ~(1u << (7u+b)); _mm_setcsr(changed);
      reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(_mm_getcsr() == changed); _mm_setcsr(mx);
      const unsigned short changed_cw = (unsigned short)(cw & ~(1u << b)); set_x87_control(changed_cw);
      reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(x87_control() == changed_cw); CHECK(_mm_getcsr() == mx); set_x87_control(cw);
    }
    for (unsigned b = 0x400u; b <= 0xc00u; b += 0x400u) {
      setup(&f, op); invalid_value(&f, op); set_x87_control((unsigned short)(cw | b));
      reject(runtime, &f, ET_KERNEL_ERROR_UNSUPPORTED, ET_KERNEL_CODE_PROVIDER_REJECTED);
      CHECK(x87_control() == (unsigned short)(cw | b)); set_x87_control(cw);
    }
    /* Direct validate is tested only with full K1 descriptor/span/alias
     * preconditions satisfied. It must preserve flags even on success. */
    for (size_t failure = 0; failure < 3u; ++failure) {
      setup(&f, op);
      if (failure == 0u) {
        if (op < 2u) {
          put_bits(f.inputs[0].data, 0u, 0x3f800000u);
          put_bits(f.inputs[0].data, 1u, 0x33800000u); /* inexact addition */
        } else if (op == 3u) {
          put_bits(mask_data(&f), 1u, 1u); /* inexact W addition */
        } else if (op == 5u) {
          put_bits(mask_data(&f), 0u, 0x40400000u);
          put_bits(mask_data(&f), 1u, 0x40800000u); /* inexact division */
        }
      }
      if (failure == 1u) memset(mask_data(&f), 0, op % 2u == 0u ? 2u : 8u);
      if (failure == 2u) invalid_value(&f, op);
      fixture before; memcpy(&before, &f, sizeof(before));
      CHECK(feclearexcept(FE_ALL_EXCEPT) == 0);
      /* Raise invalid in x87 only, then set divide-by-zero in SSE only. Merely using
       * feraiseexcept can exercise the same unit as the provider and miss an
       * incorrect restoration that merges the two independent flag sets. */
      __asm__ volatile("fldz\n\tfldz\n\tfdivp\n\tfstp %%st(0)" ::: "st");
      _mm_setcsr((_mm_getcsr() & ~0x3fu) | 0x04u);
      CHECK((x87_status() & 0x3fu) == 1u); CHECK((_mm_getcsr() & 0x3fu) == 0x04u);
      const unsigned short before_cw = x87_control(), before_sw = x87_status();
      const unsigned before_mx = _mm_getcsr(); et_kernel_error error;
      const int32_t rc = et_l3s_kernel_provider_v1()->validate_call(&f.call, &error);
      CHECK((rc != 0) == (failure != 0)); CHECK(x87_status() == before_sw);
      CHECK(x87_control() == before_cw); CHECK(_mm_getcsr() == before_mx);
      CHECK(memcmp(&before, &f, sizeof(f)) == 0);
      if (failure) {
        reject(runtime, &f, ET_KERNEL_ERROR_INVALID_ARGUMENT, ET_KERNEL_CODE_PROVIDER_REJECTED);
        CHECK(x87_status() == before_sw); CHECK(_mm_getcsr() == before_mx);
      }
      CHECK(feclearexcept(FE_ALL_EXCEPT) == 0); _mm_setcsr(mx);
    }
  }
  CHECK(fesetenv(&saved) == 0);
}

int main(void) {
  fenv_t original; CHECK(fegetenv(&original) == 0); CHECK(fesetenv(FE_DFL_ENV) == 0);
  et_kernel_runtime *runtime = NULL; et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve, NULL, &runtime, &error) == 0);
  capability_tests(runtime); positive_tests(runtime); value_rejections(runtime);
  schema_rejections(runtime); alias_rejections(runtime); compatible_table_tests(runtime);
  environment_tests(runtime);
  et_kernel_runtime_destroy(runtime); CHECK(fesetenv(&original) == 0);
  printf("L3S native: %zu checks; six operations, exact bytes, atomicity, fenv PASS\n", checks);
  return 0;
}
