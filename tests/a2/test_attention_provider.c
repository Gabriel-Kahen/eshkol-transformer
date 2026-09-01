#include "eshkol_transformer/a2_attention_abi.h"
#include "reference_vectors.h"

#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;

#define ET_ARRAY_COUNT(values) (sizeof(values) / sizeof((values)[0]))

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #condition);                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static int close_float(float actual, float expected, float tolerance) {
  return fabsf(actual - expected) <= tolerance;
}

static void expect_array(const float *actual, const float *expected,
                         size_t count, float tolerance) {
  for (size_t index = 0; index < count; index++) {
    CHECK(close_float(actual[index], expected[index], tolerance));
  }
}

static void expect_reference(const float *actual, const uint32_t *expected_bits,
                             size_t count, float tolerance) {
  for (size_t index = 0u; index < count; index++) {
    float expected;
    memcpy(&expected, &expected_bits[index], sizeof(expected));
    CHECK(close_float(actual[index], expected, tolerance));
  }
}

static void expect_positive_zeros(const float *values, size_t count) {
  for (size_t index = 0; index < count; index++) {
    uint32_t bits;
    memcpy(&bits, &values[index], sizeof(bits));
    CHECK(bits == UINT32_C(0));
  }
}

static et_kernel_tensor_view_v1 view(void *data, size_t byte_length,
                                     const char *dtype, size_t rank,
                                     const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
      .data = data,
      .byte_length = byte_length,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = rank,
      .shape = shape,
  };
  return result;
}

static et_kernel_request_v1 request(const char *operation, size_t rank,
                                    const uint64_t *shape) {
  et_kernel_request_v1 result = {
      .struct_size = sizeof(et_kernel_request_v1),
      .operation = operation,
      .dtype = "f32",
      .device = "cpu",
      .rank = rank,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  return result;
}

static et_kernel_call_v1 call(const char *capability,
                              const et_kernel_request_v1 *kernel_request,
                              et_kernel_tensor_view_v1 *inputs,
                              size_t input_count,
                              et_kernel_tensor_view_v1 *outputs,
                              size_t output_count) {
  et_kernel_call_v1 result = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = capability,
      .request = kernel_request,
      .input_count = input_count,
      .input_stride = sizeof(et_kernel_tensor_view_v1),
      .input_bytes = input_count * sizeof(et_kernel_tensor_view_v1),
      .inputs = inputs,
      .output_count = output_count,
      .output_stride = sizeof(et_kernel_tensor_view_v1),
      .output_bytes = output_count * sizeof(et_kernel_tensor_view_v1),
      .outputs = outputs,
  };
  return result;
}

static const et_kernel_provider_v1 *resolve_a2(void *context,
                                               const char *symbol) {
  (void)context;
  CHECK(strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) == 0);
  return et_a2_kernel_provider_v1();
}

static et_kernel_runtime *make_runtime(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  CHECK(et_kernel_runtime_discover(resolve_a2, NULL, &runtime, &error) == 0);
  CHECK(runtime != NULL);
  return runtime;
}

static void expect_capability_request(et_kernel_runtime *runtime,
                                      const char *capability,
                                      const char *operation, size_t rank,
                                      uint64_t *shape, int supported) {
  et_kernel_request_v1 req = request(operation, rank, shape);
  const et_kernel_capability_v1 *entry = NULL;
  et_kernel_error error;
  int32_t result = et_kernel_runtime_capability_require(
      runtime, capability, &req, &entry, &error);
  CHECK((result == 0) == supported);
  CHECK((entry != NULL) == supported);
  if (!supported) {
    CHECK(error.category == ET_KERNEL_ERROR_UNSUPPORTED);
    CHECK(error.code == ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED);
  }
}

static void test_metadata(et_kernel_runtime *runtime) {
  const et_kernel_provider_v1 *provider = et_a2_kernel_provider_v1();
  const et_kernel_capability_v1 *attention;
  const et_kernel_capability_v1 *rope;
  CHECK(ET_A2_ATTENTION_ABI_MAJOR == 1u);
  CHECK(ET_A2_ATTENTION_ABI_MINOR == 0u);
  CHECK(ET_A2_MAX_EXACT_POSITION == INT64_C(16777215));
  CHECK(provider != NULL);
  CHECK(provider->abi_major == ET_KERNEL_ABI_MAJOR);
  CHECK(provider->abi_minor == ET_KERNEL_ABI_MINOR);
  CHECK(strcmp(provider->name, "a2.cpu-f32.serial") == 0);
  CHECK(provider->capability_count == 2u);
  CHECK(provider->validate_call != NULL);
  CHECK(provider->invoke_call != NULL);
  attention = et_kernel_runtime_capability_find(runtime,
                                                "kernel.causal-attention");
  rope = et_kernel_runtime_capability_find(runtime, "kernel.rope");
  CHECK(attention != NULL && rope != NULL);
  CHECK(attention->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(rope->status == ET_KERNEL_CAPABILITY_VERIFIED);
  CHECK(attention->deterministic == 1u && rope->deterministic == 1u);
  CHECK(attention->shape_range_count == 10u);
  CHECK(rope->shape_range_count == 6u);
  for (size_t range = 0u; range < attention->shape_range_count; range++) {
    CHECK(attention->shape_ranges[range].rank == 6u);
    for (size_t dimension = 0u; dimension < 6u; dimension++) {
      const et_kernel_dimension_range_v1 *constraint =
          &attention->shape_ranges[range].dimensions[dimension];
      CHECK(constraint->maximum_unbounded == 0u);
      CHECK(constraint->minimum == constraint->maximum);
    }
  }
  for (size_t range = 0u; range < rope->shape_range_count; range++) {
    CHECK(rope->shape_ranges[range].rank == 4u);
    for (size_t dimension = 0u; dimension < 4u; dimension++) {
      const et_kernel_dimension_range_v1 *constraint =
          &rope->shape_ranges[range].dimensions[dimension];
      CHECK(constraint->maximum_unbounded == 0u);
      CHECK(constraint->minimum == constraint->maximum);
      if (dimension == 3u) CHECK(constraint->minimum % 2u == 0u);
    }
  }
  {
    uint64_t attention_shapes[][6] = {
        {1u,2u,2u,1u,1u,1u}, {1u,2u,2u,2u,2u,1u},
        {1u,2u,2u,1u,2u,1u}, {1u,2u,2u,1u,2u,2u},
        {1u,2u,2u,2u,2u,2u}, {1u,4u,2u,1u,1u,2u},
        {1u,4u,2u,3u,3u,2u}, {2u,4u,2u,2u,3u,4u},
        {2u,4u,2u,3u,3u,2u}, {2u,4u,2u,1u,3u,2u},
    };
    uint64_t rope_shapes[][4] = {
        {1u,1u,1u,2u}, {1u,1u,2u,2u}, {1u,1u,2u,4u},
        {2u,2u,3u,4u}, {2u,4u,3u,2u}, {2u,2u,3u,2u},
    };
    uint64_t hq_less_than_hkv[] = {1u,2u,4u,1u,1u,2u};
    uint64_t nondivisible_heads[] = {1u,3u,2u,1u,1u,2u};
    uint64_t outside_attention[] = {1u,2u,2u,3u,2u,1u};
    uint64_t odd_rope[] = {1u,1u,1u,3u};
    uint64_t outside_rope[] = {1u,2u,1u,2u};
    for (size_t index = 0u; index < ET_ARRAY_COUNT(attention_shapes); index++) {
      expect_capability_request(runtime, "kernel.causal-attention",
                                "causal-attention.forward", 6u,
                                attention_shapes[index], 1);
      expect_capability_request(runtime, "kernel.causal-attention",
                                "causal-attention.backward", 6u,
                                attention_shapes[index], 1);
    }
    for (size_t index = 0u; index < ET_ARRAY_COUNT(rope_shapes); index++) {
      expect_capability_request(runtime, "kernel.rope", "rope.forward", 4u,
                                rope_shapes[index], 1);
      expect_capability_request(runtime, "kernel.rope", "rope.backward", 4u,
                                rope_shapes[index], 1);
    }
    expect_capability_request(runtime, "kernel.causal-attention",
                              "causal-attention.forward", 6u,
                              hq_less_than_hkv, 0);
    expect_capability_request(runtime, "kernel.causal-attention",
                              "causal-attention.forward", 6u,
                              nondivisible_heads, 0);
    expect_capability_request(runtime, "kernel.causal-attention",
                              "causal-attention.forward", 6u,
                              outside_attention, 0);
    expect_capability_request(runtime, "kernel.rope", "rope.forward", 4u,
                              odd_rope, 0);
    expect_capability_request(runtime, "kernel.rope", "rope.forward", 4u,
                              outside_rope, 0);
  }
  {
    size_t bytes = 0u;
    et_kernel_error error;
    CHECK(et_kernel_runtime_report_json(runtime, NULL, 0u, &bytes, &error) == 0);
    char *report = (char *)malloc(bytes);
    CHECK(report != NULL);
    CHECK(et_kernel_runtime_report_json(runtime, report, bytes, &bytes, &error) ==
          0);
    CHECK(strstr(report,
        "[[1,1],[2,2],[2,2],[1,1],[1,1],[1,1]]") != NULL);
    CHECK(strstr(report,
        "[[2,2],[4,4],[2,2],[2,2],[3,3],[4,4]]") != NULL);
    CHECK(strstr(report, "[[2,2],[2,2],[3,3],[4,4]]") != NULL);
    free(report);
  }
}

static void test_mha_forward_masks(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 2u, 2u, 1u};
  uint64_t q_shape[] = {1u, 2u, 2u, 1u};
  uint64_t kv_shape[] = {1u, 2u, 2u, 1u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t mask_shape[] = {1u, 2u, 2u};
  float q[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float k[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float v[] = {2.0f, 6.0f, 10.0f, 14.0f};
  int64_t query_positions[] = {0, 1};
  int64_t key_positions[] = {0, 1};
  uint8_t mask[] = {1u, 1u, 1u, 1u};
  float output[] = {-9.0f, -9.0f, -9.0f, -9.0f};
  const float expected[] = {2.0f, 4.0f, 10.0f, 12.0f};
  float first_run[4];
  const float masked_expected[] = {0.0f, 2.0f, 0.0f, 10.0f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(query_positions, sizeof(query_positions), "i64", 2u,
           position_shape),
      view(key_positions, sizeof(key_positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 invocation = call("kernel.causal-attention", &req, inputs,
                                      6u, outputs, 1u);
  et_kernel_error error;

  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(output, expected, 4u, 1e-6f);
  memcpy(first_run, output, sizeof(first_run));
  memset(output, 0x55, sizeof(output));
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  CHECK(memcmp(output, first_run, sizeof(output)) == 0);
  /* Future keys remain excluded even when the explicit keep mask says true. */
  CHECK(output[0] == v[0] && output[2] == v[2]);

  mask[0] = 0u;
  mask[1] = 0u;
  mask[2] = 1u;
  mask[3] = 0u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(output, masked_expected, 4u, 1e-6f);
  expect_positive_zeros(&output[0], 1u);
  expect_positive_zeros(&output[2], 1u);

  memcpy(output, first_run, sizeof(output));
  query_positions[1] = 0;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, first_run, sizeof(output)) == 0);
  query_positions[1] = 1;
  key_positions[0] = -1;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, first_run, sizeof(output)) == 0);
  key_positions[0] = 0;
}

typedef struct extended_view {
  et_kernel_tensor_view_v1 prefix;
  uint64_t extension;
} extended_view;

static void test_compatible_tensor_stride(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 1u, 1u, 2u};
  uint64_t x_shape[] = {1u, 1u, 1u, 2u};
  uint64_t position_shape[] = {1u, 1u};
  uint64_t frequency_shape[] = {1u};
  float x[] = {3.0f, -4.0f};
  int64_t positions[] = {0};
  float inv_frequency[] = {1.0f};
  float output[] = {0.0f, 0.0f};
  extended_view inputs[] = {
      {.prefix = view(x, sizeof(x), "f32", 4u, x_shape), .extension = 1u},
      {.prefix = view(positions, sizeof(positions), "i64", 2u,
                      position_shape),
       .extension = 2u},
      {.prefix = view(inv_frequency, sizeof(inv_frequency), "f32", 1u,
                      frequency_shape),
       .extension = 3u},
  };
  extended_view outputs[] = {
      {.prefix = view(output, sizeof(output), "f32", 4u, x_shape),
       .extension = 4u},
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, semantic_shape);
  et_kernel_call_v1 invocation = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = "kernel.rope",
      .request = &req,
      .input_count = 3u,
      .input_stride = sizeof(extended_view),
      .input_bytes = sizeof(inputs),
      .inputs = inputs,
      .output_count = 1u,
      .output_stride = sizeof(extended_view),
      .output_bytes = sizeof(outputs),
      .outputs = outputs,
  };
  et_kernel_error error;
  for (size_t index = 0; index < 3u; index++) {
    inputs[index].prefix.struct_size = sizeof(extended_view);
  }
  outputs[0].prefix.struct_size = sizeof(extended_view);
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(output, x, 2u, 0.0f);
}

static void test_mha_backward(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 2u, 2u, 1u};
  uint64_t q_shape[] = {1u, 2u, 2u, 1u};
  uint64_t kv_shape[] = {1u, 2u, 2u, 1u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t mask_shape[] = {1u, 2u, 2u};
  float q[] = {0.0f, 0.0f, 0.0f, 0.0f};
  float k[] = {1.0f, 2.0f, 0.0f, 1.0f};
  float v[] = {3.0f, 5.0f, 7.0f, 11.0f};
  int64_t query_positions[] = {0, 1};
  int64_t key_positions[] = {0, 1};
  uint8_t mask[] = {1u, 1u, 1u, 1u};
  float upstream[] = {1.0f, 1.0f, 1.0f, 1.0f};
  float dq[4];
  float dk[4];
  float dv[4];
  const float expected_dq[] = {0.0f, 0.5f, 0.0f, 1.0f};
  const float expected_dk[] = {0.0f, 0.0f, 0.0f, 0.0f};
  const float expected_dv[] = {1.5f, 0.5f, 1.5f, 0.5f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(query_positions, sizeof(query_positions), "i64", 2u,
           position_shape),
      view(key_positions, sizeof(key_positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
      view(upstream, sizeof(upstream), "f32", 4u, q_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.backward", 6u, semantic_shape);
  et_kernel_call_v1 invocation = call("kernel.causal-attention", &req, inputs,
                                      7u, outputs, 3u);
  et_kernel_error error;

  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(dq, expected_dq, 4u, 1e-6f);
  expect_array(dk, expected_dk, 4u, 1e-6f);
  expect_array(dv, expected_dv, 4u, 1e-6f);

  /* An earliest-query-only loss cannot reach either future key/value. */
  upstream[0] = 1.0f;
  upstream[1] = 0.0f;
  upstream[2] = 1.0f;
  upstream[3] = 0.0f;
  q[0] = 0.5f;
  q[2] = -0.75f;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_positive_zeros(&dk[1], 1u);
  expect_positive_zeros(&dk[3], 1u);
  expect_positive_zeros(&dv[1], 1u);
  expect_positive_zeros(&dv[3], 1u);

  memset(mask, 0, sizeof(mask));
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_positive_zeros(dq, 4u);
  expect_positive_zeros(dk, 4u);
  expect_positive_zeros(dv, 4u);
}

static void test_gqa_forward_backward(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 4u, 2u, 1u, 1u, 2u};
  uint64_t q_shape[] = {1u, 4u, 1u, 2u};
  uint64_t kv_shape[] = {1u, 2u, 1u, 2u};
  uint64_t position_shape[] = {1u, 1u};
  uint64_t mask_shape[] = {1u, 1u, 1u};
  float q[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float k[] = {0.5f, -0.5f, 1.0f, 2.0f};
  float v[] = {10.0f, 20.0f, 30.0f, 40.0f};
  int64_t positions[] = {0};
  uint8_t mask[] = {1u};
  float output[8];
  float upstream[] = {1.0f, 2.0f, 3.0f, 4.0f,
                      5.0f, 6.0f, 7.0f, 8.0f};
  float dq[8];
  float dk[4];
  float dv[4];
  const float expected_output[] = {10.0f, 20.0f, 10.0f, 20.0f,
                                   30.0f, 40.0f, 30.0f, 40.0f};
  const float expected_dv[] = {4.0f, 6.0f, 12.0f, 14.0f};
  et_kernel_tensor_view_v1 forward_inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 forward_outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 forward_request =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 forward_call =
      call("kernel.causal-attention", &forward_request, forward_inputs, 6u,
           forward_outputs, 1u);
  et_kernel_tensor_view_v1 backward_inputs[7];
  et_kernel_tensor_view_v1 backward_outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 backward_request =
      request("causal-attention.backward", 6u, semantic_shape);
  et_kernel_call_v1 backward_call;
  et_kernel_error error;

  CHECK(et_kernel_runtime_dispatch(runtime, &forward_call, &error) == 0);
  expect_array(output, expected_output, 8u, 1e-6f);
  memcpy(backward_inputs, forward_inputs, sizeof(forward_inputs));
  backward_inputs[6] = view(upstream, sizeof(upstream), "f32", 4u, q_shape);
  backward_call = call("kernel.causal-attention", &backward_request,
                       backward_inputs, 7u, backward_outputs, 3u);
  CHECK(et_kernel_runtime_dispatch(runtime, &backward_call, &error) == 0);
  expect_positive_zeros(dq, 8u);
  expect_positive_zeros(dk, 4u);
  expect_array(dv, expected_dv, 4u, 1e-6f);
}

static void test_reciprocal_sqrt_order(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 1u, 2u, 2u};
  uint64_t q_shape[] = {1u, 2u, 1u, 2u};
  uint64_t kv_shape[] = {1u, 2u, 2u, 2u};
  uint64_t query_position_shape[] = {1u, 1u};
  uint64_t key_position_shape[] = {1u, 2u};
  uint64_t mask_shape[] = {1u, 1u, 2u};
  float q[] = {9.20307636260986328125f, 0.0f,
               9.20307636260986328125f, 0.0f};
  float k[] = {1.0f, 0.0f, 0.0f, 0.0f,
               1.0f, 0.0f, 0.0f, 0.0f};
  float v[] = {0.0f, 0.0f, 1.0f, 1.0f,
               0.0f, 0.0f, 1.0f, 1.0f};
  int64_t query_positions[] = {1};
  int64_t key_positions[] = {0, 1};
  uint8_t mask[] = {1u, 1u};
  float output[4] = {0.0f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(query_positions, sizeof(query_positions), "i64", 2u,
           query_position_shape),
      view(key_positions, sizeof(key_positions), "i64", 2u,
           key_position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 invocation = call("kernel.causal-attention", &req, inputs,
                                      6u, outputs, 1u);
  et_kernel_error error;
  const uint32_t reciprocal_then_multiply = UINT32_C(0x3ac348a4);
  const uint32_t divide_after_sum = UINT32_C(0x3ac3489e);

  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  for (size_t index = 0u; index < ET_ARRAY_COUNT(output); index++) {
    uint32_t bits;
    memcpy(&bits, &output[index], sizeof(bits));
    CHECK(bits == reciprocal_then_multiply);
    CHECK(bits != divide_after_sum);
  }
}

static float attention_loss(et_kernel_runtime *runtime,
                            et_kernel_call_v1 *invocation, float *output,
                            const float *upstream, size_t count) {
  et_kernel_error error;
  float result = 0.0f;
  CHECK(et_kernel_runtime_dispatch(runtime, invocation, &error) == 0);
  for (size_t index = 0; index < count; index++) {
    result += output[index] * upstream[index];
  }
  return result;
}

static void test_attention_finite_difference(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 2u, 2u, 2u};
  uint64_t q_shape[] = {1u, 2u, 2u, 2u};
  uint64_t kv_shape[] = {1u, 2u, 2u, 2u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t mask_shape[] = {1u, 2u, 2u};
  float q[] = {-0.3f, 0.2f, 0.5f, -0.1f, 0.4f, -0.2f, 0.1f, 0.6f};
  float k[] = {0.2f, -0.4f, 0.3f, 0.5f, -0.1f, 0.7f, 0.6f, -0.2f};
  float v[] = {0.8f, -0.3f, 0.2f, 0.9f, -0.5f, 0.4f, 0.7f, 0.1f};
  float upstream[] = {0.4f, -0.2f, 0.3f, 0.6f, -0.7f, 0.5f, 0.2f, -0.1f};
  int64_t positions[] = {0, 1};
  uint8_t mask[] = {1u, 1u, 1u, 1u};
  float output[8], dq[8], dk[8], dv[8];
  float *operands[] = {q, k, v};
  float *gradients[] = {dq, dk, dv};
  et_kernel_tensor_view_v1 forward_inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 forward_outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 forward_request =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 forward_call = call(
      "kernel.causal-attention", &forward_request, forward_inputs, 6u,
      forward_outputs, 1u);
  et_kernel_tensor_view_v1 backward_inputs[7];
  et_kernel_tensor_view_v1 backward_outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 backward_request =
      request("causal-attention.backward", 6u, semantic_shape);
  et_kernel_call_v1 backward_call;
  et_kernel_error error;
  const float epsilon = 1.0e-3f;

  memcpy(backward_inputs, forward_inputs, sizeof(forward_inputs));
  backward_inputs[6] = view(upstream, sizeof(upstream), "f32", 4u, q_shape);
  backward_call = call("kernel.causal-attention", &backward_request,
                       backward_inputs, 7u, backward_outputs, 3u);
  CHECK(et_kernel_runtime_dispatch(runtime, &backward_call, &error) == 0);
  for (size_t operand = 0; operand < 3u; operand++) {
    for (size_t index = 0; index < 8u; index++) {
      const float original = operands[operand][index];
      float positive, negative, numerical;
      operands[operand][index] = original + epsilon;
      positive = attention_loss(runtime, &forward_call, output, upstream, 8u);
      operands[operand][index] = original - epsilon;
      negative = attention_loss(runtime, &forward_call, output, upstream, 8u);
      operands[operand][index] = original;
      numerical = (positive - negative) / (2.0f * epsilon);
      CHECK(close_float(gradients[operand][index], numerical, 2.0e-3f));
    }
  }
}

static void test_n2_rectangular_attention(et_kernel_runtime *runtime) {
  enum { Q_COUNT = 64, KV_COUNT = 48 };
  uint64_t semantic_shape[] = {2u, 4u, 2u, 2u, 3u, 4u};
  uint64_t q_shape[] = {2u, 4u, 2u, 4u};
  uint64_t kv_shape[] = {2u, 2u, 3u, 4u};
  uint64_t query_position_shape[] = {2u, 2u};
  uint64_t key_position_shape[] = {2u, 3u};
  uint64_t mask_shape[] = {2u, 2u, 3u};
  float q[Q_COUNT], k[KV_COUNT], v[KV_COUNT], upstream[Q_COUNT];
  float output[Q_COUNT], repeated[Q_COUNT];
  float dq[Q_COUNT], dk[KV_COUNT], dv[KV_COUNT];
  int64_t query_positions[] = {1, 4, 2, 6};
  int64_t key_positions[] = {0, 3, 5, 1, 4, 6};
  uint8_t mask[] = {1u,1u,1u, 1u,1u,1u,
                    1u,0u,1u, 1u,1u,1u};
  et_kernel_tensor_view_v1 forward_inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(query_positions, sizeof(query_positions), "i64", 2u,
           query_position_shape),
      view(key_positions, sizeof(key_positions), "i64", 2u,
           key_position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 forward_outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 forward_request =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 forward_call = call(
      "kernel.causal-attention", &forward_request, forward_inputs, 6u,
      forward_outputs, 1u);
  et_kernel_tensor_view_v1 backward_inputs[7];
  et_kernel_tensor_view_v1 backward_outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 backward_request =
      request("causal-attention.backward", 6u, semantic_shape);
  et_kernel_call_v1 backward_call;
  et_kernel_error error;
  const size_t q_samples[] = {0u, 31u, 32u, 63u};
  const size_t kv_samples[] = {0u, 23u, 24u, 47u};
  const float epsilon = 1.0e-3f;

  for (size_t index = 0u; index < Q_COUNT; index++) {
    q[index] = (float)((int)(index % 13u) - 6) * 0.0375f;
    upstream[index] = (float)((int)(index % 9u) - 4) * 0.045f;
  }
  for (size_t index = 0u; index < KV_COUNT; index++) {
    k[index] = (float)((int)(index % 11u) - 5) * 0.0525f;
    v[index] = (float)((int)(index % 7u) - 2) * 0.0875f;
  }
  CHECK(et_kernel_runtime_dispatch(runtime, &forward_call, &error) == 0);
  memcpy(repeated, output, sizeof(output));
  memset(output, 0, sizeof(output));
  CHECK(et_kernel_runtime_dispatch(runtime, &forward_call, &error) == 0);
  CHECK(memcmp(output, repeated, sizeof(output)) == 0);
  CHECK(memcmp(&output[0], &output[Q_COUNT / 2u],
               (Q_COUNT / 2u) * sizeof(float)) != 0);

  memcpy(backward_inputs, forward_inputs, sizeof(forward_inputs));
  backward_inputs[6] = view(upstream, sizeof(upstream), "f32", 4u, q_shape);
  backward_call = call("kernel.causal-attention", &backward_request,
                       backward_inputs, 7u, backward_outputs, 3u);
  CHECK(et_kernel_runtime_dispatch(runtime, &backward_call, &error) == 0);
  for (size_t operand = 0u; operand < 3u; operand++) {
    float *values = operand == 0u ? q : (operand == 1u ? k : v);
    float *gradient = operand == 0u ? dq : (operand == 1u ? dk : dv);
    const size_t *samples = operand == 0u ? q_samples : kv_samples;
    for (size_t sample = 0u; sample < 4u; sample++) {
      const size_t index = samples[sample];
      const float original = values[index];
      float positive, negative;
      values[index] = original + epsilon;
      positive = attention_loss(runtime, &forward_call, output, upstream,
                                Q_COUNT);
      values[index] = original - epsilon;
      negative = attention_loss(runtime, &forward_call, output, upstream,
                                Q_COUNT);
      values[index] = original;
      CHECK(close_float(gradient[index],
                        (positive - negative) / (2.0f * epsilon), 4.0e-3f));
    }
  }
}

static void test_pytorch_gqa_fixture(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 4u, 2u, 3u, 3u, 2u};
  uint64_t q_shape[] = {1u, 4u, 3u, 2u};
  uint64_t kv_shape[] = {1u, 2u, 3u, 2u};
  uint64_t position_shape[] = {1u, 3u};
  uint64_t mask_shape[] = {1u, 3u, 3u};
  float q[24], k[12], v[12], upstream[24];
  int64_t positions[] = {0, 1, 2};
  uint8_t mask[] = {1u, 0u, 0u, 1u, 1u, 0u, 0u, 0u, 0u};
  float output[24], dq[24], dk[12], dv[12];
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 invocation =
      call("kernel.causal-attention", &req, inputs, 6u, outputs, 1u);
  et_kernel_tensor_view_v1 backward_inputs[7];
  et_kernel_tensor_view_v1 backward_outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_error error;
  for (size_t index = 0u; index < 24u; index++) {
    q[index] = (float)((int)(index % 7u) - 3) * 0.125f;
    upstream[index] = (float)(index % 4u + 1u) * 0.05f;
  }
  for (size_t index = 0u; index < 12u; index++) {
    k[index] = (float)((int)(index % 5u) - 2) * 0.2f;
    v[index] = (float)(index + 1u) * 0.1f;
  }
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_reference(output, et_a2_ref_gqa_output, 24u, 2.0e-5f);
  memcpy(backward_inputs, inputs, sizeof(inputs));
  backward_inputs[6] = view(upstream, sizeof(upstream), "f32", 4u, q_shape);
  req.operation = "causal-attention.backward";
  invocation = call("kernel.causal-attention", &req, backward_inputs, 7u,
                    backward_outputs, 3u);
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_reference(dq, et_a2_ref_gqa_dq, 24u, 5.0e-5f);
  expect_reference(dk, et_a2_ref_gqa_dk, 12u, 5.0e-5f);
  expect_reference(dv, et_a2_ref_gqa_dv, 12u, 5.0e-5f);
}

static void test_rope_boundary_and_backward(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 1u, 2u, 2u};
  uint64_t x_shape[] = {1u, 1u, 2u, 2u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t frequency_shape[] = {1u};
  float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
  int64_t positions[] = {0, ET_A2_MAX_EXACT_POSITION};
  float inv_frequency[] = {1.0e-7f};
  float y[4];
  float dx[4];
  float expected[4];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 4u, x_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(inv_frequency, sizeof(inv_frequency), "f32", 1u, frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(y, sizeof(y), "f32", 4u, x_shape),
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, semantic_shape);
  et_kernel_call_v1 invocation =
      call("kernel.rope", &req, inputs, 3u, outputs, 1u);
  et_kernel_error error;
  const float angle = (float)ET_A2_MAX_EXACT_POSITION * inv_frequency[0];

  expected[0] = 1.0f;
  expected[1] = 2.0f;
  expected[2] = 3.0f * cosf(angle) - 4.0f * sinf(angle);
  expected[3] = 3.0f * sinf(angle) + 4.0f * cosf(angle);
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(y, expected, 4u, 1e-6f);
  CHECK(close_float(hypotf(y[2], y[3]), 5.0f, 1e-5f));

  inputs[0] = view(y, sizeof(y), "f32", 4u, x_shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 4u, x_shape);
  req.operation = "rope.backward";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_array(dx, x, 4u, 2e-6f);
}

static void test_pytorch_rope_fixture(et_kernel_runtime *runtime) {
  uint64_t shape[] = {1u, 1u, 2u, 4u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t frequency_shape[] = {2u};
  float x[] = {1.0f, 2.0f, -3.0f, 4.0f, 0.5f, -0.25f, 2.0f, -1.0f};
  float upstream[] = {-0.5f, 0.25f, 1.5f, -2.0f,
                      0.75f, 1.25f, -0.5f, 0.125f};
  int64_t positions[] = {0, ET_A2_MAX_EXACT_POSITION};
  float inv_frequency[] = {1.0f, 0.01f};
  float output[8];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 4u, shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(inv_frequency, sizeof(inv_frequency), "f32", 1u, frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, shape),
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, shape);
  et_kernel_call_v1 invocation =
      call("kernel.rope", &req, inputs, 3u, outputs, 1u);
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_reference(output, et_a2_ref_rope_output, 8u, 5.0e-5f);
  inputs[0] = view(upstream, sizeof(upstream), "f32", 4u, shape);
  req.operation = "rope.backward";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  expect_reference(output, et_a2_ref_rope_dx, 8u, 5.0e-5f);
}

static void test_rope_finite_difference(et_kernel_runtime *runtime) {
  uint64_t shape[] = {1u, 1u, 2u, 4u};
  uint64_t position_shape[] = {1u, 2u};
  uint64_t frequency_shape[] = {2u};
  float x[] = {0.2f, -0.3f, 0.7f, 0.1f, -0.5f, 0.4f, 0.6f, -0.8f};
  float upstream[] = {0.3f, 0.2f, -0.1f, 0.6f, 0.5f, -0.4f, 0.7f, 0.1f};
  int64_t positions[] = {0, 7};
  float inv_frequency[] = {0.25f, 0.03125f};
  float output[8], dx[8];
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 4u, shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(inv_frequency, sizeof(inv_frequency), "f32", 1u, frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, shape),
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, shape);
  et_kernel_call_v1 invocation =
      call("kernel.rope", &req, inputs, 3u, outputs, 1u);
  et_kernel_error error;
  const float epsilon = 1.0e-3f;
  inputs[0] = view(upstream, sizeof(upstream), "f32", 4u, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 4u, shape);
  req.operation = "rope.backward";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  inputs[0] = view(x, sizeof(x), "f32", 4u, shape);
  outputs[0] = view(output, sizeof(output), "f32", 4u, shape);
  req.operation = "rope.forward";
  for (size_t index = 0u; index < 8u; index++) {
    const float original = x[index];
    float positive = 0.0f, negative = 0.0f;
    x[index] = original + epsilon;
    CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
    for (size_t item = 0u; item < 8u; item++) positive += output[item] * upstream[item];
    x[index] = original - epsilon;
    CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
    for (size_t item = 0u; item < 8u; item++) negative += output[item] * upstream[item];
    x[index] = original;
    CHECK(close_float(dx[index], (positive - negative) / (2.0f * epsilon),
                      1.0e-3f));
  }
}

static float rope_loss(et_kernel_runtime *runtime,
                       et_kernel_call_v1 *invocation, const float *output,
                       const float *upstream, size_t count) {
  et_kernel_error error;
  float result = 0.0f;
  CHECK(et_kernel_runtime_dispatch(runtime, invocation, &error) == 0);
  for (size_t index = 0u; index < count; index++)
    result += output[index] * upstream[index];
  return result;
}

static void test_n2_rope_forward_backward(et_kernel_runtime *runtime) {
  enum { ELEMENTS = 48 };
  uint64_t shape[] = {2u, 2u, 3u, 4u};
  uint64_t position_shape[] = {2u, 3u};
  uint64_t frequency_shape[] = {2u};
  float x[ELEMENTS], upstream[ELEMENTS], output[ELEMENTS], repeated[ELEMENTS];
  float dx[ELEMENTS];
  int64_t positions[] = {0, 3, 7, 2, 5, 9};
  float inv_frequency[] = {0.25f, 0.03125f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 4u, shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(inv_frequency, sizeof(inv_frequency), "f32", 1u, frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, shape),
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, shape);
  et_kernel_call_v1 invocation =
      call("kernel.rope", &req, inputs, 3u, outputs, 1u);
  et_kernel_error error;
  const size_t samples[] = {0u, 23u, 24u, 47u};
  const float epsilon = 1.0e-3f;

  for (size_t index = 0u; index < ELEMENTS; index++) {
    x[index] = (float)((int)(index % 17u) - 8) * 0.0625f;
    upstream[index] = (float)((int)(index % 11u) - 5) * 0.075f;
  }
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  memcpy(repeated, output, sizeof(output));
  memset(output, 0, sizeof(output));
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  CHECK(memcmp(output, repeated, sizeof(output)) == 0);
  CHECK(memcmp(&output[0], &output[ELEMENTS / 2u],
               (ELEMENTS / 2u) * sizeof(float)) != 0);
  for (size_t index = 0u; index < ELEMENTS; index += 2u) {
    CHECK(close_float(hypotf(output[index], output[index + 1u]),
                      hypotf(x[index], x[index + 1u]), 2.0e-6f));
  }

  inputs[0] = view(upstream, sizeof(upstream), "f32", 4u, shape);
  outputs[0] = view(dx, sizeof(dx), "f32", 4u, shape);
  req.operation = "rope.backward";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) == 0);
  inputs[0] = view(x, sizeof(x), "f32", 4u, shape);
  outputs[0] = view(output, sizeof(output), "f32", 4u, shape);
  req.operation = "rope.forward";
  for (size_t sample = 0u; sample < ET_ARRAY_COUNT(samples); sample++) {
    const size_t index = samples[sample];
    const float original = x[index];
    float positive, negative;
    x[index] = original + epsilon;
    positive = rope_loss(runtime, &invocation, output, upstream, ELEMENTS);
    x[index] = original - epsilon;
    negative = rope_loss(runtime, &invocation, output, upstream, ELEMENTS);
    x[index] = original;
    CHECK(close_float(dx[index], (positive - negative) / (2.0f * epsilon),
                      1.5e-3f));
  }
}

static void test_failures_are_atomic(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 1u, 1u, 1u};
  uint64_t q_shape[] = {1u, 2u, 1u, 1u};
  uint64_t kv_shape[] = {1u, 2u, 1u, 1u};
  uint64_t position_shape[] = {1u, 1u};
  uint64_t mask_shape[] = {1u, 1u, 1u};
  float q[] = {1.0f, 2.0f};
  float k[] = {1.0f, 2.0f};
  float v[] = {3.0f, 4.0f};
  int64_t query_positions[] = {0};
  int64_t key_positions[] = {0};
  uint8_t mask[] = {1u};
  float output[] = {91.0f, 92.0f};
  const float sentinel[] = {91.0f, 92.0f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(query_positions, sizeof(query_positions), "i64", 2u,
           position_shape),
      view(key_positions, sizeof(key_positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, q_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.forward", 6u, semantic_shape);
  et_kernel_call_v1 invocation = call("kernel.causal-attention", &req, inputs,
                                      6u, outputs, 1u);
  et_kernel_error error;

  q[0] = NAN;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  q[0] = 1.0f;
  query_positions[0] = ET_A2_MAX_EXACT_POSITION + 1;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  query_positions[0] = 0;
  mask[0] = 2u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  mask[0] = 1u;
  semantic_shape[2] = 1u; /* MQA is outside the verified capability range. */
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  semantic_shape[2] = 2u;
  outputs[0].data = q; /* K1 rejects output/input aliasing before provider work. */
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(error.code == ET_KERNEL_CODE_ALIASING_OUTPUT);
  CHECK(q[0] == 1.0f && q[1] == 2.0f);

  outputs[0] = view(output, sizeof(output), "f32", 4u, q_shape);
  memcpy(output, sentinel, sizeof(output));
  inputs[0].dtype = "i64";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].dtype = "f32";
  inputs[0].device = "cuda";
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].device = "cpu";
  inputs[0].layout = 99u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR;
  inputs[0].offset_bytes = 4u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].offset_bytes = 0u;
  inputs[0].rank = 3u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].rank = 4u;
  inputs[0].byte_length -= sizeof(float);
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[0].byte_length += sizeof(float);
  inputs[2].data = k;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inputs[2].data = v;
  invocation.input_count = 5u;
  invocation.input_bytes = 5u * sizeof(et_kernel_tensor_view_v1);
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  invocation.input_count = 6u;
  invocation.input_bytes = sizeof(inputs);
  invocation.input_stride = ET_KERNEL_TENSOR_VIEW_V1_0_SIZE - 1u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
}

static void test_backward_failure_atomicity(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 2u, 2u, 1u, 1u, 1u};
  uint64_t q_shape[] = {1u, 2u, 1u, 1u};
  uint64_t kv_shape[] = {1u, 2u, 1u, 1u};
  uint64_t position_shape[] = {1u, 1u};
  uint64_t mask_shape[] = {1u, 1u, 1u};
  float q[] = {FLT_MAX, 1.0f}, k[] = {2.0f, 1.0f}, v[] = {1.0f, 1.0f};
  float upstream[] = {1.0f, 1.0f};
  int64_t positions[] = {0};
  uint8_t mask[] = {1u};
  float dq[] = {11.0f, 12.0f}, dk[] = {21.0f, 22.0f};
  float dv[] = {31.0f, 32.0f};
  const float expected_dq[] = {11.0f, 12.0f};
  const float expected_dk[] = {21.0f, 22.0f};
  const float expected_dv[] = {31.0f, 32.0f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(q, sizeof(q), "f32", 4u, q_shape),
      view(k, sizeof(k), "f32", 4u, kv_shape),
      view(v, sizeof(v), "f32", 4u, kv_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(mask, sizeof(mask), "bool", 3u, mask_shape),
      view(upstream, sizeof(upstream), "f32", 4u, q_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(dq, sizeof(dq), "f32", 4u, q_shape),
      view(dk, sizeof(dk), "f32", 4u, kv_shape),
      view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 req =
      request("causal-attention.backward", 6u, semantic_shape);
  et_kernel_call_v1 invocation =
      call("kernel.causal-attention", &req, inputs, 7u, outputs, 3u);
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(dq, expected_dq, sizeof(dq)) == 0);
  CHECK(memcmp(dk, expected_dk, sizeof(dk)) == 0);
  CHECK(memcmp(dv, expected_dv, sizeof(dv)) == 0);
  q[0] = 1.0f;
  outputs[1].data = dq;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(dq, expected_dq, sizeof(dq)) == 0);
  CHECK(memcmp(dk, expected_dk, sizeof(dk)) == 0);
  CHECK(memcmp(dv, expected_dv, sizeof(dv)) == 0);
  outputs[1].data = dk;
  outputs[0].data = q;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(q[0] == 1.0f && q[1] == 1.0f);
  CHECK(memcmp(dk, expected_dk, sizeof(dk)) == 0);
  CHECK(memcmp(dv, expected_dv, sizeof(dv)) == 0);
}

static void test_rope_failures_are_atomic(et_kernel_runtime *runtime) {
  uint64_t semantic_shape[] = {1u, 1u, 1u, 2u};
  uint64_t x_shape[] = {1u, 1u, 1u, 2u};
  uint64_t position_shape[] = {1u, 1u};
  uint64_t frequency_shape[] = {1u};
  float x[] = {1.0f, 2.0f};
  int64_t positions[] = {0};
  float inv_frequency[] = {1.0f};
  float output[] = {81.0f, 82.0f};
  const float sentinel[] = {81.0f, 82.0f};
  et_kernel_tensor_view_v1 inputs[] = {
      view(x, sizeof(x), "f32", 4u, x_shape),
      view(positions, sizeof(positions), "i64", 2u, position_shape),
      view(inv_frequency, sizeof(inv_frequency), "f32", 1u, frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[] = {
      view(output, sizeof(output), "f32", 4u, x_shape),
  };
  et_kernel_request_v1 req = request("rope.forward", 4u, semantic_shape);
  et_kernel_call_v1 invocation =
      call("kernel.rope", &req, inputs, 3u, outputs, 1u);
  et_kernel_error error;

  inv_frequency[0] = 0.0f;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inv_frequency[0] = 1.0f;
  positions[0] = -1;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  positions[0] = 0;
  inv_frequency[0] = INFINITY;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  inv_frequency[0] = 1.0f;
  positions[0] = ET_A2_MAX_EXACT_POSITION + 1;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  positions[0] = 0;
  semantic_shape[3] = 3u;
  CHECK(et_kernel_runtime_dispatch(runtime, &invocation, &error) != 0);
  CHECK(memcmp(output, sentinel, sizeof(output)) == 0);
  semantic_shape[3] = 2u;
}

int main(void) {
  et_kernel_runtime *runtime = make_runtime();
  test_metadata(runtime);
  test_mha_forward_masks(runtime);
  test_mha_backward(runtime);
  test_gqa_forward_backward(runtime);
  test_reciprocal_sqrt_order(runtime);
  test_attention_finite_difference(runtime);
  test_n2_rectangular_attention(runtime);
  test_pytorch_gqa_fixture(runtime);
  test_rope_boundary_and_backward(runtime);
  test_pytorch_rope_fixture(runtime);
  test_rope_finite_difference(runtime);
  test_n2_rope_forward_backward(runtime);
  test_compatible_tensor_stride(runtime);
  test_failures_are_atomic(runtime);
  test_backward_failure_atomicity(runtime);
  test_rope_failures_are_atomic(runtime);
  et_kernel_runtime_destroy(runtime);
  (void)printf("A2 provider PASS (%zu checks)\n", checks);
  return 0;
}
