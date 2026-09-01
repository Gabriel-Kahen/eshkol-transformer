#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/a2_kv_cache.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 1, HQ = 4, HKV = 2, C = 3, DH = 2, L = 1 };
#define CACHE_PARITY_ABSOLUTE_TOLERANCE 2.0e-6f
static size_t checks;

#define CHECK(condition)                                                       \
  do {                                                                         \
    checks++;                                                                  \
    if (!(condition)) {                                                        \
      (void)fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,           \
                    #condition);                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

static et_kernel_tensor_view_v1 tensor_view(void *data, size_t bytes,
                                             const char *dtype, size_t rank,
                                             const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(result),
      .data = data,
      .byte_length = bytes,
      .dtype = dtype,
      .device = "cpu",
      .layout = ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR,
      .offset_bytes = 0u,
      .rank = rank,
      .shape = shape,
  };
  return result;
}

static et_kernel_request_v1 attention_request(const uint64_t *shape) {
  et_kernel_request_v1 result = {
      .struct_size = sizeof(result),
      .operation = "causal-attention.forward",
      .dtype = "f32",
      .device = "cpu",
      .rank = 6u,
      .shape = shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  return result;
}

static et_kernel_call_v1 attention_call(
    const et_kernel_request_v1 *request,
    et_kernel_tensor_view_v1 inputs[6], et_kernel_tensor_view_v1 outputs[1]) {
  et_kernel_call_v1 result = {
      .struct_size = sizeof(result),
      .capability = "kernel.causal-attention",
      .request = request,
      .input_count = 6u,
      .input_stride = sizeof(et_kernel_tensor_view_v1),
      .input_bytes = 6u * sizeof(et_kernel_tensor_view_v1),
      .inputs = inputs,
      .output_count = 1u,
      .output_stride = sizeof(et_kernel_tensor_view_v1),
      .output_bytes = sizeof(et_kernel_tensor_view_v1),
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

static void dispatch_attention(et_kernel_runtime *runtime,
                               const uint64_t semantic_shape[6],
                               et_kernel_tensor_view_v1 inputs[6],
                               et_kernel_tensor_view_v1 outputs[1]) {
  et_kernel_request_v1 request = attention_request(semantic_shape);
  et_kernel_call_v1 call = attention_call(&request, inputs, outputs);
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime, &call, &error) == 0);
}

static void dispatch_rope(et_kernel_runtime *runtime, const float *input,
                          float *output, uint64_t heads,
                          const int64_t positions[C]) {
  static const float inverse_frequency[1] = {0.125f};
  uint64_t semantic_shape[4] = {N,heads,C,DH};
  uint64_t tensor_shape[4] = {N,heads,C,DH};
  uint64_t position_shape[2] = {N,C};
  uint64_t frequency_shape[1] = {DH/2u};
  et_kernel_request_v1 request = {
      .struct_size = sizeof(request),
      .operation = "rope.forward",
      .dtype = "f32",
      .device = "cpu",
      .rank = 4u,
      .shape = semantic_shape,
      .deterministic = 1u,
      .reserved = {0},
  };
  et_kernel_tensor_view_v1 inputs[3] = {
      tensor_view((void *)input,heads*C*DH*sizeof(float),"f32",4u,tensor_shape),
      tensor_view((void *)positions,C*sizeof(int64_t),"i64",2u,
                  position_shape),
      tensor_view((void *)inverse_frequency,sizeof(inverse_frequency),"f32",1u,
                  frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[1] = {
      tensor_view(output,heads*C*DH*sizeof(float),"f32",4u,tensor_shape),
  };
  et_kernel_call_v1 call = {
      .struct_size = sizeof(call),
      .capability = "kernel.rope",
      .request = &request,
      .input_count = 3u,
      .input_stride = sizeof(et_kernel_tensor_view_v1),
      .input_bytes = sizeof(inputs),
      .inputs = inputs,
      .output_count = 1u,
      .output_stride = sizeof(et_kernel_tensor_view_v1),
      .output_bytes = sizeof(outputs),
      .outputs = outputs,
  };
  et_kernel_error error;
  CHECK(et_kernel_runtime_dispatch(runtime,&call,&error) == 0);
}

static void require_positive_zero(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  CHECK(bits == UINT32_C(0));
}

/* Stores full output followed by three incremental one-query outputs. */
static void run_scenario(float result[48]) {
  static const float q[HQ * C * DH] = {
      0.20f, -0.10f, 0.35f, 0.15f, -0.25f, 0.40f,
      0.10f, 0.30f, -0.45f, 0.20f, 0.50f, -0.05f,
      -0.30f, 0.20f, 0.15f, -0.50f, 0.45f, 0.35f,
      0.25f, -0.40f, 0.55f, 0.10f, -0.05f, 0.60f,
  };
  static const float k[HKV * C * DH] = {
      0.30f, -0.20f, 0.10f, 0.45f, -0.35f, 0.25f,
      0.40f, 0.05f, -0.15f, 0.30f, 0.20f, -0.40f,
  };
  static const float v[HKV * C * DH] = {
      0.50f, -0.25f, 0.20f, 0.75f, -0.40f, 0.10f,
      -0.10f, 0.60f, 0.80f, -0.30f, 0.25f, 0.45f,
  };
  static const int64_t positions[C] = {0, 1, 2};
  static const uint8_t full_mask[C * C] = {
      1u, 0u, 0u,
      1u, 1u, 0u,
      1u, 1u, 1u,
  };
  uint64_t full_semantic[6] = {N,HQ,HKV,C,C,DH};
  uint64_t full_tensor_shape[4] = {N,HQ,C,DH};
  uint64_t full_kv_shape[4] = {N,HKV,C,DH};
  uint64_t full_position_shape[2] = {N,C};
  uint64_t full_mask_shape[3] = {N,C,C};
  float q_rope[HQ * C * DH], k_rope[HKV * C * DH];
  float full_output[HQ * C * DH];
  et_kernel_runtime *runtime = make_runtime();

  dispatch_rope(runtime,q,q_rope,HQ,positions);
  dispatch_rope(runtime,k,k_rope,HKV,positions);
  et_kernel_tensor_view_v1 full_inputs[6] = {
      tensor_view(q_rope,sizeof(q_rope),"f32",4u,full_tensor_shape),
      tensor_view(k_rope,sizeof(k_rope),"f32",4u,full_kv_shape),
      tensor_view((void *)v,sizeof(v),"f32",4u,full_kv_shape),
      tensor_view((void *)positions,sizeof(positions),"i64",2u,
                  full_position_shape),
      tensor_view((void *)positions,sizeof(positions),"i64",2u,
                  full_position_shape),
      tensor_view((void *)full_mask,sizeof(full_mask),"bool",3u,
                  full_mask_shape),
  };
  et_kernel_tensor_view_v1 full_outputs[1] = {
      tensor_view(full_output,sizeof(full_output),"f32",4u,
                  full_tensor_shape),
  };
  et_a2_kv_cache *cache = NULL;
  et_kernel_error error;

  dispatch_attention(runtime,full_semantic,full_inputs,full_outputs);
  memcpy(result,full_output,sizeof(full_output));
  CHECK(et_a2_kv_cache_create_v1(L,N,HKV,C,DH,&cache,&error) == 0);

  for (size_t token = 0u; token < C; token++) {
    uint64_t count_shape[1] = {N};
    uint64_t source_shape[4] = {N,HKV,1u,DH};
    uint64_t query_shape[4] = {N,HQ,1u,DH};
    uint64_t query_position_shape[2] = {N,1u};
    uint64_t step_mask_shape[3] = {N,1u,C};
    uint64_t step_semantic[6] = {N,HQ,HKV,1u,C,DH};
    int64_t append_count[1] = {1};
    int64_t query_position[1] = {(int64_t)token};
    float staged_k[HKV * DH], staged_v[HKV * DH], query[HQ * DH];
    float step_output[HQ * DH];
    uint8_t step_mask[C];
    et_a2_kv_cache_transaction *transaction = NULL;
    et_a2_kv_cache_transaction_view *transaction_view = NULL;
    et_a2_kv_cache_read_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *cached_k = NULL, *cached_v = NULL;
    const et_kernel_tensor_view_v1 *effective_lengths = NULL;
    const et_kernel_tensor_view_v1 *cache_mask = NULL;
    et_kernel_tensor_view_v1 count_view = tensor_view(
        append_count,sizeof(append_count),"i64",1u,count_shape);

    for (size_t head = 0u; head < HKV; head++)
      for (size_t d = 0u; d < DH; d++) {
        staged_k[head*DH+d] = k_rope[(head*C+token)*DH+d];
        staged_v[head*DH+d] = v[(head*C+token)*DH+d];
      }
    for (size_t head = 0u; head < HQ; head++)
      for (size_t d = 0u; d < DH; d++)
        query[head*DH+d] = q_rope[(head*C+token)*DH+d];
    et_kernel_tensor_view_v1 staged_key_view = tensor_view(
        staged_k,sizeof(staged_k),"f32",4u,source_shape);
    et_kernel_tensor_view_v1 staged_value_view = tensor_view(
        staged_v,sizeof(staged_v),"f32",4u,source_shape);

    CHECK(et_a2_kv_cache_transaction_begin_v1(
        cache,1u,&count_view,&transaction,&error) == 0);
    CHECK(et_a2_kv_cache_transaction_stage_layer_v1(
        transaction,0u,&staged_key_view,&staged_value_view,&error) == 0);
    CHECK(et_a2_kv_cache_transaction_view_begin_v1(
        transaction,0u,&transaction_view,&error) == 0);
    CHECK(et_a2_kv_cache_transaction_view_tensors_v1(
        transaction_view,&cached_k,&cached_v,&effective_lengths,&cache_mask,
        &error) == 0);
    CHECK(cached_k->shape[0] == N && cached_k->shape[1] == HKV);
    CHECK(cached_k->shape[2] == C && cached_k->shape[3] == DH);
    CHECK(cached_v->byte_length == sizeof(k));
    CHECK(((const int64_t *)effective_lengths->data)[0] == (int64_t)token+1);
    CHECK(cache_mask->rank == 2u && cache_mask->shape[0] == N);
    CHECK(cache_mask->shape[1] == C && cache_mask->byte_length == C);
    CHECK(strcmp(cache_mask->dtype,"bool") == 0);
    CHECK(strcmp(cache_mask->device,"cpu") == 0);
    CHECK(cache_mask->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
    CHECK(cache_mask->offset_bytes == 0u);
    for (size_t position = 0u; position < C; position++) {
      uint8_t expected = (uint8_t)(position <= token);
      CHECK(((const uint8_t *)cache_mask->data)[position] == expected);
      step_mask[position] = ((const uint8_t *)cache_mask->data)[position];
      for (size_t head = 0u; head < HKV; head++)
        for (size_t d = 0u; d < DH; d++) {
          size_t index = (head*C+position)*DH+d;
          CHECK(isfinite(((const float *)cached_k->data)[index]));
          CHECK(isfinite(((const float *)cached_v->data)[index]));
          if (position > token) {
            require_positive_zero(((const float *)cached_k->data)[index]);
            require_positive_zero(((const float *)cached_v->data)[index]);
          }
        }
    }

    et_kernel_tensor_view_v1 step_inputs[6] = {
        tensor_view(query,sizeof(query),"f32",4u,query_shape),
        *cached_k,
        *cached_v,
        tensor_view(query_position,sizeof(query_position),"i64",2u,
                    query_position_shape),
        tensor_view((void *)positions,sizeof(positions),"i64",2u,
                    full_position_shape),
        tensor_view(step_mask,sizeof(step_mask),"bool",3u,step_mask_shape),
    };
    et_kernel_tensor_view_v1 step_outputs[1] = {
        tensor_view(step_output,sizeof(step_output),"f32",4u,query_shape),
    };
    dispatch_attention(runtime,step_semantic,step_inputs,step_outputs);
    for (size_t head = 0u; head < HQ; head++)
      for (size_t d = 0u; d < DH; d++) {
        float expected = full_output[(head*C+token)*DH+d];
        float actual = step_output[head*DH+d];
        CHECK(fabsf(actual-expected) <= CACHE_PARITY_ABSOLUTE_TOLERANCE);
        result[HQ*C*DH + token*HQ*DH + head*DH+d] = actual;
      }
    CHECK(et_a2_kv_cache_transaction_view_end_v1(
        &transaction_view,&error) == 0);
    CHECK(et_a2_kv_cache_transaction_commit_v1(&transaction,&error) == 0);

    cached_k = cached_v = effective_lengths = cache_mask = NULL;
    CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error) == 0);
    CHECK(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0u,&cached_k,&cached_v,&effective_lengths,&cache_mask,&error) == 0);
    CHECK(((const int64_t *)effective_lengths->data)[0] == (int64_t)token+1);
    CHECK(memcmp(cache_mask->data,step_mask,C) == 0);
    CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error) == 0);
  }
  CHECK(et_a2_kv_cache_destroy_v1(&cache,&error) == 0);
  et_kernel_runtime_destroy(runtime);
}

int main(void) {
  float first[48], second[48];
  run_scenario(first);
  run_scenario(second);
  CHECK(memcmp(first,second,sizeof(first)) == 0);
  (void)printf("A2 CACHE ATTENTION PASS: %zu incremental/full parity, mask, tail, and determinism checks\n",
               checks);
  return 0;
}
