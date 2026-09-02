#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/a2_kv_cache.h"
#include "reference_vectors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { N = 2, HQ = 4, HKV = 2, C = 3, DH = 2, L = 1 };
#define CACHE_PARITY_ABSOLUTE_TOLERANCE 2.0e-6f
#define CACHE_ORACLE_ABSOLUTE_TOLERANCE 2.0e-5f
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
      tensor_view((void *)input,N*heads*C*DH*sizeof(float),"f32",4u,tensor_shape),
      tensor_view((void *)positions,N*C*sizeof(int64_t),"i64",2u,
                  position_shape),
      tensor_view((void *)inverse_frequency,sizeof(inverse_frequency),"f32",1u,
                  frequency_shape),
  };
  et_kernel_tensor_view_v1 outputs[1] = {
      tensor_view(output,N*heads*C*DH*sizeof(float),"f32",4u,tensor_shape),
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

static float reference_float(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

/* Stores the full output followed by every incremental one-query output. */
static void run_scenario(float result[96]) {
  float q[N * HQ * C * DH];
  float k[N * HKV * C * DH];
  float v[N * HKV * C * DH];
  int64_t positions[N * C];
  uint8_t full_mask[N * C * C];
  uint64_t full_semantic[6] = {N,HQ,HKV,C,C,DH};
  uint64_t full_tensor_shape[4] = {N,HQ,C,DH};
  uint64_t full_kv_shape[4] = {N,HKV,C,DH};
  uint64_t full_position_shape[2] = {N,C};
  uint64_t full_mask_shape[3] = {N,C,C};
  float q_rope[N * HQ * C * DH], k_rope[N * HKV * C * DH];
  float full_output[N * HQ * C * DH];
  et_kernel_runtime *runtime = make_runtime();

  for (size_t n = 0u; n < N; n++) {
    for (size_t token = 0u; token < C; token++) {
      positions[n*C+token] = (int64_t)(n * 10u + token);
      for (size_t key = 0u; key < C; key++)
        full_mask[(n*C+token)*C+key] = (uint8_t)(key <= token);
    }
    for (size_t head = 0u; head < HQ; head++)
      for (size_t token = 0u; token < C; token++)
        for (size_t d = 0u; d < DH; d++) {
          size_t index = (((n*HQ+head)*C+token)*DH+d);
          q[index] = (float)((int)((index + 3u*n) % 17u) - 8) * 0.055f;
        }
    for (size_t head = 0u; head < HKV; head++)
      for (size_t token = 0u; token < C; token++)
        for (size_t d = 0u; d < DH; d++) {
          size_t index = (((n*HKV+head)*C+token)*DH+d);
          k[index] = (float)((int)((index + 5u*n) % 13u) - 6) * 0.0475f;
          v[index] = (float)((int)((index + 7u*n) % 11u) - 4) * 0.0825f;
        }
  }

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
  for (size_t n = 0u; n < N; n++)
    for (size_t token = 0u; token < C; token++)
      for (size_t head = 0u; head < HQ; head++)
        for (size_t d = 0u; d < DH; d++) {
          size_t actual_index = (((n*HQ+head)*C+token)*DH+d);
          size_t oracle_index = (((n*C+token)*HQ+head)*DH+d);
          float expected = reference_float(
              et_a2_ref_n2_cache_incremental[oracle_index]);
          CHECK(fabsf(full_output[actual_index]-expected) <=
                CACHE_ORACLE_ABSOLUTE_TOLERANCE);
        }
  memcpy(result,full_output,sizeof(full_output));
  CHECK(et_a2_kv_cache_create_v1(L,N,HKV,C,DH,&cache,&error) == 0);

  for (size_t token = 0u; token < C; token++) {
    uint64_t count_shape[1] = {N};
    uint64_t source_shape[4] = {N,HKV,1u,DH};
    uint64_t query_shape[4] = {N,HQ,1u,DH};
    uint64_t query_position_shape[2] = {N,1u};
    uint64_t step_mask_shape[3] = {N,1u,C};
    uint64_t step_semantic[6] = {N,HQ,HKV,1u,C,DH};
    int64_t append_count[N];
    int64_t query_position[N];
    float staged_k[N * HKV * DH], staged_v[N * HKV * DH];
    float query[N * HQ * DH];
    float step_output[N * HQ * DH];
    uint8_t step_mask[N * C];
    et_a2_kv_cache_transaction *transaction = NULL;
    et_a2_kv_cache_transaction_view *transaction_view = NULL;
    et_a2_kv_cache_read_borrow *borrow = NULL;
    const et_kernel_tensor_view_v1 *cached_k = NULL, *cached_v = NULL;
    const et_kernel_tensor_view_v1 *effective_lengths = NULL;
    const et_kernel_tensor_view_v1 *cache_mask = NULL;
    et_kernel_tensor_view_v1 count_view = tensor_view(
        append_count,sizeof(append_count),"i64",1u,count_shape);

    for (size_t n = 0u; n < N; n++) {
      append_count[n] = 1;
      query_position[n] = positions[n*C+token];
      for (size_t head = 0u; head < HKV; head++)
        for (size_t d = 0u; d < DH; d++) {
          size_t source = (((n*HKV+head)*C+token)*DH+d);
          size_t staged = ((n*HKV+head)*DH+d);
          staged_k[staged] = k_rope[source];
          staged_v[staged] = v[source];
        }
      for (size_t head = 0u; head < HQ; head++)
        for (size_t d = 0u; d < DH; d++)
          query[(n*HQ+head)*DH+d] =
              q_rope[((n*HQ+head)*C+token)*DH+d];
    }
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
    for (size_t n = 0u; n < N; n++)
      CHECK(((const int64_t *)effective_lengths->data)[n] ==
            (int64_t)token+1);
    CHECK(cache_mask->rank == 2u && cache_mask->shape[0] == N);
    CHECK(cache_mask->shape[1] == C && cache_mask->byte_length == N*C);
    CHECK(strcmp(cache_mask->dtype,"bool") == 0);
    CHECK(strcmp(cache_mask->device,"cpu") == 0);
    CHECK(cache_mask->layout == ET_KERNEL_LAYOUT_DENSE_ROW_MAJOR);
    CHECK(cache_mask->offset_bytes == 0u);
    for (size_t n = 0u; n < N; n++)
      for (size_t position = 0u; position < C; position++) {
        size_t mask_index = n*C+position;
        uint8_t expected = (uint8_t)(position <= token);
        CHECK(((const uint8_t *)cache_mask->data)[mask_index] == expected);
        step_mask[mask_index] =
            ((const uint8_t *)cache_mask->data)[mask_index];
        for (size_t head = 0u; head < HKV; head++)
          for (size_t d = 0u; d < DH; d++) {
            size_t index = (((n*HKV+head)*C+position)*DH+d);
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
    for (size_t n = 0u; n < N; n++)
      for (size_t head = 0u; head < HQ; head++)
        for (size_t d = 0u; d < DH; d++) {
          float expected = full_output[((n*HQ+head)*C+token)*DH+d];
          float actual = step_output[(n*HQ+head)*DH+d];
          size_t incremental =
              N*HQ*C*DH + ((n*C+token)*HQ+head)*DH+d;
          size_t oracle_index = ((n*C+token)*HQ+head)*DH+d;
          float oracle = reference_float(
              et_a2_ref_n2_cache_incremental[oracle_index]);
          CHECK(fabsf(actual-expected) <= CACHE_PARITY_ABSOLUTE_TOLERANCE);
          CHECK(fabsf(actual-oracle) <= CACHE_ORACLE_ABSOLUTE_TOLERANCE);
          result[incremental] = actual;
        }
    CHECK(et_a2_kv_cache_transaction_view_end_v1(
        &transaction_view,&error) == 0);
    CHECK(et_a2_kv_cache_transaction_commit_v1(&transaction,&error) == 0);

    cached_k = cached_v = effective_lengths = cache_mask = NULL;
    CHECK(et_a2_kv_cache_read_borrow_begin_v1(cache,&borrow,&error) == 0);
    CHECK(et_a2_kv_cache_read_borrow_layer_v1(
        borrow,0u,&cached_k,&cached_v,&effective_lengths,&cache_mask,&error) == 0);
    for (size_t n = 0u; n < N; n++)
      CHECK(((const int64_t *)effective_lengths->data)[n] ==
            (int64_t)token+1);
    CHECK(memcmp(cache_mask->data,step_mask,N*C) == 0);
    CHECK(et_a2_kv_cache_read_borrow_end_v1(&borrow,&error) == 0);
  }
  CHECK(et_a2_kv_cache_destroy_v1(&cache,&error) == 0);
  et_kernel_runtime_destroy(runtime);
}

int main(void) {
  float first[96], second[96];
  run_scenario(first);
  run_scenario(second);
  CHECK(memcmp(first,second,sizeof(first)) == 0);
  (void)printf("A2 CACHE ATTENTION PASS: %zu incremental/full parity, mask, tail, and determinism checks\n",
               checks);
  return 0;
}
