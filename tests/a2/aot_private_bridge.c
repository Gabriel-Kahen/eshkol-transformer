#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/a2_kv_cache.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

/* Test-only fixed-buffer transport.  This is not a tensor carrier API. */

static et_kernel_tensor_view_v1 make_view(void *data, size_t bytes,
                                          const char *dtype, size_t rank,
                                          const uint64_t *shape) {
  et_kernel_tensor_view_v1 result = {
      .struct_size = sizeof(et_kernel_tensor_view_v1),
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

static et_kernel_request_v1 make_request(const char *operation, size_t rank,
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

static et_kernel_call_v1 make_call(
    const char *capability, const et_kernel_request_v1 *request,
    et_kernel_tensor_view_v1 *inputs, size_t input_count,
    et_kernel_tensor_view_v1 *outputs, size_t output_count) {
  et_kernel_call_v1 result = {
      .struct_size = sizeof(et_kernel_call_v1),
      .capability = capability,
      .request = request,
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
  if (symbol == NULL || strcmp(symbol, ET_KERNEL_PROVIDER_SYMBOL_V1) != 0) {
    return NULL;
  }
  return et_a2_kernel_provider_v1();
}

static int close_float(float actual, float expected) {
  return isfinite(actual) && fabsf(actual - expected) <= 2.0e-5f;
}

int64_t et_a2_test_provider_transport_v1(void) {
  et_kernel_runtime *runtime = NULL;
  et_kernel_error error;
  uint64_t attention_shape[] = {1u, 2u, 2u, 1u, 2u, 1u};
  uint64_t q_shape[] = {1u, 2u, 1u, 1u};
  uint64_t kv_shape[] = {1u, 2u, 2u, 1u};
  uint64_t query_position_shape[] = {1u, 1u};
  uint64_t key_position_shape[] = {1u, 2u};
  uint64_t mask_shape[] = {1u, 1u, 2u};
  float q[] = {0.0f, 0.0f};
  float k[] = {1.0f, 2.0f, 3.0f, 4.0f};
  float v[] = {2.0f, 6.0f, 10.0f, 14.0f};
  int64_t query_positions[] = {1};
  int64_t key_positions[] = {0, 1};
  uint8_t mask[] = {1u, 1u};
  float attention_output[2] = {0.0f, 0.0f};
  float upstream[] = {1.0f, 1.0f};
  float dq[2] = {0.0f, 0.0f};
  float dk[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float dv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  et_kernel_tensor_view_v1 attention_inputs[] = {
      make_view(q, sizeof(q), "f32", 4u, q_shape),
      make_view(k, sizeof(k), "f32", 4u, kv_shape),
      make_view(v, sizeof(v), "f32", 4u, kv_shape),
      make_view(query_positions, sizeof(query_positions), "i64", 2u,
                query_position_shape),
      make_view(key_positions, sizeof(key_positions), "i64", 2u,
                key_position_shape),
      make_view(mask, sizeof(mask), "bool", 3u, mask_shape),
  };
  et_kernel_tensor_view_v1 attention_outputs[] = {
      make_view(attention_output, sizeof(attention_output), "f32", 4u,
                q_shape),
  };
  et_kernel_request_v1 attention_request =
      make_request("causal-attention.forward", 6u, attention_shape);
  et_kernel_call_v1 attention_call = make_call(
      "kernel.causal-attention", &attention_request, attention_inputs, 6u,
      attention_outputs, 1u);
  et_kernel_tensor_view_v1 backward_inputs[7];
  et_kernel_tensor_view_v1 backward_outputs[] = {
      make_view(dq, sizeof(dq), "f32", 4u, q_shape),
      make_view(dk, sizeof(dk), "f32", 4u, kv_shape),
      make_view(dv, sizeof(dv), "f32", 4u, kv_shape),
  };
  et_kernel_request_v1 backward_request =
      make_request("causal-attention.backward", 6u, attention_shape);
  et_kernel_call_v1 backward_call;
  uint64_t rope_shape[] = {1u, 1u, 1u, 2u};
  uint64_t rope_position_shape[] = {1u, 1u};
  uint64_t frequency_shape[] = {1u};
  float x[] = {1.0f, 2.0f};
  int64_t rope_positions[] = {1};
  float inv_frequency[] = {1.57079632679489661923f};
  float y[2] = {0.0f, 0.0f};
  float dx[2] = {0.0f, 0.0f};
  et_kernel_tensor_view_v1 rope_inputs[] = {
      make_view(x, sizeof(x), "f32", 4u, rope_shape),
      make_view(rope_positions, sizeof(rope_positions), "i64", 2u,
                rope_position_shape),
      make_view(inv_frequency, sizeof(inv_frequency), "f32", 1u,
                frequency_shape),
  };
  et_kernel_tensor_view_v1 rope_outputs[] = {
      make_view(y, sizeof(y), "f32", 4u, rope_shape),
  };
  et_kernel_request_v1 rope_request =
      make_request("rope.forward", 4u, rope_shape);
  et_kernel_call_v1 rope_call =
      make_call("kernel.rope", &rope_request, rope_inputs, 3u, rope_outputs,
                1u);
  const et_kernel_provider_v1 *provider = et_a2_kernel_provider_v1();

  if (provider == NULL || provider->capability_count != 2u ||
      et_kernel_runtime_discover(resolve_a2, NULL, &runtime, &error) != 0 ||
      runtime == NULL ||
      et_kernel_runtime_dispatch(runtime, &attention_call, &error) != 0 ||
      !close_float(attention_output[0], 4.0f) ||
      !close_float(attention_output[1], 12.0f)) {
    et_kernel_runtime_destroy(runtime);
    return 0;
  }

  memcpy(backward_inputs, attention_inputs, sizeof(attention_inputs));
  backward_inputs[6] =
      make_view(upstream, sizeof(upstream), "f32", 4u, q_shape);
  backward_call = make_call("kernel.causal-attention", &backward_request,
                            backward_inputs, 7u, backward_outputs, 3u);
  if (et_kernel_runtime_dispatch(runtime, &backward_call, &error) != 0 ||
      !close_float(dq[0], 1.0f) || !close_float(dq[1], 1.0f) ||
      !close_float(dk[0], 0.0f) || !close_float(dk[1], 0.0f) ||
      !close_float(dk[2], 0.0f) || !close_float(dk[3], 0.0f) ||
      !close_float(dv[0], 0.5f) || !close_float(dv[1], 0.5f) ||
      !close_float(dv[2], 0.5f) || !close_float(dv[3], 0.5f) ||
      et_kernel_runtime_dispatch(runtime, &rope_call, &error) != 0 ||
      !close_float(y[0], -2.0f) || !close_float(y[1], 1.0f)) {
    et_kernel_runtime_destroy(runtime);
    return 0;
  }

  rope_inputs[0] = make_view(y, sizeof(y), "f32", 4u, rope_shape);
  rope_outputs[0] = make_view(dx, sizeof(dx), "f32", 4u, rope_shape);
  rope_request.operation = "rope.backward";
  if (et_kernel_runtime_dispatch(runtime, &rope_call, &error) != 0 ||
      !close_float(dx[0], x[0]) || !close_float(dx[1], x[1])) {
    et_kernel_runtime_destroy(runtime);
    return 0;
  }
  et_kernel_runtime_destroy(runtime);
  return 1;
}

int64_t et_a2_test_cache_transport_v1(void) {
  et_a2_kv_cache *cache = NULL;
  et_a2_kv_cache_transaction *transaction = NULL;
  et_a2_kv_cache_transaction_view *transaction_view = NULL;
  et_a2_kv_cache_read_borrow *borrow = NULL;
  const et_kernel_tensor_view_v1 *keys_view = NULL;
  const et_kernel_tensor_view_v1 *values_view = NULL;
  const et_kernel_tensor_view_v1 *lengths_view = NULL;
  const et_kernel_tensor_view_v1 *mask_view = NULL;
  et_kernel_error error;
  uint64_t count_shape[] = {1u};
  uint64_t source_shape[] = {1u, 2u, 1u, 1u};
  int64_t count_data[] = {1};
  float key_data[] = {7.0f, 8.0f};
  float value_data[] = {9.0f, 10.0f};
  et_kernel_tensor_view_v1 counts =
      make_view(count_data, sizeof(count_data), "i64", 1u, count_shape);
  et_kernel_tensor_view_v1 keys =
      make_view(key_data, sizeof(key_data), "f32", 4u, source_shape);
  et_kernel_tensor_view_v1 values =
      make_view(value_data, sizeof(value_data), "f32", 4u, source_shape);
  int success = 0;

  if (et_a2_kv_cache_create_v1(1u, 1u, 2u, 2u, 1u, &cache, &error) != 0 ||
      et_a2_kv_cache_transaction_begin_v1(cache, 1u, &counts, &transaction,
                                          &error) != 0 ||
      et_a2_kv_cache_transaction_stage_layer_v1(
          transaction, 0u, &keys, &values, &error) != 0 ||
      et_a2_kv_cache_transaction_view_begin_v1(
          transaction, 0u, &transaction_view, &error) != 0 ||
      et_a2_kv_cache_transaction_view_tensors_v1(
          transaction_view, &keys_view, &values_view, &lengths_view,
          &mask_view, &error) != 0 ||
      keys_view == NULL || values_view == NULL || lengths_view == NULL ||
      mask_view == NULL || keys_view->rank != 4u ||
      keys_view->shape[0] != 1u || keys_view->shape[1] != 2u ||
      keys_view->shape[2] != 2u || keys_view->shape[3] != 1u ||
      ((const float *)keys_view->data)[0] != 7.0f ||
      ((const float *)keys_view->data)[2] != 8.0f ||
      ((const float *)values_view->data)[0] != 9.0f ||
      ((const float *)values_view->data)[2] != 10.0f ||
      ((const int64_t *)lengths_view->data)[0] != 1 ||
      mask_view->rank != 2u || ((const uint8_t *)mask_view->data)[0] != 1u ||
      ((const uint8_t *)mask_view->data)[1] != 0u ||
      et_a2_kv_cache_transaction_view_end_v1(&transaction_view, &error) != 0 ||
      et_a2_kv_cache_transaction_commit_v1(&transaction, &error) != 0) {
    goto cleanup;
  }
  keys_view = NULL;
  values_view = NULL;
  lengths_view = NULL;
  mask_view = NULL;
  if (
      et_a2_kv_cache_read_borrow_begin_v1(cache, &borrow, &error) != 0 ||
      et_a2_kv_cache_read_borrow_layer_v1(
          borrow, 0u, &keys_view, &values_view, &lengths_view, &mask_view,
          &error) != 0 ||
      ((const int64_t *)lengths_view->data)[0] != 1 ||
      ((const uint8_t *)mask_view->data)[0] != 1u ||
      ((const uint8_t *)mask_view->data)[1] != 0u ||
      ((const float *)keys_view->data)[0] != 7.0f ||
      ((const float *)values_view->data)[2] != 10.0f ||
      et_a2_kv_cache_read_borrow_end_v1(&borrow, &error) != 0) {
    goto cleanup;
  }
  success = 1;

cleanup:
  if (transaction_view != NULL) {
    (void)et_a2_kv_cache_transaction_view_end_v1(&transaction_view, &error);
  }
  if (borrow != NULL) {
    (void)et_a2_kv_cache_read_borrow_end_v1(&borrow, &error);
  }
  if (transaction != NULL) {
    (void)et_a2_kv_cache_transaction_abort_v1(&transaction, &error);
  }
  if (cache != NULL &&
      et_a2_kv_cache_destroy_v1(&cache, &error) != 0) {
    success = 0;
  }
  return success;
}

#ifdef ET_A2_AOT_BRIDGE_STANDALONE
int main(void) {
  return et_a2_test_provider_transport_v1() == 1 &&
                 et_a2_test_cache_transport_v1() == 1
             ? 0
             : 1;
}
#endif
