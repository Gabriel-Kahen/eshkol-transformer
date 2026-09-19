#include "c2_checkpoint_save_bridge.h"

#include "c2_checkpoint_codec.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define C2_MAX_X1 ((size_t)16384u)
#define C2_MAX_CURSOR ((size_t)400u)
#define C2_MAX_FINGERPRINT ((size_t)96u)

typedef struct c2_span {
  const uint8_t *header;
  const uint8_t *bytes;
  size_t length;
  size_t allocation;
} c2_span;

typedef struct c2_save_inputs {
  et_c2_checkpoint_encode_request_v1 request;
  c2_span spans[7];
} c2_save_inputs;

#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
static int64_t c2_save_fail_stage;
void et_c2_checkpoint_save_test_fail_stage_v1(int64_t stage) {
  c2_save_fail_stage = stage;
}
#endif

static void put_u64(uint8_t *p, uint64_t value) {
  unsigned i;
  for (i = 0u; i < 8u; ++i) p[i] = (uint8_t)(value >> (8u * i));
}

static uint32_t get_u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}

static uint64_t get_u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

static int add_size(size_t left, size_t right, size_t *out) {
  if (left > SIZE_MAX - right) return 0;
  *out = left + right;
  return 1;
}

static int span_from_bytevector(const void *opaque, size_t maximum,
                                size_t exact, c2_span *out) {
  int64_t declared;
  uintptr_t start;
  if (opaque == NULL || out == NULL) return 0;
  memcpy(&declared, opaque, sizeof(declared));
  if (declared < 0 || (uint64_t)declared > (uint64_t)maximum ||
      (exact != SIZE_MAX && (uint64_t)declared != (uint64_t)exact))
    return 0;
  start = (uintptr_t)opaque;
  if (start > UINTPTR_MAX - sizeof(declared) ||
      (size_t)declared > UINTPTR_MAX - start - sizeof(declared))
    return 0;
  out->header = (const uint8_t *)opaque;
  out->bytes = (const uint8_t *)(start + sizeof(declared));
  out->length = (size_t)declared;
  out->allocation = sizeof(declared) + (size_t)declared;
  return 1;
}

static int overlaps(const c2_span *left, const c2_span *right) {
  uintptr_t a = (uintptr_t)left->header;
  uintptr_t b = (uintptr_t)right->header;
  if (left->allocation > UINTPTR_MAX - a ||
      right->allocation > UINTPTR_MAX - b)
    return 1;
  return a < b + right->allocation && b < a + left->allocation;
}

static int disjoint_from_inputs(const c2_span *candidate,
                                const c2_save_inputs *inputs) {
  size_t i;
  for (i = 0u; i < 7u; ++i)
    if (overlaps(candidate, &inputs->spans[i])) return 0;
  return 1;
}

static void publish_failure(uint8_t *result, uint64_t phase,
                            const et_c2_checkpoint_format_error_v1 *error) {
  memset(result, 0, (size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES);
  put_u64(result, phase);
  if (error != NULL) {
    put_u64(result + 8u, error->category);
    put_u64(result + 16u, error->code);
    put_u64(result + 24u, error->offset);
  }
}

static int collect_inputs(c2_save_inputs *out, const void *tokenizer,
                          const void *x1, const void *current,
                          const void *epoch, const void *model,
                          const void *optimizer_metadata,
                          const void *optimizer_payload, int64_t total_tokens,
                          int64_t epochs, int64_t rng_key_bits,
                          int64_t rng_counter_low_bits,
                          int64_t rng_counter_high_bits) {
  memset(out, 0, sizeof(*out));
  if (total_tokens < 0 || epochs < 0 ||
      !span_from_bytevector(tokenizer, C2_MAX_FINGERPRINT, SIZE_MAX,
                            &out->spans[0]) ||
      !span_from_bytevector(x1, C2_MAX_X1, SIZE_MAX, &out->spans[1]) ||
      !span_from_bytevector(current, C2_MAX_CURSOR, SIZE_MAX, &out->spans[2]) ||
      !span_from_bytevector(epoch, C2_MAX_CURSOR, SIZE_MAX, &out->spans[3]) ||
      !span_from_bytevector(model, (size_t)ET_C2_WIRE_MAX_FILE_BYTES,
                            SIZE_MAX, &out->spans[4]) ||
      !span_from_bytevector(optimizer_metadata,
                            (size_t)ET_C2_WIRE_MAX_METADATA_BYTES, SIZE_MAX,
                            &out->spans[5]) ||
      !span_from_bytevector(optimizer_payload,
                            (size_t)ET_C2_WIRE_MAX_FILE_BYTES, SIZE_MAX,
                            &out->spans[6]))
    return 0;
  out->request.struct_size = sizeof(out->request);
  out->request.tokenizer_fingerprint = out->spans[0].bytes;
  out->request.tokenizer_fingerprint_bytes = out->spans[0].length;
  out->request.x1_canonical = out->spans[1].bytes;
  out->request.x1_canonical_bytes = out->spans[1].length;
  out->request.current_cursor = out->spans[2].bytes;
  out->request.current_cursor_bytes = out->spans[2].length;
  out->request.epoch_cursor = out->spans[3].bytes;
  out->request.epoch_cursor_bytes = out->spans[3].length;
  out->request.model_c1 = out->spans[4].bytes;
  out->request.model_c1_bytes = out->spans[4].length;
  out->request.optimizer_metadata = out->spans[5].bytes;
  out->request.optimizer_metadata_bytes = out->spans[5].length;
  out->request.optimizer_payload = out->spans[6].bytes;
  out->request.optimizer_payload_bytes = out->spans[6].length;
  out->request.total_tokens = (uint64_t)total_tokens;
  out->request.epochs = (uint64_t)epochs;
  out->request.rng_key_bits = (uint64_t)rng_key_bits;
  out->request.rng_counter_low_bits = (uint64_t)rng_counter_low_bits;
  out->request.rng_counter_high_bits = (uint64_t)rng_counter_high_bits;
  return 1;
}

typedef struct c2_save_metrics {
  size_t file_bytes;
  size_t metadata_bytes;
  uint64_t maximum_tensor_bytes;
  uint32_t tensor_count;
} c2_save_metrics;

/* The codec measure has already validated all spans before this projection. */
static int collect_metrics(const c2_save_inputs *in, size_t file_bytes,
                           c2_save_metrics *metrics) {
  const uint8_t *model = in->spans[4].bytes;
  const uint8_t *optimizer = in->spans[5].bytes;
  size_t outer_metadata = (size_t)240u;
  size_t at, end, i, c1_metadata;
  uint32_t model_count = get_u32(model + 80u);
  uint32_t unique_count = get_u32(optimizer + 52u);
  uint64_t maximum = 0u;
  if (!add_size(outer_metadata, in->spans[0].length, &outer_metadata) ||
      !add_size(outer_metadata, in->spans[1].length, &outer_metadata) ||
      !add_size(outer_metadata, in->spans[2].length, &outer_metadata) ||
      !add_size(outer_metadata, in->spans[3].length, &outer_metadata) ||
      !add_size(outer_metadata, in->spans[5].length, &outer_metadata) ||
      get_u64(model + 56u) > SIZE_MAX)
    return 0;
  c1_metadata = (size_t)get_u64(model + 56u);
  if (!add_size(outer_metadata, c1_metadata, &metrics->metadata_bytes) ||
      unique_count > (UINT32_MAX - model_count) / 2u)
    return 0;
  metrics->file_bytes = file_bytes;
  metrics->tensor_count = model_count + 2u * unique_count;
  at = 128u + (size_t)get_u32(model + 88u);
  end = (size_t)get_u64(model + 64u);
  for (i = 0u; i < (size_t)model_count; ++i) {
    if (at > end || end - at < 80u) return 0;
    uint64_t record_bytes = get_u64(model + at);
    uint64_t tensor_bytes = get_u64(model + at + 16u);
    if (tensor_bytes > maximum) maximum = tensor_bytes;
    if (record_bytes > SIZE_MAX || (size_t)record_bytes > end - at) return 0;
    at += (size_t)record_bytes;
  }
  at = (size_t)get_u64(optimizer + 104u);
  end = in->spans[5].length;
  for (i = 0u; i < (size_t)unique_count; ++i) {
    if (at > end || end - at < 120u) return 0;
    uint64_t record_bytes = get_u64(optimizer + at);
    uint64_t tensor_bytes = get_u64(optimizer + at + 32u);
    if (tensor_bytes > maximum) maximum = tensor_bytes;
    if (record_bytes > SIZE_MAX || (size_t)record_bytes > end - at) return 0;
    at += (size_t)record_bytes;
  }
  metrics->maximum_tensor_bytes = maximum;
  return 1;
}

static int within_limits(const c2_save_metrics *metrics, uint64_t file,
                         uint64_t metadata, uint64_t tensor,
                         uint32_t tensors) {
  return metrics->file_bytes <= file && metrics->metadata_bytes <= metadata &&
         metrics->maximum_tensor_bytes <= tensor &&
         metrics->tensor_count <= tensors;
}

static int valid_limits(int64_t file, int64_t metadata, int64_t tensor,
                        int64_t tensors, int64_t profile) {
  return file > 0 && (uint64_t)file <= ET_C2_WIRE_MAX_FILE_BYTES &&
         metadata > 0 &&
         (uint64_t)metadata <= ET_C2_WIRE_MAX_METADATA_BYTES && tensor > 0 &&
         (uint64_t)tensor <= ET_C2_WIRE_MAX_TENSOR_BYTES && tensors > 0 &&
         tensors <= (int64_t)ET_C2_WIRE_MAX_TENSORS &&
         (profile == 0 || profile == 1);
}

int64_t et_c2_private_checkpoint_save_measure_v1(
    const void *tokenizer_fingerprint, const void *x1_canonical,
    const void *current_cursor, const void *epoch_cursor, const void *model_c1,
    const void *optimizer_metadata, const void *optimizer_payload,
    int64_t total_tokens, int64_t epochs, int64_t rng_key_bits,
    int64_t rng_counter_low_bits, int64_t rng_counter_high_bits,
    int64_t maximum_file_bytes, int64_t maximum_metadata_bytes,
    int64_t maximum_tensor_bytes, int64_t maximum_tensors,
    int64_t enforce_operational_profile, void *result) {
  c2_save_inputs in;
  c2_span result_span;
  c2_save_metrics metrics;
  et_c2_checkpoint_format_error_v1 error;
  size_t measured = 0u;
  uint8_t *out;
  int32_t status;
  if (!span_from_bytevector(result,
                            (size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES,
                            (size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES,
                            &result_span))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (!collect_inputs(&in, tokenizer_fingerprint, x1_canonical, current_cursor,
                      epoch_cursor, model_c1, optimizer_metadata,
                      optimizer_payload, total_tokens, epochs, rng_key_bits,
                      rng_counter_low_bits, rng_counter_high_bits) ||
      !valid_limits(maximum_file_bytes, maximum_metadata_bytes,
                    maximum_tensor_bytes, maximum_tensors,
                    enforce_operational_profile) ||
      !disjoint_from_inputs(&result_span, &in))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  out = (uint8_t *)(uintptr_t)result_span.bytes;
  memset(out, 0, result_span.length);
  memset(&error, 0, sizeof(error));
  error.struct_size = sizeof(error);
#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
  if (c2_save_fail_stage == ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE) {
    error.category = ET_C2_FORMAT_CORRUPT_DATA;
    error.code = ET_C2_FORMAT_CODE_FIXED_FIELD;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE, &error);
    return ET_C2_FORMAT_CORRUPT_DATA;
  }
#endif
  status = et_c2_checkpoint_encode_measure_v1(&in.request, &measured, &error);
  if (status != ET_C2_FORMAT_OK) {
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE, &error);
    return status;
  }
  if (!collect_metrics(&in, measured, &metrics)) {
    error.category = ET_C2_FORMAT_CORRUPT_DATA;
    error.code = ET_C2_FORMAT_CODE_FIXED_FIELD;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE, &error);
    return ET_C2_FORMAT_CORRUPT_DATA;
  }
  if (!within_limits(&metrics, (uint64_t)maximum_file_bytes,
                     (uint64_t)maximum_metadata_bytes,
                     (uint64_t)maximum_tensor_bytes,
                     (uint32_t)maximum_tensors)) {
    error.category = ET_C2_FORMAT_CORRUPT_DATA;
    error.code = ET_C2_FORMAT_CODE_LIMIT;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE, &error);
    return ET_C2_FORMAT_CORRUPT_DATA;
  }
  if (enforce_operational_profile != 0 &&
      !within_limits(&metrics, ET_C2_PROFILE_MAX_FILE_BYTES,
                     ET_C2_PROFILE_MAX_METADATA_BYTES,
                     ET_C2_PROFILE_MAX_TENSOR_BYTES,
                     ET_C2_PROFILE_MAX_TENSORS)) {
    error.category = ET_C2_FORMAT_UNSUPPORTED;
    error.code = ET_C2_FORMAT_CODE_PROFILE;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_MEASURE, &error);
    return ET_C2_FORMAT_UNSUPPORTED;
  }
  put_u64(out, ET_C2_CHECKPOINT_SAVE_RESULT_MAGIC);
  put_u64(out + 8u, measured);
  return ET_C2_FORMAT_OK;
}

int64_t et_c2_private_checkpoint_save_encode_validate_v1(
    const void *tokenizer_fingerprint, const void *x1_canonical,
    const void *current_cursor, const void *epoch_cursor, const void *model_c1,
    const void *optimizer_metadata, const void *optimizer_payload,
    int64_t total_tokens, int64_t epochs, int64_t rng_key_bits,
    int64_t rng_counter_low_bits, int64_t rng_counter_high_bits,
    int64_t maximum_file_bytes, int64_t maximum_metadata_bytes,
    int64_t maximum_tensor_bytes, int64_t maximum_tensors,
    int64_t enforce_operational_profile, void *destination, void *result) {
  c2_save_inputs in;
  c2_span destination_span, result_span;
  et_c2_checkpoint_format_error_v1 error;
  et_c2_checkpoint_limits_v1 limits;
  et_c2_checkpoint_view_v1 view;
  uint8_t *dest, *out;
  size_t measured = 0u;
  int32_t status;
  if (!span_from_bytevector(result,
                            (size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES,
                            (size_t)ET_C2_CHECKPOINT_SAVE_RESULT_BYTES,
                            &result_span) ||
      !collect_inputs(&in, tokenizer_fingerprint, x1_canonical, current_cursor,
                      epoch_cursor, model_c1, optimizer_metadata,
                      optimizer_payload, total_tokens, epochs, rng_key_bits,
                      rng_counter_low_bits, rng_counter_high_bits) ||
      !valid_limits(maximum_file_bytes, maximum_metadata_bytes,
                    maximum_tensor_bytes, maximum_tensors,
                    enforce_operational_profile) ||
      !disjoint_from_inputs(&result_span, &in))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  out = (uint8_t *)(uintptr_t)result_span.bytes;
  memset(out, 0, result_span.length);
  memset(&error, 0, sizeof(error));
  error.struct_size = sizeof(error);
  status = et_c2_checkpoint_encode_measure_v1(&in.request, &measured, &error);
  if (status != ET_C2_FORMAT_OK || measured > (size_t)maximum_file_bytes ||
      !span_from_bytevector(destination, (size_t)maximum_file_bytes, measured,
                            &destination_span) ||
      !disjoint_from_inputs(&destination_span, &in) ||
      overlaps(&destination_span, &result_span)) {
    if (status == ET_C2_FORMAT_OK) {
      error.category = ET_C2_FORMAT_INVALID_ARGUMENT;
      error.code = ET_C2_FORMAT_CODE_LIMIT;
      error.offset = 0u;
    }
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE, &error);
    return status == ET_C2_FORMAT_OK ? ET_C2_FORMAT_INVALID_ARGUMENT : status;
  }
  dest = (uint8_t *)(uintptr_t)destination_span.bytes;
#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
  if (c2_save_fail_stage == ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE) {
    error.category = ET_C2_FORMAT_CORRUPT_DATA;
    /* Exercise the phase-aware mapper: an encode-time LIMIT denotes a
       trusted destination/result contract defect, not caller state excess. */
    error.code = ET_C2_FORMAT_CODE_LIMIT;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE, &error);
    return ET_C2_FORMAT_CORRUPT_DATA;
  }
#endif
  status = et_c2_checkpoint_encode_v1(&in.request, dest,
                                      destination_span.length, &error);
  if (status != ET_C2_FORMAT_OK) {
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_ENCODE, &error);
    return status;
  }
#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
  if (c2_save_fail_stage == ET_C2_CHECKPOINT_SAVE_PHASE_PARSE &&
      destination_span.length > 240u)
    dest[240u] = 1u;
#endif
  memset(&limits, 0, sizeof(limits));
  limits.struct_size = sizeof(limits);
  limits.maximum_file_bytes = (uint64_t)maximum_file_bytes;
  limits.maximum_metadata_bytes = (uint64_t)maximum_metadata_bytes;
  limits.maximum_tensor_bytes = (uint64_t)maximum_tensor_bytes;
  limits.maximum_tensors = (uint32_t)maximum_tensors;
  limits.enforce_operational_profile = (uint32_t)enforce_operational_profile;
  memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  status = et_c2_checkpoint_parse_v1(dest, destination_span.length, &limits,
                                     ET_C2_FORMAT_LOAD, &view, &error);
  if (status != ET_C2_FORMAT_OK) {
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_PARSE, &error);
    return status;
  }
#ifdef ET_C2_CHECKPOINT_SAVE_TESTING
  if (c2_save_fail_stage == ET_C2_CHECKPOINT_SAVE_PHASE_POSTCONDITION)
    view.file_bytes = 0u;
#endif
  if (view.bytes != dest || view.file_bytes != destination_span.length ||
      view.model_container_bytes != in.spans[4].length ||
      view.optimizer_metadata_bytes != in.spans[5].length ||
      view.optimizer_payload_bytes != in.spans[6].length ||
      view.x1_bytes != in.spans[1].length ||
      view.total_tokens != (uint64_t)total_tokens ||
      view.epochs != (uint64_t)epochs ||
      view.rng_key_bits != (uint64_t)rng_key_bits ||
      view.rng_counter_low_bits != (uint64_t)rng_counter_low_bits ||
      view.rng_counter_high_bits != (uint64_t)rng_counter_high_bits) {
    error.category = ET_C2_FORMAT_CORRUPT_DATA;
    error.code = ET_C2_FORMAT_CODE_FIXED_FIELD;
    error.offset = 0u;
    publish_failure(out, ET_C2_CHECKPOINT_SAVE_PHASE_POSTCONDITION, &error);
    return ET_C2_FORMAT_CORRUPT_DATA;
  }
  put_u64(out, ET_C2_CHECKPOINT_SAVE_RESULT_MAGIC);
  put_u64(out + 8u, destination_span.length);
  return ET_C2_FORMAT_OK;
}
