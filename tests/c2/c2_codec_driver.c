#include "c2_checkpoint_codec.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}
static uint64_t u64(const uint8_t *p) {
  uint64_t value = 0u; unsigned i;
  for (i = 0u; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}
static int read_all(const char *path, uint8_t **bytes, size_t *length) {
  FILE *file = fopen(path, "rb"); long end;
  if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
      (end = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) return 0;
  *length = (size_t)end; *bytes = malloc(*length ? *length : 1u);
  if (*bytes == NULL || fread(*bytes, 1u, *length, file) != *length ||
      fclose(file) != 0) { free(*bytes); return 0; }
  return 1;
}
static int write_all(const char *path, const uint8_t *bytes, size_t length) {
  FILE *file = fopen(path, "wb");
  return file != NULL && fwrite(bytes, 1u, length, file) == length &&
         fclose(file) == 0;
}

int main(int argc, char **argv) {
  uint8_t *source = NULL, *optimizer = NULL, *output = NULL, *before = NULL;
  uint8_t alias_before[sizeof(size_t)];
  size_t source_bytes, output_bytes = 0u, metadata, x1, current, epoch, o2;
  size_t model, moments; int mode; int32_t status;
  et_c2_checkpoint_encode_request_v1 request = {0};
  et_c2_checkpoint_encode_request_v1 request_before;
  et_c2_checkpoint_format_error_v1 error = {0};
  et_c2_checkpoint_format_error_v1 error_before;
  if (argc != 4 || !read_all(argv[1], &source, &source_bytes)) return 111;
  mode = atoi(argv[3]); metadata = (size_t)u64(source + 56u);
  x1 = 256u + u32(source + 136u) + 93u + 19u + 57u + 71u;
  current = x1 + (size_t)u64(source + 120u);
  epoch = current + u32(source + 128u); o2 = epoch + u32(source + 132u);
  model = 256u + metadata; moments = model + (size_t)u64(source + 96u);
  optimizer = malloc((size_t)u64(source + 104u));
  if (optimizer == NULL) return 111;
  memcpy(optimizer, source + o2, (size_t)u64(source + 104u));
  request.struct_size = sizeof(request);
  request.tokenizer_fingerprint = source + 256u;
  request.tokenizer_fingerprint_bytes = u32(source + 136u);
  request.x1_canonical = source + x1;
  request.x1_canonical_bytes = (size_t)u64(source + 120u);
  request.current_cursor = source + current;
  request.current_cursor_bytes = u32(source + 128u);
  request.epoch_cursor = source + epoch;
  request.epoch_cursor_bytes = u32(source + 132u);
  request.model_c1 = source + model;
  request.model_c1_bytes = (size_t)u64(source + 96u);
  request.optimizer_metadata = optimizer;
  request.optimizer_metadata_bytes = (size_t)u64(source + 104u);
  request.optimizer_payload = source + moments;
  request.optimizer_payload_bytes = (size_t)u64(source + 112u);
  request.total_tokens = u64(source + 184u); request.epochs = u64(source + 200u);
  request.rng_key_bits = u64(source + 216u);
  request.rng_counter_low_bits = u64(source + 224u);
  request.rng_counter_high_bits = u64(source + 232u);
  error.struct_size = sizeof(error);
  status = et_c2_checkpoint_encode_measure_v1(&request, &output_bytes, &error);
  if (status != ET_C2_FORMAT_OK) return 112;
  output = malloc(output_bytes); before = malloc(output_bytes);
  if (output == NULL || before == NULL) return 111;
  memset(output, 0xa5, output_bytes); memcpy(before, output, output_bytes);
  if (mode == 1) {
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes - 1u, &error);
  } else if (mode == 2) {
    memcpy(output + 256u, request.tokenizer_fingerprint,
           request.tokenizer_fingerprint_bytes);
    memcpy(before, output, output_bytes);
    request.tokenizer_fingerprint = output + 256u;
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
  } else if (mode == 3) {
    optimizer[0] ^= 1u;
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
  } else if (mode == 4) {
    source[model + request.model_c1_bytes - 1u] ^= 1u;
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
  } else if (mode == 5) {
    request_before = request;
    status = et_c2_checkpoint_encode_measure_v1(
        &request, (size_t *)(void *)&request, &error);
    if (memcmp(&request, &request_before, sizeof(request)) != 0) return 113;
  } else if (mode == 6) {
    memcpy(alias_before, request.tokenizer_fingerprint, sizeof(alias_before));
    status = et_c2_checkpoint_encode_measure_v1(
        &request, (size_t *)(void *)request.tokenizer_fingerprint, &error);
    if (memcmp(request.tokenizer_fingerprint, alias_before,
               sizeof(alias_before)) != 0) return 113;
  } else if (mode == 7) {
    error_before = error;
    status = et_c2_checkpoint_encode_measure_v1(
        &request, (size_t *)(void *)&error, &error);
    if (memcmp(&error, &error_before, sizeof(error)) != 0) return 113;
  } else if (mode == 8) {
    request_before = request;
    status = et_c2_checkpoint_encode_v1(
        &request, (uint8_t *)(void *)&request, output_bytes, &error);
    if (memcmp(&request, &request_before, sizeof(request)) != 0) return 113;
  } else if (mode == 9) {
    error_before = error;
    status = et_c2_checkpoint_encode_v1(
        &request, (uint8_t *)(void *)&error, output_bytes, &error);
    if (memcmp(&error, &error_before, sizeof(error)) != 0) return 113;
  } else if (mode == 10) {
    size_t measured_before = output_bytes;
    optimizer[0] ^= 1u;
    status = et_c2_checkpoint_encode_measure_v1(
        &request, &output_bytes, &error);
    if (output_bytes != measured_before) return 113;
  } else if (mode == 11) {
    size_t measured_before = output_bytes;
    request.x1_canonical_bytes = SIZE_MAX;
    status = et_c2_checkpoint_encode_measure_v1(
        &request, &output_bytes, &error);
    if (output_bytes != measured_before) return 113;
  } else if (mode == 12) {
    size_t digest_at = (size_t)u64(optimizer + 104u) + 56u;
    optimizer[digest_at] ^= 1u;
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
    optimizer[digest_at] ^= 1u;
    if (status != ET_C2_FORMAT_OK ||
        memcmp(output, source, output_bytes) != 0) return 113;
  } else if (mode == 13) {
    size_t digest_at = current + request.current_cursor_bytes - 1u;
    source[digest_at] ^= 1u;
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
    source[digest_at] ^= 1u;
    if (status != ET_C2_FORMAT_OK ||
        memcmp(output, source, output_bytes) != 0) return 113;
  } else {
    status = et_c2_checkpoint_encode_v1(&request, output, output_bytes, &error);
  }
  if (mode > 0 && mode < 12 &&
      (status == ET_C2_FORMAT_OK ||
       memcmp(output, before, output_bytes) != 0)) return 113;
  if (mode == 0) {
    et_c2_checkpoint_limits_v1 limits = {0};
    et_c2_checkpoint_view_v1 view = {0};
    limits.struct_size = sizeof(limits);
    limits.maximum_file_bytes = ET_C2_WIRE_MAX_FILE_BYTES;
    limits.maximum_metadata_bytes = ET_C2_WIRE_MAX_METADATA_BYTES;
    limits.maximum_tensor_bytes = ET_C2_WIRE_MAX_TENSOR_BYTES;
    limits.maximum_tensors = ET_C2_WIRE_MAX_TENSORS;
    view.struct_size = sizeof(view); error.struct_size = sizeof(error);
    if (status != ET_C2_FORMAT_OK ||
        et_c2_checkpoint_parse_v1(output, output_bytes, &limits,
                                  ET_C2_FORMAT_INSPECT, &view,
                                  &error) != ET_C2_FORMAT_OK ||
        !write_all(argv[2], output, output_bytes)) return 114;
  }
  printf("%d %u %zu\n", status, error.code, output_bytes);
  free(before); free(output); free(optimizer); free(source); return 0;
}
