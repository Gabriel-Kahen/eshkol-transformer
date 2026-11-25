/* Bounded raw Q8_0 comparison through GGML's GGUF reader and dequantizer. */
#include "eshkol_transformer/io5_quant_abi.h"
#include "gguf.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define BLOCKS 6u
#define ELEMENTS (BLOCKS * ET_IO5_QK8_0)

static int check(int ok, const char *why) {
  if (!ok) fprintf(stderr, "IO5 Q8 GGML witness failed: %s\n", why);
  return ok;
}

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const struct ggml_type_traits *traits = ggml_get_type_traits(GGML_TYPE_Q8_0);
  if (!check(traits && traits->to_float && traits->is_quantized &&
             traits->blck_size == ET_IO5_QK8_0 &&
             traits->type_size == ET_IO5_Q8_0_BLOCK_BYTES &&
             ggml_blck_size(GGML_TYPE_Q8_0) == ET_IO5_QK8_0 &&
             ggml_type_size(GGML_TYPE_Q8_0) == ET_IO5_Q8_0_BLOCK_BYTES,
             "GGML Q8_0 geometry")) return 1;

  /* Includes signed quants, signed zero, subnormal/normal and max finite scales. */
  static const uint16_t scales[BLOCKS] = {
      0x3c00u, 0x0000u, 0x8000u, 0x0001u, 0x0400u, 0x7bffu};
  uint8_t raw[BLOCKS * ET_IO5_Q8_0_BLOCK_BYTES];
  for (size_t b = 0; b < BLOCKS; ++b) {
    uint8_t *block = raw + b * ET_IO5_Q8_0_BLOCK_BYTES;
    block[0] = (uint8_t)scales[b];
    block[1] = (uint8_t)(scales[b] >> 8);
    for (size_t i = 0; i < ET_IO5_QK8_0; ++i) {
      static const int8_t signed_quants[] = {-128, -127, -31, -1, 0, 1, 31, 127};
      block[2 + i] = (uint8_t)signed_quants[(i + b) % 8];
    }
  }

  uint8_t eshkol_bits[ELEMENTS * 4 + 1];
  if (!check(et_io5_dequantize_q8_0(raw, sizeof raw, eshkol_bits + 1,
                                     ELEMENTS * 4, ELEMENTS) == ET_IO5_QUANT_OK,
             "Eshkol raw decode")) return 1;

  struct ggml_context *write_ctx = ggml_init((struct ggml_init_params){
      .mem_size = 1024 * 1024, .mem_buffer = NULL, .no_alloc = false});
  struct gguf_context *writer = gguf_init_empty();
  if (!check(write_ctx && writer, "GGML writer allocation")) return 1;
  struct ggml_tensor *tensor = ggml_new_tensor_2d(write_ctx, GGML_TYPE_Q8_0, 64, 3);
  if (!check(tensor && tensor->data && ggml_nbytes(tensor) == sizeof raw,
             "GGML writer tensor extent")) return 1;
  ggml_set_name(tensor, "raw.q8_0");
  memcpy(tensor->data, raw, sizeof raw);
  gguf_add_tensor(writer, tensor);
  if (!check(gguf_write_to_file(writer, argv[1], false), "GGUF write")) return 1;
  gguf_free(writer);
  ggml_free(write_ctx);

  struct ggml_context *read_ctx = NULL;
  struct gguf_context *reader = gguf_init_from_file(
      argv[1], (struct gguf_init_params){.no_alloc = false, .ctx = &read_ctx});
  if (!check(reader && read_ctx, "GGUF native read")) return 1;
  const struct ggml_tensor *read_tensor = ggml_get_tensor(read_ctx, "raw.q8_0");
  if (!check(gguf_get_n_tensors(reader) == 1 &&
             gguf_get_tensor_type(reader, 0) == GGML_TYPE_Q8_0 &&
             gguf_get_tensor_size(reader, 0) == sizeof raw &&
             read_tensor && read_tensor->data &&
             read_tensor->type == GGML_TYPE_Q8_0 &&
             read_tensor->ne[0] == 64 && read_tensor->ne[1] == 3 &&
             read_tensor->ne[2] == 1 && read_tensor->ne[3] == 1 &&
             ggml_nbytes(read_tensor) == sizeof raw &&
             memcmp(read_tensor->data, raw, sizeof raw) == 0,
             "GGUF reader raw bytes and checked row geometry")) return 1;

  float ggml_values[ELEMENTS];
  traits->to_float(read_tensor->data, ggml_values, ELEMENTS);
  for (size_t i = 0; i < ELEMENTS; ++i) {
    uint32_t got, expected;
    memcpy(&got, eshkol_bits + 1 + i * 4, sizeof got);
    memcpy(&expected, &ggml_values[i], sizeof expected);
    if (!check(got == expected, "native dequantized F32 bits")) return 1;
  }
  gguf_free(reader);
  ggml_free(read_ctx);
  printf("IO5 Q8 GGML decoder witness: PASS (%u blocks, %u exact F32 bits)\n",
         BLOCKS, ELEMENTS);
  return 0;
}
