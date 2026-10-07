/* Test-owned replacement for exactly one E3 native translation unit. */
#include "../../src/eshkol_transformer/e3_frame.c"

#if defined(ET_E3_RETENTION_TESTING) && \
    defined(ET_E3_FIRST_EVAL_MUTATION_TESTING)
static int first_eval_mutation_used;

/* Fixed controls only; absent-to-present delegates to the existing seed. */
int64_t et_e3_test_first_eval_corrupt_v1(void *pointer, int64_t selector) {
  if (first_eval_mutation_used || selector < 0 || selector > 2) return -1;
  owner *model = admit(pointer, OWNER, 0);
  if (!model || model->r.state != 1 || model->active) return -1;
  et_f32_tensor *gradient = e3_test_model_tensor(model, 0, 1);
  size_t bytes = 0;
  et_f32_tensor_error error;
  if (!gradient || et_f32_tensor_byte_length_v1(gradient, &bytes, &error) != 0 ||
      bytes == 0 || bytes % sizeof(uint32_t) || bytes > 4096) return -1;
  const size_t count = bytes / sizeof(uint32_t);
  uint32_t words[1024] = {0};
  if (et_f32_tensor_copy_bits_to_v1(gradient, words, count, &error) != 0)
    return -1;
  et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
  if (et_f32_parameter_gradient_metadata_v1(model->p[0], &metadata,
                                             &error) != 0) return -1;
  if (selector == 0) {
    if (metadata.state != ET_F32_GRADIENT_ABSENT) return -1;
  } else {
    if (words[0] != UINT32_C(0x3f000000) ||
        metadata.state != ET_F32_GRADIENT_PRESENT ||
        metadata.contribution_count != 1 ||
        metadata.normalization_weight_bits != UINT32_C(0x3f800000)) return -1;
    for (size_t i = 1; i < count; ++i)
      if (words[i] != 0) return -1;
  }

  if (selector == 0) {
    if (et_e3_test_seed_gradient_v1(pointer) != 0) abort();
    words[0] = UINT32_C(0x3f000000);
    for (size_t i = 1; i < count; ++i) words[i] = 0;
  } else if (selector == 1) {
    words[0] = UINT32_C(0x3f000001);
    et_f32_parameter_test_set_gradient_bits_v1(model->p[0], words, count);
  } else {
    et_f32_parameter_test_set_metadata_v1(model->p[0],
                                          ET_F32_GRADIENT_PRESENT, 2,
                                          UINT32_C(0x3f800000));
  }
  for (size_t i = 0; i < count; ++i) {
    const int64_t expected = selector == 2
        ? (i == 0 ? UINT32_C(0x3f000000) : 0) : words[i];
    if (et_e3_test_model_word_v1(pointer, 0, 1, (int64_t)i) != expected)
      abort();
  }
  if (et_e3_test_gradient_metadata_v1(pointer, 0, 0) !=
          ET_F32_GRADIENT_PRESENT ||
      et_e3_test_gradient_metadata_v1(pointer, 0, 1) !=
          (selector == 2 ? 2 : 1) ||
      et_e3_test_gradient_metadata_v1(pointer, 0, 2) !=
          UINT32_C(0x3f800000)) abort();
  first_eval_mutation_used = 1;
  return 0;
}
#endif
