#include "mod5_v256_c2_transfer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%d: %s\n", __LINE__, #condition);                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

enum { SLOT_COUNT = 12, MAX_WORDS = 4096, TOTAL_WORDS = 10544 };
static const size_t words_per_slot[SLOT_COUNT] = {
    128u, 256u, 256u, 128u, 512u, 512u, 512u, 16u, 16u, 4096u, 16u, 4096u};

static uint32_t bits[TOTAL_WORDS];

static int no_gradients(et_f32_parameter *const *parameters, size_t count) {
  for (size_t slot = 0u; slot < count; ++slot) {
    et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    CHECK(et_f32_parameter_gradient_metadata_v1(parameters[slot], &metadata,
                                                  NULL) == 0);
    if (metadata.contribution_count != 0u || metadata.normalization_weight_bits != 0u)
      return 0;
  }
  return 1;
}

static void assert_gradients(et_f32_parameter *const *parameters, size_t count) {
  uint32_t observed[MAX_WORDS];
  size_t offset = 0u;
  for (size_t slot = 0u; slot < count; ++slot) {
    et_f32_gradient_metadata_v1 metadata = {.struct_size = sizeof(metadata)};
    et_f32_tensor *snapshot = NULL;
    const size_t words = words_per_slot[slot];
    CHECK(et_f32_parameter_gradient_metadata_v1(parameters[slot], &metadata,
                                                  NULL) == 0);
    CHECK(metadata.contribution_count == 1u);
    CHECK(metadata.normalization_weight_bits == 0x3f800000u);
    CHECK(et_f32_parameter_gradient_snapshot_v1(parameters[slot], &snapshot,
                                                  NULL) == 0);
    CHECK(et_f32_tensor_copy_bits_to_v1(snapshot, observed, words, NULL) == 0);
    CHECK(memcmp(observed, bits + offset, words * sizeof(*observed)) == 0);
    CHECK(et_f32_tensor_destroy_v1(&snapshot, NULL) == 0);
    offset += words;
  }
}

static et_f32_parameter *transposed_head(void) {
  const uint64_t dimensions[] = {16u, 256u};
  et_f32_tensor *value = NULL;
  et_f32_parameter *parameter = NULL;
  CHECK(et_f32_tensor_create_v1(2u, dimensions, &value, NULL) == 0);
  CHECK(et_f32_parameter_create_v1(value, &parameter, NULL) == 0);
  CHECK(et_f32_tensor_destroy_v1(&value, NULL) == 0);
  return parameter;
}

static void run_variant(int64_t tied) {
  const size_t count = (size_t)(tied ? 11 : 12);
  et_f32_parameter *parameters[SLOT_COUNT] = {NULL};
  uint32_t observed[MAX_WORDS];
  size_t offset = 0u;
  for (size_t slot = 0u; slot < count; ++slot) {
    const size_t words = words_per_slot[slot];
    parameters[slot] =
        et_mod5_v256_test_parameter_v1((int64_t)slot, bits + offset, (int64_t)words);
    CHECK(parameters[slot] != NULL);
    CHECK(et_f32_parameter_bind_identity_v1(parameters[slot], parameters[slot],
                                              NULL) == 0);
    CHECK(et_mod5_v256_test_parameter_bits_v1((int64_t)slot, parameters[slot],
                                                observed, (int64_t)words) == 0);
    CHECK(memcmp(observed, bits + offset, words * sizeof(*observed)) == 0);
    offset += words;
  }
  CHECK(offset == (tied ? 6448u : TOTAL_WORDS));
  CHECK(no_gradients(parameters, count));
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset - 1, tied) != 0);
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, 2) != 0);
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, !tied) != 0);
  et_f32_parameter *transposed = transposed_head();
  et_f32_parameter *head = parameters[9];
  parameters[9] = transposed;
  observed[0] = 0xdeadbeefu;
  CHECK(et_mod5_v256_test_parameter_bits_v1(9, transposed, observed, MAX_WORDS) != 0);
  CHECK(observed[0] == 0xdeadbeefu);
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, tied) != 0);
  parameters[9] = head;
  CHECK(et_f32_parameter_destroy_v1(&transposed, NULL) == 0);
  et_f32_parameter *query = parameters[2];
  parameters[2] = parameters[1];
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, tied) != 0);
  parameters[2] = query;
  CHECK(no_gradients(parameters, count));

  const uint32_t last = bits[offset - 1u];
  bits[offset - 1u] = 0x7fc00000u;
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, tied) != 0);
  CHECK(no_gradients(parameters, count));
  bits[offset - 1u] = last;
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, tied) == 0);
  assert_gradients(parameters, count);
  CHECK(et_mod5_v256_test_gradient_transfer_v1(parameters, (int64_t)count,
                                                  bits, (int64_t)offset, tied) != 0);
  assert_gradients(parameters, count);
  et_f32_gradient_reset_plan *reset = NULL;
  CHECK(et_f32_gradient_reset_plan_prepare_v1(count, parameters, &reset, NULL) == 0);
  CHECK(et_f32_gradient_reset_plan_commit_v1(reset, NULL) == 0);
  CHECK(et_f32_gradient_reset_plan_release_v1(&reset, NULL) == 0);
  CHECK(no_gradients(parameters, count));
  for (size_t slot = 0u; slot < count; ++slot)
    CHECK(et_f32_parameter_destroy_v1(&parameters[slot], NULL) == 0);
}

int main(void) {
  for (size_t index = 0u; index < TOTAL_WORDS; ++index)
    bits[index] = 0x3e800000u + (uint32_t)index;
  CHECK(et_mod5_v256_test_parameter_v1(12, bits, MAX_WORDS) == NULL);
  CHECK(et_mod5_v256_test_parameter_v1(11, bits, MAX_WORDS - 1) == NULL);
  run_variant(0);
  run_variant(1);
  puts("MOD5 V256 I2 gradient transfer PASS");
  return EXIT_SUCCESS;
}
