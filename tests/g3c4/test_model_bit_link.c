#include "../../src/eshkol_transformer/g3c4_model_owner_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned char handles[14];

static int expect(int condition, const char *message) {
  if (condition) return 1;
  fprintf(stderr, "C4 model-bit link: %s\n", message);
  return 0;
}

int main(void) {
  unsigned char bytes[1192u * 4u];
  unsigned char before[sizeof(bytes)];
  void *owner = et_g3c4_private_model_owner_create_seeded_v1(53);
  if (!expect(owner != NULL, "create")) return 1;
  for (int64_t index = 0; index < 14; index++) {
    if (!expect(et_g3c4_private_model_owner_bind_v1(
                    owner, index, &handles[index]) == 0, "bind")) return 1;
  }
  for (int64_t index = 0; index < 14; index++) {
    if (!expect(et_g3c4_private_model_owner_initialize_v1(owner, index) == 0,
                "initialize")) return 1;
  }
  if (!expect(et_g3c4_private_model_owner_prepare_seal_v1(owner) == 0,
              "prepare")) return 1;
  if (!expect(et_g3c4_private_model_owner_commit_seal_v1(owner) == 0,
              "seal")) return 1;
  for (int64_t index = 0; index < 14; index++) {
    void *parameter = et_g3c4_private_model_owner_parameter_v1(owner, index);
    if (!expect(parameter != NULL, "parameter")) return 1;
    if (!expect(et_g3c4_private_model_owner_binding_matches_v1(
                    owner, index, &handles[index], parameter) == 0,
                "binding")) return 1;
  }
  if (!expect(et_g3c4_private_model_owner_binding_matches_v1(
                  owner, 0, &handles[1],
                  et_g3c4_private_model_owner_parameter_v1(owner, 0)) != 0,
              "wrong handle accepted") ||
      !expect(et_g3c4_private_model_owner_binding_matches_v1(
                  owner, 0, &handles[0],
                  et_g3c4_private_model_owner_parameter_v1(owner, 1)) != 0,
              "wrong parameter accepted")) return 1;
  memset(bytes, 0xa5, sizeof(bytes));
  memcpy(before, bytes, sizeof(bytes));
  if (!expect(et_g3c4_private_model_owner_copy_bits_v1(
                  owner, bytes, (int64_t)sizeof(bytes) - 1) != 0,
              "short extent") ||
      !expect(memcmp(bytes, before, sizeof(bytes)) == 0,
              "short extent changed destination")) return 1;
  if (!expect(et_g3c4_private_model_owner_copy_bits_v1(
                  owner, owner, (int64_t)sizeof(bytes)) != 0,
              "owner alias accepted") ||
      !expect(memcmp(bytes, before, sizeof(bytes)) == 0,
              "alias changed destination")) return 1;
  if (!expect(et_g3c4_private_model_owner_copy_bits_v1(
                  owner, bytes, (int64_t)sizeof(bytes)) == 0,
              "complete copy") ||
      !expect(memcmp(bytes, before, sizeof(bytes)) != 0,
              "copy left sentinel untouched")) return 1;
  puts("C4 model-bit separate-TU link: passed");
  return 0;
}
