#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tr3_c_private_factory_bridge.h"

static int manifest_digest(const char *directory, uint8_t out[32]) {
  char path[4096];
  if (snprintf(path, sizeof(path), "%s/manifest.etm", directory) >=
      (int)sizeof(path)) return 0;
  FILE *file = fopen(path, "rb");
  if (!file) return 0;
  int ok = fseek(file, -32, SEEK_END) == 0 && fread(out, 1, 32, file) == 32;
  if (fclose(file) != 0) ok = 0;
  return ok;
}

int main(int argc, char **argv) {
  if (argc != 4) return 2;
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    return 3;
  }
  int (*initialize)(void) = (int (*)(void))dlsym(
      library, "et_tr3_c_private_initialize_v1");
  et_tr3_c_result_v1 (*create)(const et_tr3_c_create_request_v1 *) =
      (et_tr3_c_result_v1 (*)(const et_tr3_c_create_request_v1 *))dlsym(
          library, "et_tr3_c_private_trainer_create_v1");
  et_tr3_c_result_v1 (*close)(et_tr3_c_handle_v1 *) =
      (et_tr3_c_result_v1 (*)(et_tr3_c_handle_v1 *))dlsym(
          library, "et_tr3_c_private_trainer_close_v1");
  if (!initialize || !create || !close ||
      dlsym(library, "tr3-c-factory-create-source") ||
      dlsym(library, "et_tr3_c_factory_input_v1") ||
      dlsym(library, "tr3-lease-create-internal") ||
      dlsym(library, "tr3-lease-unenroll-internal!")) return 4;
  if (initialize() != ET_TR3_C_INIT_READY_V1) return 5;

  static const uint8_t json[] =
      "{\"config-schema-major\":1,\"config-schema-minor\":0,"
      "\"model.context-length\":2,\"model.hidden-size\":4,"
      "\"model.layer-count\":1,\"model.query-head-count\":2,"
      "\"model.vocabulary-size\":256,\"run.seed\":1729}";
  static const uint8_t alternate_seed_json[] =
      "{\"config-schema-major\":1,\"config-schema-minor\":0,"
      "\"model.context-length\":2,\"model.hidden-size\":4,"
      "\"model.layer-count\":1,\"model.query-head-count\":2,"
      "\"model.vocabulary-size\":256,\"run.seed\":2718}";
  static const uint8_t wrong_profile_json[] =
      "{\"config-schema-major\":1,\"config-schema-minor\":0,"
      "\"model.context-length\":2,\"model.hidden-size\":8,"
      "\"model.layer-count\":1,\"model.query-head-count\":2,"
      "\"model.kv-head-count\":2,\"model.head-size\":4,"
      "\"model.vocabulary-size\":256,\"run.seed\":1729}";
  static const uint8_t bad_json[] = "{bad}";
  static const char absent[] = "/out/corpus-missing";
  const int alternate_seed = strcmp(argv[3], "seed-2718") == 0;
  const int wrong_profile = strcmp(argv[3], "profile") == 0;
  const int d2_limit = strcmp(argv[3], "d2-limit") == 0;
  const int success_mode = strcmp(argv[3], "success") == 0 || alternate_seed;
  et_tr3_c_create_request_v1 request = {0};
  request.size = sizeof(request);
  request.major = 1;
  request.x1_json = json;
  request.x1_len = (uint32_t)strlen((const char *)json);
  request.directory = (const uint8_t *)argv[2];
  request.directory_len = (uint32_t)strlen(argv[2]);
  request.batch_size = 1;
  request.sequence_length = 2;
  request.maximum_manifest_bytes = 1048576;
  request.maximum_shard_bytes = 1048576;
  request.maximum_total_tokens = 4;
  request.maximum_batch_bytes = 34;
  request.shuffle_window_rows = 1;
  request.packing = 1;
  request.adamw_f32_bits[0] = UINT32_C(0x3dcccccd);
  request.adamw_f32_bits[1] = UINT32_C(0x3f000000);
  request.adamw_f32_bits[2] = UINT32_C(0x3f000000);
  request.adamw_f32_bits[3] = UINT32_C(0x3a83126f);
  if (!manifest_digest(argv[2], request.expected_manifest_sha256)) return 6;

  et_tr3_c_result_v1 rejected = create(NULL);
  if (rejected.status != ET_TR3_C_INVALID_ARGUMENT ||
      rejected.stage != ET_TR3_C_STAGE_ADMISSION ||
      rejected.reason != ET_TR3_C_REASON_MALFORMED_REQUEST ||
      rejected.handle) return 7;
  if (strcmp(argv[3], "digest") == 0) request.expected_manifest_sha256[0] ^= 1;
  if (strcmp(argv[3], "x1") == 0) {
    request.x1_json = bad_json;
    request.x1_len = sizeof(bad_json) - 1;
  }
  if (alternate_seed) {
    request.x1_json = alternate_seed_json;
    request.x1_len = sizeof(alternate_seed_json) - 1;
  }
  if (wrong_profile) {
    request.x1_json = wrong_profile_json;
    request.x1_len = sizeof(wrong_profile_json) - 1;
  }
  if (strcmp(argv[3], "missing") == 0 || wrong_profile || d2_limit) {
    request.directory = (const uint8_t *)absent;
    request.directory_len = sizeof(absent) - 1;
  }
  if (d2_limit) request.maximum_batch_bytes = 1;

  et_tr3_c_result_v1 first = create(&request);
  if (success_mode) {
    if (first.status != ET_TR3_C_OK || first.stage != ET_TR3_C_STAGE_NONE ||
        first.handle == NULL) return 8;
  } else {
    uint32_t expected_stage = strcmp(argv[3], "x1") == 0 || wrong_profile
                                  ? ET_TR3_C_STAGE_X1
                                  : strcmp(argv[3], "digest") == 0
                                        ? ET_TR3_C_STAGE_CORPUS_IDENTITY
                                        : ET_TR3_C_STAGE_D2;
    if (first.status == ET_TR3_C_OK || first.stage != expected_stage ||
        first.handle != NULL) return 9;
    if (strcmp(argv[3], "digest") == 0 &&
        (first.status != ET_TR3_C_INVALID_ARGUMENT ||
         first.reason != ET_TR3_C_REASON_DIGEST_MISMATCH)) return 10;
    if (strcmp(argv[3], "x1") == 0 &&
        (first.status != ET_TR3_C_INVALID_ARGUMENT ||
         first.reason != ET_TR3_C_REASON_RAISED_E1)) return 11;
    if (wrong_profile &&
        (first.status != ET_TR3_C_UNSUPPORTED ||
         first.reason != ET_TR3_C_REASON_RAISED_E1)) return 16;
    if (d2_limit &&
        (first.status != ET_TR3_C_INVALID_ARGUMENT ||
         first.reason != ET_TR3_C_REASON_RAISED_E1)) return 17;
  }
  if (first.original_category != 0) return 18;
  et_tr3_c_result_v1 again = create(&request);
  if (again.status != ET_TR3_C_INVALID_STATE ||
      again.reason != ET_TR3_C_REASON_ATTEMPT_USED || again.handle) return 12;
  if (success_mode) {
    et_tr3_c_result_v1 forged = close(NULL);
    if (forged.status != ET_TR3_C_INVALID_ARGUMENT ||
        forged.reason != ET_TR3_C_REASON_BAD_HANDLE) return 13;
    et_tr3_c_result_v1 ended = close(first.handle);
    if (ended.status != ET_TR3_C_OK || ended.handle) return 14;
    et_tr3_c_result_v1 repeated = close(first.handle);
    if (repeated.status != ET_TR3_C_INVALID_STATE ||
        repeated.reason != ET_TR3_C_REASON_ALREADY_CLOSED) return 15;
  }
  if (alternate_seed || wrong_profile || d2_limit)
    printf("TR3 private factory %s status=%u stage=%u reason=%u original=%u PASS\n",
           argv[3], first.status, first.stage, first.reason,
           first.original_category);
  else
    printf("TR3 private factory %s PASS\n", argv[3]);
  return 0;
}
