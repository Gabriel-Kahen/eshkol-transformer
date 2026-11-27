#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <stdio.h>
#include <stdlib.h>

static void check(int success, const char *message) {
  if (!success) {
    fprintf(stderr, "FAIL T2 native corpus fixture: %s\n", message);
    exit(EXIT_FAILURE);
  }
}

static void fixture_path(char path[4096], const char *directory,
                         const char *name) {
  const int length = snprintf(path, 4096, "%s/%s", directory, name);
  check(length > 0 && length < 4096, "path");
}

static et_t2_bpe_model *train(const et_t2_bpe_document *docs, size_t count,
                              uint32_t vocab, uint32_t frequency,
                              uint32_t specials, uint32_t merges) {
  const et_t2_bpe_train_config config = {vocab, frequency, specials};
  et_t2_bpe_model *model = NULL;
  et_t2_bpe_info info;
  check(et_t2_bpe_train(docs, count, &config, &model) == ET_T2_BPE_OK,
        "train");
  check(et_t2_bpe_get_info(model, &info) == ET_T2_BPE_OK &&
            info.merge_count == merges,
        "learned merge count");
  return model;
}

static void save_model(const char *directory, const char *name,
                       const et_t2_bpe_model *model) {
  char path[4096];
  fixture_path(path, directory, name);
  check(et_t2_bpe_save_file(model, path, NULL) == ET_T2_BPE_OK, name);
}

int main(int argc, char **argv) {
  check(argc == 2, "expected DIRECTORY");
  const uint8_t raw[] = {0, 255, 0, 255, 237, 160, 128};
  const et_t2_bpe_document docs[] = {
      {(const uint8_t *)"abababab", 8}, {(const uint8_t *)"babababa", 8},
      {raw, sizeof(raw)}, {NULL, 0}};
  et_t2_bpe_model *model = train(docs, 4, 259, 2, ET_T2_BPE_SPECIAL_NONE, 3);
  save_model(argv[1], "trained.etbpe", model);

  char path[4096];
  fixture_path(path, argv[1], "native.i64");
  FILE *out = fopen(path, "wb");
  check(out != NULL, "open native staging");
  size_t total = 0;
  for (size_t index = 0; index < 4; ++index) {
    uint32_t ids[8];
    size_t count = 0;
    check(et_t2_bpe_encode(model, docs[index].bytes, docs[index].byte_count,
                           ids, 8, &count) == ET_T2_BPE_OK,
          "native encode");
    for (size_t token = 0; token < count; ++token) {
      uint64_t value = ids[token];
      for (unsigned byte = 0; byte < 8; ++byte) {
        check(fputc((int)(value & 255u), out) != EOF, "write staging");
        value >>= 8;
      }
    }
    total += count;
  }
  check(total < 23 && total > 3, "learned merges and multiple shards");
  check(fclose(out) == 0, "close native staging");
  et_t2_bpe_destroy(model);

  model = train(NULL, 0, 256, 1, ET_T2_BPE_SPECIAL_NONE, 0);
  save_model(argv[1], "empty.etbpe", model);
  et_t2_bpe_destroy(model);
  model = train(NULL, 0, 257, 1, ET_T2_BPE_SPECIAL_BOS, 0);
  save_model(argv[1], "special.etbpe", model);
  et_t2_bpe_destroy(model);

  uint8_t pairs[257][2];
  et_t2_bpe_document distinct[257];
  for (size_t index = 0; index < 257; ++index) {
    pairs[index][0] = (uint8_t)(index >> 8);
    pairs[index][1] = (uint8_t)index;
    distinct[index] = (et_t2_bpe_document){pairs[index], 2};
  }
  model = train(distinct, 257, 512, 1, ET_T2_BPE_SPECIAL_NONE, 256);
  save_model(argv[1], "limit.etbpe", model);
  et_t2_bpe_destroy(model);
  model = train(distinct, 257, 513, 1, ET_T2_BPE_SPECIAL_NONE, 257);
  save_model(argv[1], "over-limit.etbpe", model);
  et_t2_bpe_destroy(model);
  return EXIT_SUCCESS;
}
