#include "eshkol_transformer/t2_bpe_trainer_abi.h"

#include <string.h>

enum {
  T2_MAX_MERGES = 256,
  MERGE_BYTES = 12,
  STAGING_BYTES = T2_MAX_MERGES * MERGE_BYTES,
  T2_IMPORT_UNSUPPORTED = -13
};

static void store_u32_le(uint8_t *bytes, uint32_t value) {
  for (unsigned index = 0; index < 4u; ++index) {
    bytes[index] = (uint8_t)(value >> (8u * index));
  }
}

/* Trusted same-aggregate AOT caller supplies an Eshkol bytevector header.
 * Return the merge count or a negative status, leaving staging intact on
 * failure. The private wire carries (result, left, right) in rank order. */
int64_t et_t2_private_import_native_bpe_v1(const char *path, void *staging) {
  int64_t length = 0;
  if (staging == NULL) return ET_T2_BPE_ERR_ARGUMENT;
  memcpy(&length, staging, sizeof(length));
  if (length != STAGING_BYTES) return ET_T2_BPE_ERR_ARGUMENT;

  et_t2_bpe_model *model = NULL;
  et_t2_bpe_status status = et_t2_bpe_load_file(path, &model, NULL);
  if (status != ET_T2_BPE_OK) return status;

  et_t2_bpe_info info;
  status = et_t2_bpe_get_info(model, &info);
  if (status == ET_T2_BPE_OK &&
      (info.merge_count > T2_MAX_MERGES ||
       info.special_mask != ET_T2_BPE_SPECIAL_NONE)) {
    status = T2_IMPORT_UNSUPPORTED;
  }

  uint8_t bytes[STAGING_BYTES];
  if (status == ET_T2_BPE_OK) {
    for (uint32_t rank = 0; rank < info.merge_count; ++rank) {
      et_t2_bpe_merge merge;
      status = et_t2_bpe_get_merge(model, rank, &merge);
      if (status != ET_T2_BPE_OK) break;
      uint8_t *record = bytes + rank * MERGE_BYTES;
      store_u32_le(record, merge.new_id);
      store_u32_le(record + 4, merge.left);
      store_u32_le(record + 8, merge.right);
    }
  }
  et_t2_bpe_destroy(model);
  if (status != ET_T2_BPE_OK) return status;

  memcpy((uint8_t *)staging + sizeof(length), bytes,
         info.merge_count * MERGE_BYTES);
  return info.merge_count;
}
