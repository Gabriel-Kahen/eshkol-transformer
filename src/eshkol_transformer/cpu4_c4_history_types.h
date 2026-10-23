#ifndef ET_CPU4_C4_HISTORY_TYPES_H
#define ET_CPU4_C4_HISTORY_TYPES_H
#ifdef ET_CPU4_C4_HISTORY_PRIVATE
#if !defined(ET_G3C4_GENERATOR_PRIVATE) || \
    !defined(ET_G3C4_ACTIVE_CALL_PRIVATE) || \
    !defined(ET_G3C4_TOKEN_FRAME_PRIVATE) || \
    !defined(ET_G3C4_TOKEN_FORWARD_PRIVATE) || \
    !defined(ET_G3C4_NATIVE_PINS_PRIVATE) || \
    !defined(ET_G3C4_PREFILL3_PRIVATE) || \
    !defined(ET_G3C4_OUTPUT_DECODE_IDS_PRIVATE) || \
    !defined(ET_A2_KV_CACHE_STORAGE_QUERY_PRIVATE) || \
    !defined(ET_I64_TENSOR_STORAGE_QUERY_PRIVATE) || \
    !defined(ET_F32_TENSOR_STORAGE_QUERY_PRIVATE)
#error "CPU4 C4 history requires the complete private owner/pin/prefill/storage tuple"
#endif
#include <stdint.h>
#include <stddef.h>
typedef struct cpu4_history_record {
  int64_t ids[4];
  uint32_t origin[4];
  uint32_t generated_eos_mask, count, prompt_count, generated_count;
  uint32_t manual_count, state;
  uint64_t commit_epoch;
} cpu4_history_record;
_Static_assert(sizeof(cpu4_history_record) == 80u, "CPU4 history extent");
_Static_assert(offsetof(cpu4_history_record, commit_epoch) == 72u,
               "CPU4 history epoch offset");
#endif
#endif
