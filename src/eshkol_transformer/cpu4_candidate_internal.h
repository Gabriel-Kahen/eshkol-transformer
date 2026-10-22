#ifndef ET_CPU4_CANDIDATE_INTERNAL_H
#define ET_CPU4_CANDIDATE_INTERNAL_H
#include "eshkol_transformer/a2_kv_cache.h"
#ifdef ET_CPU4_CANDIDATE_PRIVATE
int32_t et_cpu4_kv_cache_destroy_preflight_internal(
    et_a2_kv_cache *, et_kernel_error *);
#endif
#endif
