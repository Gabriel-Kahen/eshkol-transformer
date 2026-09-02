#include "eshkol_transformer/a2_attention_abi.h"
#include "eshkol_transformer/a2_kv_cache.h"

int main() {
  return (ET_A2_ATTENTION_ABI_MAJOR == 1u &&
          ET_A2_KV_CACHE_ABI_MAJOR == 1u) ? 0 : 1;
}
