#include "c2_checkpoint_load_bridge.h"

#include <type_traits>

using measure_type = int64_t (*)(const void *, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, void *);
using stage_type = int64_t (*)(const void *, int64_t, int64_t, int64_t,
                               int64_t, int64_t, const void *, void *, void *,
                               void *, void *, void *, void *, void *, void *,
                               void *);

static_assert(std::is_same_v<decltype(&et_c2_private_checkpoint_load_measure_v1),
                             measure_type>);
static_assert(std::is_same_v<decltype(&et_c2_private_checkpoint_load_stage_v1),
                             stage_type>);

int main() {
  return ET_C2_CHECKPOINT_LOAD_RESULT_BYTES == 192 &&
                 ET_C2_CHECKPOINT_LOAD_RESULT_MAGIC != 0u
             ? 0
             : 1;
}
