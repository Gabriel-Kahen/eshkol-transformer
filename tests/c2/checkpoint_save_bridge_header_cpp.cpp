#include "c2_checkpoint_save_bridge.h"

#include <type_traits>

using measure_type = int64_t (*)(
    const void *, const void *, const void *, const void *, const void *,
    const void *, const void *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, void *);
using encode_type = int64_t (*)(
    const void *, const void *, const void *, const void *, const void *,
    const void *, const void *, int64_t, int64_t, int64_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, void *, void *);

static_assert(std::is_same_v<decltype(&et_c2_private_checkpoint_save_measure_v1),
                             measure_type>);
static_assert(
    std::is_same_v<decltype(&et_c2_private_checkpoint_save_encode_validate_v1),
                   encode_type>);
static_assert(ET_C2_CHECKPOINT_SAVE_RESULT_BYTES == 32);

int main() { return 0; }
