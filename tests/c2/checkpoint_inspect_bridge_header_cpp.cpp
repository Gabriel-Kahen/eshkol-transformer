#include "c2_checkpoint_inspect_bridge.h"

#include <type_traits>

static_assert(std::is_same_v<
    decltype(&et_c2_private_checkpoint_inspect_bridge_v1),
    int64_t (*)(void *, int64_t, int64_t, int64_t, int64_t, int64_t, void *)>);

int main() { return ET_C2_INSPECT_RESULT_BYTES == 320 ? 0 : 1; }
