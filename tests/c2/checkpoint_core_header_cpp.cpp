#include "c2_checkpoint_core.h"

#include <type_traits>

static_assert(
    std::is_same_v<decltype(&et_c2_private_checkpoint_load_image_v1),
                   int32_t (*)(const char *,
                               const et_c2_checkpoint_limits_v1 *, uint32_t,
                               et_c2_private_checkpoint_image **,
                               et_c2_checkpoint_core_error_v1 *)>);
static_assert(
    std::is_same_v<decltype(&et_c2_private_checkpoint_image_view_v1),
                   int32_t (*)(const et_c2_private_checkpoint_image *,
                               const et_c2_checkpoint_view_v1 **,
                               et_c2_checkpoint_core_error_v1 *)>);
static_assert(
    std::is_same_v<decltype(&et_c2_private_checkpoint_image_release_v1),
                   int32_t (*)(et_c2_private_checkpoint_image **,
                               et_c2_checkpoint_core_error_v1 *)>);

int main() { return 0; }
