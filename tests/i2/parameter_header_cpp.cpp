#include "f32_parameter_internal.h"

#include <cstddef>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_f32_gradient_metadata_v1>);
static_assert(sizeof(et_f32_gradient_metadata_v1) == 24u);
static_assert(offsetof(et_f32_gradient_metadata_v1, state) == 8u);
static_assert(offsetof(et_f32_gradient_metadata_v1,
                       normalization_weight_bits) == 12u);
static_assert(offsetof(et_f32_gradient_metadata_v1, contribution_count) ==
              16u);
static_assert(std::is_standard_layout_v<et_f32_gradient_contribution_v1>);
static_assert(sizeof(et_f32_gradient_contribution_v1) == 32u);

int main() { return 0; }
