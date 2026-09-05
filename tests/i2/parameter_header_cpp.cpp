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

static_assert(std::is_same_v<decltype(&et_f32_tensor_storage_owner_identical_v1),
                             int32_t (*)(const et_f32_tensor *,
                                         const et_f32_tensor *)>);
static_assert(std::is_same_v<decltype(&et_f32_tensor_canonical_owner_v1),
                             const et_f32_tensor *(*)(
                                 const et_f32_tensor *)>);
static_assert(std::is_same_v<decltype(&et_f32_parameter_canonical_owner_v1),
                             const et_f32_tensor *(*)(
                                 const et_f32_parameter *)>);
static_assert(std::is_same_v<decltype(&et_f32_owned_tensor_clone_v1),
                             int32_t (*)(const et_f32_tensor *,
                                         et_f32_tensor **,
                                         et_f32_tensor_error *)>);
static_assert(std::is_same_v<decltype(&et_f32_owned_tensor_release_v1),
                             int32_t (*)(et_f32_tensor *,
                                         et_f32_tensor_error *)>);
static_assert(std::is_same_v<decltype(&et_f32_parameter_validate_identity_v1),
                             int32_t (*)(const et_f32_parameter *,
                                         const void *,
                                         et_f32_tensor_error *)>);

int main() { return 0; }
