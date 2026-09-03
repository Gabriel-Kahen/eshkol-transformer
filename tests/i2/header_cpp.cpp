#include "eshkol_transformer/f32_tensor.h"

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(ET_F32_TENSOR_ABI_MAJOR == 1u);
static_assert(ET_F32_TENSOR_ABI_MINOR == 0u);
static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128);
static_assert(std::is_standard_layout_v<et_f32_tensor_error>);
static_assert(sizeof(et_f32_tensor_error) == 264u);
static_assert(offsetof(et_f32_tensor_error, category) == 0u);
static_assert(offsetof(et_f32_tensor_error, code) == 4u);
static_assert(offsetof(et_f32_tensor_error, operation) == 8u);
static_assert(offsetof(et_f32_tensor_error, message) == 72u);
static_assert(std::is_standard_layout_v<et_f32_tensor_copy_assignment_v1>);
static_assert(ET_F32_TENSOR_COPY_ASSIGNMENT_V1_0_SIZE == 24u);
static_assert(sizeof(et_f32_tensor_copy_assignment_v1) == 24u);
static_assert(offsetof(et_f32_tensor_copy_assignment_v1, struct_size) == 0u);
static_assert(offsetof(et_f32_tensor_copy_assignment_v1, destination) == 8u);
static_assert(offsetof(et_f32_tensor_copy_assignment_v1, source) == 16u);

int main() {
  et_f32_tensor_error error{};
  if (et_f32_tensor_abi_major_v1() != 1 ||
      et_f32_tensor_abi_minor_v1() != 0) {
    return 1;
  }
  return et_f32_tensor_abi_require_v1(1u, 0u, &error) == 0 ? 0 : 1;
}
