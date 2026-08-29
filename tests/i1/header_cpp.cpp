#include "eshkol_transformer/i64_tensor.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

static_assert(ET_I64_TENSOR_ABI_MAJOR == 1u);
static_assert(ET_I64_TENSOR_ABI_MINOR == 0u);
static_assert(sizeof(std::int64_t) == 8u);
static_assert(std::is_standard_layout_v<et_i64_tensor_error>);
static_assert(sizeof(et_i64_tensor_error) == 264u);
static_assert(offsetof(et_i64_tensor_error, category) == 0u);
static_assert(offsetof(et_i64_tensor_error, code) == 4u);
static_assert(offsetof(et_i64_tensor_error, operation) == 8u);
static_assert(offsetof(et_i64_tensor_error, message) == 72u);

int main() {
  et_i64_tensor_error error{};
  if (et_i64_tensor_abi_major_v1() != 1 ||
      et_i64_tensor_abi_minor_v1() != 0) {
    return 1;
  }
  return et_i64_tensor_abi_require_v1(1u, 0u, &error) == 0 ? 0 : 1;
}
