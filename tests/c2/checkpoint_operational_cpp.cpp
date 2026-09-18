#include <cstddef>
#include <type_traits>

extern "C" std::size_t et_f32_tensor_test_control_bytes_v1(void);
extern "C" std::size_t et_i2_test_decode_builder_control_bytes_v1(void);

using size_witness = std::size_t (*)(void);
static_assert(std::is_same_v<
              decltype(&et_f32_tensor_test_control_bytes_v1), size_witness>);
static_assert(std::is_same_v<
              decltype(&et_i2_test_decode_builder_control_bytes_v1),
              size_witness>);

int main() {
  return et_f32_tensor_test_control_bytes_v1() == 0u ||
                 et_i2_test_decode_builder_control_bytes_v1() == 0u
             ? 1
             : 0;
}
