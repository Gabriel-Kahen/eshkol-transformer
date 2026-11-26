#include <cstdint>
#include <cstring>

extern "C" std::int64_t et_c2_private_model_sha256_v1(
    const void *input_bv, std::int64_t input_bytes, void *digest_bv);

struct bytevector_32 {
  std::int64_t length;
  std::uint8_t bytes[32];
};

int main() {
  bytevector_32 input{0, {}}, output{32, {}};
  const std::uint8_t empty_sha[32] = {
      0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
      0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
      0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
      0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55};
  return et_c2_private_model_sha256_v1(&input, 0, &output) == 0 &&
                 std::memcmp(output.bytes, empty_sha, sizeof(empty_sha)) == 0
             ? 0 : 1;
}
