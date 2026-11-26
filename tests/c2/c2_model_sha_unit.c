#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int64_t et_c2_private_model_sha256_v1(const void *input_bv,
                                      int64_t input_bytes, void *digest_bv);

typedef struct bytevector {
  int64_t length;
  uint8_t bytes[1056];
} bytevector;

static const struct {
  int64_t length;
  const char *hex;
} cases[] = {
    {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
    {1, "ab897fbdedfa502b2d839b6a56100887dccdc507555c282e59589e06300a62e2"},
    {55, "0c43a30970ee9525a6603d5656803c325bded148fa65bbd363df54cb3ac3dee5"},
    {56, "f4dcdc09c846d1b3621701516b300055c1e7875e216ba019dd4423f5a420f2d0"},
    {57, "579e08b0aa0a7cb68eed325a658ecc20557e144ffd5c5d9addaa84055fc51566"},
    {63, "ba3bb62c4369fe74e7b94392d0f04ed284a7b5d7295bfa48fabdb25c01d85ab7"},
    {64, "cb6862268ca7c02e7b6af558f0e3d236ce9ec7ff41cc746b2848c04c582da94e"},
    {65, "b68af55df9ce8ee60870e45d10e36b2e8363c9896f6e62478dcb6f593a4d7133"},
    {119, "175624f0f95b0a47a1b3b52f5b149f88802d3822b184325d0c20a1a7c132a687"},
    {120, "0598a3d8b5a82710dda54f05c4aa11fdd1c0aa5fafb216840fccbae323f07520"},
    {121, "22cd13d4f3251bfe1d793f401ddc6d489ee3421958920e873e7e0138c576a136"},
    {127, "a8365b6e6ed1daf56c41bae89ab089c03b671240ae103070480a7f271bdbad57"},
    {128, "d399f065c9f21e2fd51e2aeadb7768eab7e6e45e5150f31227c9711934a4d1d3"},
    {129, "d672408e0571636ea21092304eb31ba8fc1fbffc5d5ea6531d62b072714599db"},
    {1024, "e051d1007607de494c073da3c29903d6c0abfee7a4c0609f560a340a1947b470"},
};

static int hex_digit(char c) { return c <= '9' ? c - '0' : c - 'a' + 10; }

static int matches(const uint8_t digest[32], const char *hex) {
  size_t i;
  for (i = 0; i < 32u; ++i)
    if (digest[i] != (uint8_t)((hex_digit(hex[2u * i]) << 4u) |
                                hex_digit(hex[2u * i + 1u])))
      return 0;
  return 1;
}

#define CHECK(x) do { if (!(x)) { \
  fprintf(stderr, "C2 model SHA check failed at line %d\n", __LINE__); \
  return 1; \
} } while (0)

int main(int argc, char **argv) {
  bytevector input = {0, {0}}, output = {32, {0}};
  bytevector original, before_alias;
  size_t i;
  for (i = 0; i < sizeof(input.bytes); ++i)
    input.bytes[i] = (uint8_t)((i * 73u + 19u) & 255u);
  original = input;
  for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    input.length = cases[i].length;
    memset(output.bytes, 0xa5, sizeof(output.bytes));
    CHECK(et_c2_private_model_sha256_v1(&input, input.length, &output) == 0);
    CHECK(output.length == 32 && matches(output.bytes, cases[i].hex));
    CHECK(input.length == cases[i].length &&
          memcmp(input.bytes, original.bytes, sizeof(input.bytes)) == 0);
    for (size_t j = 32u; j < sizeof(output.bytes); ++j)
      CHECK(output.bytes[j] == 0xa5);
  }
  input.length = 1024;
  memset(output.bytes, 0xa5, sizeof(output.bytes));
  CHECK(et_c2_private_model_sha256_v1(NULL, 0, &output) == 1);
  CHECK(et_c2_private_model_sha256_v1(&input, 0, NULL) == 1);
  CHECK(et_c2_private_model_sha256_v1(&input, -1, &output) == 1);
  CHECK(et_c2_private_model_sha256_v1(&input, 55, &output) == 1);
  CHECK(et_c2_private_model_sha256_v1(&input, 1025, &output) == 1);
  output.length = 31;
  CHECK(et_c2_private_model_sha256_v1(&input, 1024, &output) == 1);
  output.length = 33;
  CHECK(et_c2_private_model_sha256_v1(&input, 1024, &output) == 1);
  output.length = 32;
  input.length = -1;
  CHECK(et_c2_private_model_sha256_v1(&input, 0, &output) == 1);
  input.length = INT64_MAX;
  CHECK(et_c2_private_model_sha256_v1(&input, INT64_MAX, &output) == 1);
  input.length = 32;
  before_alias = input;
  CHECK(et_c2_private_model_sha256_v1(&input, 32, &input) == 1);
  CHECK(memcmp(&input, &before_alias, sizeof(input)) == 0);
  input.length = 1024;
  memcpy(input.bytes + 8, &output.length, sizeof(output.length));
  before_alias = input;
  CHECK(et_c2_private_model_sha256_v1(&input, 1024, input.bytes + 8) == 1);
  CHECK(memcmp(&input, &before_alias, sizeof(input)) == 0);
  /* An output in the input tail is still an alias of the owned bytevector. */
  memcpy(input.bytes + 600, &output.length, sizeof(output.length));
  before_alias = input;
  CHECK(et_c2_private_model_sha256_v1(&input, 1024, input.bytes + 600) == 1);
  CHECK(memcmp(&input, &before_alias, sizeof(input)) == 0);
  for (i = 0; i < sizeof(output.bytes); ++i) CHECK(output.bytes[i] == 0xa5);
  CHECK(output.length == 32);

  if (argc == 3) {
    FILE *fixture = fopen(argv[1], "rb");
    int trailing;
    CHECK(fixture != NULL);
    input.length = (int64_t)fread(input.bytes, 1u, sizeof(input.bytes), fixture);
    trailing = fgetc(fixture);
    CHECK(ferror(fixture) == 0 && trailing == EOF && fclose(fixture) == 0);
    before_alias = input;
    CHECK(strlen(argv[2]) == 64u);
    CHECK(et_c2_private_model_sha256_v1(&input, input.length, &output) == 0);
    CHECK(matches(output.bytes, argv[2]));
    CHECK(memcmp(&input, &before_alias, sizeof(input)) == 0);
  } else CHECK(argc == 1);
  puts("c2 private model SHA: PASS");
  return 0;
}
