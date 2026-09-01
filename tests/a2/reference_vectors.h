#ifndef ESHKOL_TRANSFORMER_A2_REFERENCE_VECTORS_H
#define ESHKOL_TRANSFORMER_A2_REFERENCE_VECTORS_H

#include <stdint.h>

#define ET_A2_REFERENCE_FIXTURE_SHA256 \
  "6885dd74f4b79bfdba2f045bdb5c54fd5e39708c3479ef3c02c3f993633cd2f3"

static const uint32_t et_a2_ref_gqa_output[] = {
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e4afd70), UINT32_C(0x3e98b1eb), UINT32_C(0), UINT32_C(0),
  UINT32_C(0x3dcccccd), UINT32_C(0x3e4ccccd), UINT32_C(0x3e475fe0), UINT32_C(0x3e96e324), UINT32_C(0), UINT32_C(0),
  UINT32_C(0x3f333333), UINT32_C(0x3f4ccccd), UINT32_C(0x3f4a8b0a), UINT32_C(0x3f6424a4), UINT32_C(0), UINT32_C(0),
  UINT32_C(0x3f333333), UINT32_C(0x3f4ccccd), UINT32_C(0x3f4ccccd), UINT32_C(0x3f666666), UINT32_C(0), UINT32_C(0),
};
static const uint32_t et_a2_ref_gqa_dq[] = {
  UINT32_C(0), UINT32_C(0), UINT32_C(0x3ba22483), UINT32_C(0x3ba22484), UINT32_C(0), UINT32_C(0),
  UINT32_C(0), UINT32_C(0), UINT32_C(0x3b0aa1f3), UINT32_C(0x3b0aa1f2), UINT32_C(0), UINT32_C(0),
  UINT32_C(0), UINT32_C(0), UINT32_C(0x3ba0eec7), UINT32_C(0x3ba0eece), UINT32_C(0), UINT32_C(0),
  UINT32_C(0), UINT32_C(0), UINT32_C(0x3b0b05db), UINT32_C(0x3b0b05db), UINT32_C(0), UINT32_C(0),
};
static const uint32_t et_a2_ref_gqa_dk[] = {
  UINT32_C(0x3b3bfc0a), UINT32_C(0x3a2d4a6f), UINT32_C(0xbb3bfc09), UINT32_C(0xba2d4a6d), UINT32_C(0), UINT32_C(0),
  UINT32_C(0x3b2b6a2b), UINT32_C(0x3ba5bff6), UINT32_C(0xbb2b6a45), UINT32_C(0xbba5bfff), UINT32_C(0), UINT32_C(0),
};
static const uint32_t et_a2_ref_gqa_dv[] = {
  UINT32_C(0x3e9af4fa), UINT32_C(0x3ee8a950), UINT32_C(0x3dc75f4c), UINT32_C(0x3e1513c6), UINT32_C(0), UINT32_C(0),
  UINT32_C(0x3e9cfc3d), UINT32_C(0x3eeae9ea), UINT32_C(0x3dbf4240), UINT32_C(0x3e109291), UINT32_C(0), UINT32_C(0),
};
static const uint32_t et_a2_ref_rope_output[] = {
  UINT32_C(0x3f800000), UINT32_C(0x40000000), UINT32_C(0xc0400000), UINT32_C(0x40800000),
  UINT32_C(0xbecaac61), UINT32_C(0xbeca1908), UINT32_C(0xbf4d061f), UINT32_C(0xc0059d4d),
};
static const uint32_t et_a2_ref_rope_dx[] = {
  UINT32_C(0xbf000000), UINT32_C(0x3e800000), UINT32_C(0x3fc00000), UINT32_C(0xc0000000),
  UINT32_C(0xbfb6345e), UINT32_C(0x3ea0df56), UINT32_C(0xbe312870), UINT32_C(0xbef89142),
};

#endif
