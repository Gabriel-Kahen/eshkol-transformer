/* Focused data-codec execution; independent OpenSSL SHA oracle and fixed input. */
#define ET_CPU4_C4_HISTORY_PRIVATE 1
#define ET_CPU4_C4_P3_WIRE_PRIVATE 1
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include "cpu4_c4_capture_private.h"

enum { CPU4_CAPTURE_VALUES = 4768, ET_G3C4_INVALID_ARGUMENT = 1,
       ET_G3C4_CODE_ALIAS = 2, ET_G3C4_CODE_IDENTITY = 3 };
typedef struct cpu4_capture_payload {
  et_cpu4_c4_capture_record_v1 record;
  unsigned char values[CPU4_CAPTURE_VALUES];
} cpu4_capture_payload;
typedef struct cpu4_sha256_state { uint8_t bytes[512]; size_t used; } cpu4_sha256_state;
static void cpu4_sha256_init(cpu4_sha256_state *s) { s->used = 0u; }
static void cpu4_sha256_update(cpu4_sha256_state *s, const uint8_t *p, size_t n) {
  if (n > sizeof(s->bytes) - s->used) abort();
  memcpy(s->bytes + s->used, p, n); s->used += n;
}
static void cpu4_sha256_final(cpu4_sha256_state *s, uint8_t d[32]) {
  SHA256(s->bytes, s->used, d);
}
static void cpu4_capture_le16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8u);
}
static void cpu4_capture_le32(uint8_t *p, uint32_t v) {
  for (unsigned i = 0; i < 4u; ++i) p[i] = (uint8_t)(v >> (8u*i));
}
static void cpu4_capture_le64(uint8_t *p, uint64_t v) {
  for (unsigned i = 0; i < 8u; ++i) p[i] = (uint8_t)(v >> (8u*i));
}
static int et_g3c4_range_valid(const void *p, size_t n) {
  return p != NULL && (uintptr_t)p <= UINTPTR_MAX - n;
}
static int et_g3c4_ranges_overlap(const void *a, size_t an,
                                   const void *b, size_t bn) {
  return (uintptr_t)a < (uintptr_t)b + bn &&
         (uintptr_t)b < (uintptr_t)a + an;
}
static uint8_t tls[400];
static int cpu4_capture_tls_alias(const void *p, uint64_t n, size_t fixed) {
  (void)fixed;
  return n <= SIZE_MAX && et_g3c4_range_valid(p, (size_t)n) &&
         et_g3c4_ranges_overlap(p, (size_t)n, tls, sizeof(tls));
}
static int64_t cpu4_capture_copy_alias(const void *p, size_t n) {
  return cpu4_capture_tls_alias(p, n, n) ? ET_G3C4_INVALID_ARGUMENT : 0;
}
static void et_g3c4_error_reset_internal(void) {}
static int64_t et_g3c4_fail(int64_t category, int64_t code) {
  (void)code; return category;
}
static uint8_t expected_capture_digest[32];
static int cpu4_capture_record_valid(const cpu4_capture_payload *p,
                                     const unsigned char seal[32]) {
  for (size_t i = 0u; i < sizeof(p->values); ++i)
    if (p->values[i] != 0u) return 0;
  return p->record.policy[5] == 41 &&
         memcmp(p->record.digest, expected_capture_digest, 32u) == 0 &&
         memcmp(seal, expected_capture_digest, 32u) == 0;
}
#include "cpu4_c4_p3_wire.inc"

static size_t checks;
#define CHECK(x) do { ++checks; if (!(x)) { \
  fprintf(stderr,"wire check failed at %d: %s\n",__LINE__,#x); abort(); \
} } while (0)
static void from_hex(const char *hex, uint8_t *out, size_t n) {
  for (size_t i = 0u; i < n; ++i) {
    unsigned value;
    if (sscanf(hex + 2u*i, "%2x", &value) != 1) abort();
    out[i] = (uint8_t)value;
  }
}
static void digest_wire(uint8_t wire[400]) {
  SHA256(wire, 368u, wire + 368u);
}
static void rejected(const uint8_t wire[400], const uint8_t values[4768],
                     const uint8_t c2[32], const uint8_t t1[96]) {
  et_cpu4_c4_capture_record_v1 out;
  memset(&out, 0xa5, sizeof(out));
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,400u,values,4768u,c2,t1,&out)
        == ET_G3C4_INVALID_ARGUMENT);
  for (size_t i = 0u; i < sizeof(out); ++i)
    CHECK(((const uint8_t *)&out)[i] == 0xa5u);
}
int main(void) {
  static const char capture_hex[] =
      "cce6f6fb7768bde5a18404adcb2656a993e2f4cb3b5b0a9a5a27c526e00b8ee0";
  static const char wire_hex[] =
      "9d4cd70114a1a244e192e4c19d1183d2510f007b2a1f405c258e97d414d20883";
  static const char outer_hex[] =
      "86534e590e84fd563744af7f5ca346962c52702e3d605aee426bc6bd8ea113da";
  et_cpu4_c4_capture_record_v1 r = {0}, out;
  uint8_t values[4768] = {0}, c2[32], t1[96], wire[400], bad[400], hash[32], expected[32];
  from_hex(capture_hex, expected_capture_digest, 32u);
  memcpy(t1, cpu4_p3_t1, 96u);
  for (unsigned i = 0u; i < 32u; ++i) c2[i] = (uint8_t)i;
  r.size=224u; r.major=1u; r.count=3u; r.cache_capacity=4u;
  r.prompt_count=2u; r.generated_count=1u; r.commit_epoch=2u;
  r.ids[0]=3; r.ids[1]=197; r.ids[2]=3;
  r.origins[0]=1u; r.origins[1]=1u; r.origins[2]=2u;
  r.policy[1]=INT64_C(0x3f800000); r.policy[2]=256;
  r.policy[3]=INT64_C(0x3f800000); r.policy[4]=2; r.policy[5]=41;
  r.rng[0]=1; r.rng[1]=41; r.parameter_count=14u;
  r.values_bytes=4768u; r.state=1u;
  memcpy(r.digest, expected_capture_digest, 32u);
  memset(wire, 0xa5, sizeof(wire));
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,wire,400u)==0);
  SHA256(wire,sizeof(wire),hash); from_hex(wire_hex,expected,32u);
  CHECK(memcmp(hash,expected,32u)==0);
  from_hex(outer_hex,expected,32u);
  CHECK(memcmp(wire+368u,expected,32u)==0);
  CHECK(memcmp(wire,cpu4_p3_magic,8u)==0);
  CHECK(memcmp(wire+240u,c2,32u)==0 && memcmp(wire+272u,t1,96u)==0);
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,400u,values,4768u,c2,t1,&out)==0);
  CHECK(memcmp(&r,&out,sizeof(r))==0);
  memset(bad,0xa5,sizeof(bad));
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,bad,399u)
        == ET_G3C4_INVALID_ARGUMENT);
  for (size_t i=0u;i<sizeof(bad);++i) CHECK(bad[i]==0xa5u);
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4767u,c2,t1,bad,400u)
        == ET_G3C4_INVALID_ARGUMENT);
  for (size_t i=0u;i<sizeof(bad);++i) CHECK(bad[i]==0xa5u);
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,bad,401u)
        == ET_G3C4_INVALID_ARGUMENT);
  for (size_t i=0u;i<sizeof(bad);++i) CHECK(bad[i]==0xa5u);
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,399u,values,4768u,c2,t1,&out)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,401u,values,4768u,c2,t1,&out)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,400u,values,4767u,c2,t1,&out)
        == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_cpu4_private_c4_p3_wire_decode_v1(wire,400u,values,4768u,c2,t1,
        (et_cpu4_c4_capture_record_v1 *)wire) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,
        (void *)values,400u) == ET_G3C4_INVALID_ARGUMENT);
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,tls,400u)
        == ET_G3C4_INVALID_ARGUMENT);
  memcpy(bad,wire,sizeof(bad)); bad[0]^=1u; rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[8]^=1u; rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[12]^=1u; rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[36]^=1u; digest_wire(bad); rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[48]^=1u; digest_wire(bad); rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[208]^=1u; digest_wire(bad); rejected(bad,values,c2,t1);
  memcpy(bad,wire,sizeof(bad)); bad[368]^=1u; rejected(bad,values,c2,t1);
  c2[0]^=1u; rejected(wire,values,c2,t1); c2[0]^=1u;
  t1[0]^=1u; rejected(wire,values,c2,t1); t1[0]^=1u;
  values[0]^=1u; rejected(wire,values,c2,t1); values[0]^=1u;
  r.policy[5]=255; /* Same capture digest is now invalid; output stays intact. */
  memset(bad,0xa5,sizeof(bad));
  CHECK(et_cpu4_private_c4_p3_wire_encode_v1(&r,values,4768u,c2,t1,bad,400u)
        == ET_G3C4_INVALID_ARGUMENT);
  for (size_t i=0u;i<sizeof(bad);++i) CHECK(bad[i]==0xa5u);
  printf("CPU4 C4 P3 wire native PASS checks=%zu\n",checks);
  return 0;
}
