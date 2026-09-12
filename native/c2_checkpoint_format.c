#include "c2_checkpoint_format.h"
#include "c2_x1_canonical.h"

#include <limits.h>
#include <string.h>

#define C2_HEADER_BYTES UINT64_C(256)
#define C2_CHECKSUM_BYTES UINT64_C(32)
#define C1_HEADER_BYTES UINT64_C(128)
#define C1_RECORD_BYTES UINT64_C(80)
#define O2_HEADER_BYTES UINT64_C(128)
#define O2_CONFIG_BYTES UINT64_C(64)
#define I64_MAX_U UINT64_C(9223372036854775807)

static const uint8_t c2_magic[16] = {0x89, 0x45, 0x53, 0x48, 0x4b, 0x54,
                                     0x52, 0x4e, 0x53, 0x54, 0x41, 0x54,
                                     0x45, 0x0d, 0x0a, 0x00};
static const uint8_t c1_magic[16] = {0x89, 0x45, 0x53, 0x48, 0x4b, 0x4f,
                                     0x4c, 0x43, 0x4b, 0x50, 0x54, 0x0d,
                                     0x0a, 0x1a, 0x0a, 0x00};
static const uint8_t optimizer_magic[8] = {'E', 'S', 'H', 'K',
                                           'O', 'P', 'T', '1'};
static const uint8_t provider_id[] = "i2-dense-cpu-f32-v1";
static const uint8_t library_id[] =
    "eshkol-transformer\0"
    "0.1.0-draft\0"
    "eshkol-training-state:1.0\0";
static const uint8_t compiler_id[] =
    "Eshkol Compiler v1.3.4-evolve\0"
    "90cbd7130f47b8184bcc77b8d5c1b0026da980de\0";
static const uint8_t c2_domain[] = "eshkol-training-state-container-v1\0";
static const uint8_t c1_domain[] = "eshkol-checkpoint-container-v1\0";
static const uint8_t c1_tensor_domain[] = "eshkol-checkpoint-tensor-v1\0";
static const uint8_t moment_domain[] = "eshkol-training-state-moment-v1\0";
static const uint8_t cursor_domain[] =
    "eshkol-token-dataset-cursor-checksum-v1\n";

typedef struct sha256_state {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} sha256_state;

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static void sha256_block(sha256_state *state, const uint8_t block[64]) {
  static const uint32_t k[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
      0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
      0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
      0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
      0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
      0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
      0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
      0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
      0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
      0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
      0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
      0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
  uint32_t w[64];
  uint32_t a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) |
           ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 = rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^
                        (w[i - 15u] >> 3u);
    const uint32_t s1 = rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^
                        (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  a = state->h[0]; b = state->h[1]; c = state->h[2]; d = state->h[3];
  e = state->h[4]; f = state->h[5]; g = state->h[6]; h = state->h[7];
  for (i = 0; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
  }
  state->h[0] += a; state->h[1] += b; state->h[2] += c; state->h[3] += d;
  state->h[4] += e; state->h[5] += f; state->h[6] += g; state->h[7] += h;
}

static void sha256_init(sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u,
                                      0x3c6ef372u, 0xa54ff53au,
                                      0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(state->h, initial, sizeof(initial));
  state->length = 0u;
  state->used = 0u;
}

static void sha256_update(sha256_state *state, const uint8_t *bytes,
                          size_t length) {
  state->length += (uint64_t)length;
  while (length != 0u) {
    size_t take = 64u - state->used;
    if (take > length) take = length;
    memcpy(state->block + state->used, bytes, take);
    state->used += take; bytes += take; length -= take;
    if (state->used == 64u) {
      sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void sha256_final(sha256_state *state, uint8_t digest[32]) {
  uint64_t bits = state->length * UINT64_C(8);
  size_t i;
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (i = 0; i < 8u; ++i) state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static uint16_t u16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8u));
}
static uint32_t u32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) |
         ((uint32_t)p[2] << 16u) | ((uint32_t)p[3] << 24u);
}
static uint64_t u64(const uint8_t *p) {
  uint64_t value = 0u;
  unsigned i;
  for (i = 0; i < 8u; ++i) value |= (uint64_t)p[i] << (8u * i);
  return value;
}

typedef struct parser {
  const uint8_t *bytes;
  size_t size;
  const et_c2_checkpoint_limits_v1 *limits;
  uint32_t mode;
  et_c2_checkpoint_format_error_v1 *error;
} parser;

static int32_t fail(parser *p, uint32_t category, uint32_t code,
                    uint64_t offset) {
  if (p->error != NULL) {
    p->error->category = category;
    p->error->code = code;
    p->error->offset = offset;
  }
  return (int32_t)category;
}

static int add_u64(uint64_t a, uint64_t b, uint64_t *result) {
  if (a > UINT64_MAX - b) return 0;
  *result = a + b;
  return 1;
}
static int mul_u64(uint64_t a, uint64_t b, uint64_t *result) {
  if (a != 0u && b > UINT64_MAX / a) return 0;
  *result = a * b;
  return 1;
}
static int span_fits(uint64_t offset, uint64_t length, uint64_t end) {
  return offset <= end && length <= end - offset;
}
static int zero_bytes(const uint8_t *bytes, size_t count) {
  size_t i;
  for (i = 0; i < count; ++i) if (bytes[i] != 0u) return 0;
  return 1;
}

static int utf8_valid(const uint8_t *s, size_t n) {
  size_t i = 0u;
  while (i < n) {
    const uint8_t a = s[i++];
    if (a <= 0x7fu) continue;
    if (a >= 0xc2u && a <= 0xdfu) {
      if (i >= n || (s[i++] & 0xc0u) != 0x80u) return 0;
    } else if (a == 0xe0u) {
      if (i + 1u >= n || s[i] < 0xa0u || s[i] > 0xbfu ||
          (s[i + 1u] & 0xc0u) != 0x80u) return 0;
      i += 2u;
    } else if ((a >= 0xe1u && a <= 0xecu) || (a >= 0xeeu && a <= 0xefu)) {
      if (i + 1u >= n || (s[i] & 0xc0u) != 0x80u ||
          (s[i + 1u] & 0xc0u) != 0x80u) return 0;
      i += 2u;
    } else if (a == 0xedu) {
      if (i + 1u >= n || s[i] < 0x80u || s[i] > 0x9fu ||
          (s[i + 1u] & 0xc0u) != 0x80u) return 0;
      i += 2u;
    } else if (a == 0xf0u) {
      if (i + 2u >= n || s[i] < 0x90u || s[i] > 0xbfu ||
          (s[i + 1u] & 0xc0u) != 0x80u || (s[i + 2u] & 0xc0u) != 0x80u)
        return 0;
      i += 3u;
    } else if (a >= 0xf1u && a <= 0xf3u) {
      if (i + 2u >= n || (s[i] & 0xc0u) != 0x80u ||
          (s[i + 1u] & 0xc0u) != 0x80u || (s[i + 2u] & 0xc0u) != 0x80u)
        return 0;
      i += 3u;
    } else if (a == 0xf4u) {
      if (i + 2u >= n || s[i] < 0x80u || s[i] > 0x8fu ||
          (s[i + 1u] & 0xc0u) != 0x80u || (s[i + 2u] & 0xc0u) != 0x80u)
        return 0;
      i += 3u;
    } else return 0;
  }
  return 1;
}

static int lower_hex64(const uint8_t *p) {
  size_t i;
  for (i = 0; i < 64u; ++i)
    if (!((p[i] >= '0' && p[i] <= '9') || (p[i] >= 'a' && p[i] <= 'f')))
      return 0;
  return 1;
}

static int fingerprint_valid(const uint8_t *p, size_t n, int tokenizer) {
  static const uint8_t byte_prefix[] = "sha256:eshkol-byte-tokenizer-v1:";
  static const uint8_t bpe_prefix[] = "sha256:eshkol-bpe-tokenizer-v1:";
  static const uint8_t config_prefix[] = "sha256:eshkol-config-json-v1:";
  if (tokenizer) {
    if (n == 96u && memcmp(p, byte_prefix, sizeof(byte_prefix) - 1u) == 0)
      return lower_hex64(p + sizeof(byte_prefix) - 1u);
    if (n == 95u && memcmp(p, bpe_prefix, sizeof(bpe_prefix) - 1u) == 0)
      return lower_hex64(p + sizeof(bpe_prefix) - 1u);
    return 0;
  }
  return n == 93u && memcmp(p, config_prefix, sizeof(config_prefix) - 1u) == 0 &&
         lower_hex64(p + sizeof(config_prefix) - 1u);
}

static int digest_matches(const uint8_t *domain, size_t domain_bytes,
                          const uint8_t *bytes, size_t bytes_count,
                          const uint8_t expected[32]) {
  sha256_state state;
  uint8_t digest[32];
  sha256_init(&state);
  sha256_update(&state, domain, domain_bytes);
  sha256_update(&state, bytes, bytes_count);
  sha256_final(&state, digest);
  return memcmp(digest, expected, 32u) == 0;
}

typedef struct c1_record {
  uint64_t start, end, payload_offset, payload_bytes, path_offset, path_bytes;
  uint64_t element_count;
  uint16_t segment_count, rank;
  uint8_t kind, dtype;
} c1_record;

typedef struct c1_info {
  uint64_t base, bytes, metadata_bytes, payload_offset, payload_bytes;
  uint64_t aliases_offset;
  uint64_t maximum_tensor_bytes;
  uint32_t entries, alias_groups, unique_parameters;
} c1_info;

static int c1_record_at(const parser *p, const c1_info *c1, uint32_t wanted,
                        c1_record *record) {
  uint64_t at = c1->base + C1_HEADER_BYTES + 19u;
  uint32_t i;
  for (i = 0; i <= wanted; ++i) {
    const uint8_t *r = p->bytes + at;
    const uint64_t length = u64(r);
    if (i == wanted) {
      record->start = at; record->end = at + length;
      record->payload_offset = u64(r + 8u); record->payload_bytes = u64(r + 16u);
      record->path_bytes = u32(r + 24u); record->segment_count = u16(r + 28u);
      record->rank = u16(r + 30u); record->kind = r[32]; record->dtype = r[33];
      record->element_count = u64(r + 40u);
      record->path_offset = at + C1_RECORD_BYTES + UINT64_C(8) * record->rank;
      return 1;
    }
    at += length;
  }
  return 0;
}

static int path_compare(const uint8_t *bytes, uint64_t a, uint16_t ac,
                        uint64_t b, uint16_t bc) {
  uint16_t i = 0u;
  while (i < ac && i < bc) {
    uint32_t an = u32(bytes + a), bn = u32(bytes + b);
    size_t common = an < bn ? an : bn;
    int cmp = memcmp(bytes + a + 4u, bytes + b + 4u, common);
    if (cmp != 0) return cmp < 0 ? -1 : 1;
    if (an != bn) return an < bn ? -1 : 1;
    a += UINT64_C(4) + an; b += UINT64_C(4) + bn; ++i;
  }
  if (ac == bc) return 0;
  return ac < bc ? -1 : 1;
}

static int path_validate(const parser *p, uint64_t start, uint64_t bytes_count,
                         uint16_t segments) {
  uint64_t at = start, end = start + bytes_count;
  uint16_t i;
  if (segments == 0u || segments > 64u) return 0;
  for (i = 0; i < segments; ++i) {
    uint32_t n;
    if (!span_fits(at, 4u, end)) return 0;
    n = u32(p->bytes + at); at += 4u;
    if (n == 0u || n > 65536u || !span_fits(at, n, end) ||
        !utf8_valid(p->bytes + at, n)) return 0;
    at += n;
  }
  return at == end;
}

static int c1_alias_member(const parser *p, const c1_info *c1, uint32_t entry,
                           int include_first) {
  uint64_t at = c1->aliases_offset;
  uint32_t g;
  for (g = 0; g < c1->alias_groups; ++g) {
    uint32_t count = u32(p->bytes + at), i;
    for (i = include_first ? 0u : 1u; i < count; ++i)
      if (u32(p->bytes + at + 8u + UINT64_C(4) * i) == entry) return 1;
    at += UINT64_C(8) + UINT64_C(4) * count;
  }
  return 0;
}

static int32_t validate_c1(parser *p, uint64_t base, uint64_t declared_bytes,
                           uint64_t outer_metadata_bytes, c1_info *out) {
  const uint8_t *h;
  uint64_t file_bytes, metadata_bytes, payload_offset, payload_bytes, unsigned_end;
  uint64_t at, expected_payload = 0u, alias_members = 0u;
  uint32_t entries, aliases, provider_bytes, i, parameter_count = 0u;
  c1_record previous;
  int have_previous = 0;
  if (!span_fits(base, C1_HEADER_BYTES + C2_CHECKSUM_BYTES, p->size))
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_TRUNCATED, base);
  h = p->bytes + base;
  if (memcmp(h, c1_magic, sizeof(c1_magic)) != 0)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_MAGIC, base);
  if (u16(h + 16u) != 1u || u16(h + 18u) != 0u || u64(h + 32u) != 0u ||
      u16(h + 92u) != 1u || u16(h + 94u) != 0u || u16(h + 96u) != 2u ||
      u16(h + 98u) != 0u)
    return fail(p, ET_C2_FORMAT_VERSION_MISMATCH, ET_C2_FORMAT_CODE_VERSION,
                base + 16u);
  if (u32(h + 20u) != 128u || u32(h + 24u) != 1u || u32(h + 28u) != 1u)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_FIXED_FIELD,
                base + 20u);
  if (!zero_bytes(h + 100u, 28u))
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_RESERVED,
                base + 100u);
  file_bytes = u64(h + 40u); metadata_bytes = u64(h + 56u);
  payload_offset = u64(h + 64u); payload_bytes = u64(h + 72u);
  entries = u32(h + 80u); aliases = u32(h + 84u); provider_bytes = u32(h + 88u);
  if (file_bytes != declared_bytes || file_bytes > ET_C2_WIRE_MAX_FILE_BYTES ||
      u64(h + 48u) != 128u || metadata_bytes > ET_C2_WIRE_MAX_METADATA_BYTES ||
      entries > 4096u || aliases > 4096u || provider_bytes != 19u ||
      !add_u64(128u, metadata_bytes, &at) || payload_offset != at ||
      !add_u64(payload_offset, payload_bytes, &unsigned_end) ||
      !add_u64(unsigned_end, 32u, &at) || at != file_bytes)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, base + 40u);
  if (outer_metadata_bytes > p->limits->maximum_metadata_bytes ||
      metadata_bytes > p->limits->maximum_metadata_bytes - outer_metadata_bytes)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_LIMIT, base + 56u);
  if (p->limits->enforce_operational_profile &&
      (outer_metadata_bytes > ET_C2_PROFILE_MAX_METADATA_BYTES ||
       metadata_bytes > ET_C2_PROFILE_MAX_METADATA_BYTES - outer_metadata_bytes))
    return fail(p, ET_C2_FORMAT_UNSUPPORTED, ET_C2_FORMAT_CODE_PROFILE, base + 56u);
  if (!span_fits(base, file_bytes, p->size) ||
      !digest_matches(c1_domain, sizeof(c1_domain) - 1u, h,
                      (size_t)(file_bytes - 32u), h + file_bytes - 32u))
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CHECKSUM,
                base + file_bytes - 32u);
  if (memcmp(h + 128u, provider_id, 19u) != 0)
    return fail(p, utf8_valid(h + 128u, 19u) ? ET_C2_FORMAT_UNSUPPORTED :
                ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY, base + 128u);
  out->base = base; out->bytes = file_bytes; out->metadata_bytes = metadata_bytes;
  out->payload_offset = base + payload_offset; out->payload_bytes = payload_bytes;
  out->entries = entries; out->alias_groups = aliases;
  out->maximum_tensor_bytes = 0u;
  at = base + C1_HEADER_BYTES + provider_bytes;
  for (i = 0; i < entries; ++i) {
    const uint8_t *r;
    c1_record rec;
    uint64_t record_bytes, shape_bytes, expect_record, elements = 1u, tensor_bytes;
    uint16_t d;
    uint8_t width;
    uint8_t digest[32];
    sha256_state state;
    if (!span_fits(at, C1_RECORD_BYTES, base + payload_offset))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, at);
    r = p->bytes + at; record_bytes = u64(r); rec.rank = u16(r + 30u);
    rec.segment_count = u16(r + 28u); rec.path_bytes = u32(r + 24u);
    if (rec.rank > 64u || !mul_u64(rec.rank, 8u, &shape_bytes) ||
        !add_u64(C1_RECORD_BYTES, shape_bytes, &expect_record) ||
        !add_u64(expect_record, rec.path_bytes, &expect_record) ||
        record_bytes != expect_record || !span_fits(at, record_bytes, base + payload_offset))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, at);
    rec.start = at; rec.end = at + record_bytes; rec.payload_offset = u64(r + 8u);
    rec.payload_bytes = u64(r + 16u); rec.path_offset = at + 80u + shape_bytes;
    rec.kind = r[32]; rec.dtype = r[33]; rec.element_count = u64(r + 40u);
    if (rec.dtype < 1u || rec.dtype > 3u || (rec.kind == 1u && rec.dtype != 3u))
      return fail(p, ET_C2_FORMAT_DTYPE_MISMATCH, ET_C2_FORMAT_CODE_FIXED_FIELD,
                  at + 33u);
    if (r[34] != 1u)
      return fail(p, ET_C2_FORMAT_DEVICE_MISMATCH, ET_C2_FORMAT_CODE_FIXED_FIELD,
                  at + 34u);
    if (r[35] != 1u)
      return fail(p, ET_C2_FORMAT_NONCONTIGUOUS, ET_C2_FORMAT_CODE_FIXED_FIELD,
                  at + 35u);
    if (!zero_bytes(r + 36u, 4u) || (rec.kind != 1u && rec.kind != 2u) ||
        rec.payload_offset != expected_payload ||
        !path_validate(p, rec.path_offset, rec.path_bytes, rec.segment_count))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_FIXED_FIELD, at);
    if (have_previous && path_compare(p->bytes, previous.path_offset,
                                      previous.segment_count, rec.path_offset,
                                      rec.segment_count) >= 0)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ORDER,
                  rec.path_offset);
    for (d = 0; d < rec.rank; ++d) {
      uint64_t next;
      if (!mul_u64(elements, u64(r + 80u + UINT64_C(8) * d), &next))
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OVERFLOW,
                    at + 80u + UINT64_C(8) * d);
      elements = next;
    }
    width = rec.dtype == 1u ? 1u : (rec.dtype == 2u ? 8u : 4u);
    if (!mul_u64(elements, width, &tensor_bytes) || elements != rec.element_count ||
        tensor_bytes != rec.payload_bytes ||
        rec.payload_bytes > ET_C2_WIRE_MAX_TENSOR_BYTES)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SHAPE, at + 16u);
    if (rec.payload_bytes > out->maximum_tensor_bytes)
      out->maximum_tensor_bytes = rec.payload_bytes;
    if (!span_fits(out->payload_offset + rec.payload_offset, rec.payload_bytes,
                   out->payload_offset + payload_bytes))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, at + 8u);
    sha256_init(&state);
    sha256_update(&state, c1_tensor_domain, sizeof(c1_tensor_domain) - 1u);
    sha256_update(&state, r, 48u);
    { uint8_t zeros[32] = {0}; sha256_update(&state, zeros, sizeof(zeros)); }
    sha256_update(&state, r + 80u, (size_t)(record_bytes - 80u));
    sha256_update(&state, p->bytes + out->payload_offset + rec.payload_offset,
                  (size_t)rec.payload_bytes);
    sha256_final(&state, digest);
    if (memcmp(digest, r + 48u, 32u) != 0)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CHECKSUM, at + 48u);
    if (rec.dtype == 1u) {
      uint64_t q;
      for (q = 0; q < rec.payload_bytes; ++q) {
        uint8_t v = p->bytes[out->payload_offset + rec.payload_offset + q];
        if (v > 1u) return fail(p, ET_C2_FORMAT_CORRUPT_DATA,
                               ET_C2_FORMAT_CODE_FIXED_FIELD,
                               out->payload_offset + rec.payload_offset + q);
      }
    }
    if (rec.kind == 1u) ++parameter_count;
    if (!add_u64(expected_payload, rec.payload_bytes, &expected_payload))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OVERFLOW, at);
    previous = rec; have_previous = 1; at += record_bytes;
  }
  out->aliases_offset = at;
  { uint32_t previous_first = 0u; int have_first = 0;
    for (i = 0; i < aliases; ++i) {
      uint32_t count, j, first;
      if (!span_fits(at, 8u, base + payload_offset))
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS, at);
      count = u32(p->bytes + at);
      if (count < 2u || !zero_bytes(p->bytes + at + 4u, 4u) ||
          !span_fits(at + 8u, UINT64_C(4) * count, base + payload_offset))
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS, at);
      first = u32(p->bytes + at + 8u);
      if (have_first && first <= previous_first)
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS, at);
      for (j = 0; j < count; ++j) {
        uint32_t member = u32(p->bytes + at + 8u + UINT64_C(4) * j);
        c1_record rec, canonical;
        uint64_t earlier = out->aliases_offset;
        uint32_t eg;
        if (member >= entries || (j && member <= u32(p->bytes + at + 4u + UINT64_C(4) * j)) ||
            !c1_record_at(p, out, member, &rec) || rec.kind != 1u)
          return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS,
                      at + 8u + UINT64_C(4) * j);
        for (eg = 0; eg < i; ++eg) {
          uint32_t ec = u32(p->bytes + earlier), ek;
          for (ek = 0; ek < ec; ++ek)
            if (u32(p->bytes + earlier + 8u + UINT64_C(4) * ek) == member)
              return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS,
                          at + 8u + UINT64_C(4) * j);
          earlier += UINT64_C(8) + UINT64_C(4) * ec;
        }
        if (j != 0u) {
          c1_record_at(p, out, first, &canonical);
          if (rec.payload_bytes != canonical.payload_bytes ||
              memcmp(p->bytes + out->payload_offset + rec.payload_offset,
                     p->bytes + out->payload_offset + canonical.payload_offset,
                     (size_t)rec.payload_bytes) != 0)
            return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS,
                        at + 8u + UINT64_C(4) * j);
        }
      }
      alias_members += count; previous_first = first; have_first = 1;
      at += UINT64_C(8) + UINT64_C(4) * count;
    }
  }
  if (at != base + payload_offset || expected_payload != payload_bytes ||
      alias_members > 4096u || alias_members > parameter_count + aliases)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS, at);
  out->unique_parameters = parameter_count - (uint32_t)alias_members + aliases;
  return ET_C2_FORMAT_OK;
}

static int bits_finite(uint32_t bits) { return (bits & 0x7f800000u) != 0x7f800000u; }
static int bits_nonnegative(uint32_t bits) {
  return (bits & 0x80000000u) == 0u && bits_finite(bits);
}
static int bits_nonnegative_or_negative_zero(uint32_t bits) {
  return bits_finite(bits) &&
         ((bits & 0x80000000u) == 0u || (bits & 0x7fffffffu) == 0u);
}
static int bits_positive(uint32_t bits) {
  return bits_nonnegative(bits) && (bits & 0x7fffffffu) != 0u;
}

static int32_t fail_x1(parser *p, const et_c2_x1_error_v1 *x1_error,
                       uint64_t outer_offset) {
  uint32_t category = ET_C2_FORMAT_CORRUPT_DATA;
  uint32_t code = ET_C2_FORMAT_CODE_FIXED_FIELD;
  if (x1_error->code == ET_C2_X1_CODE_UTF8)
    code = ET_C2_FORMAT_CODE_UTF8;
  else if (x1_error->code == ET_C2_X1_CODE_FINGERPRINT)
    code = ET_C2_FORMAT_CODE_CHECKSUM;
  else if (x1_error->category == ET_C2_X1_VERSION_MISMATCH) {
    category = ET_C2_FORMAT_VERSION_MISMATCH;
    code = ET_C2_FORMAT_CODE_VERSION;
  } else if (x1_error->category == ET_C2_X1_DTYPE_MISMATCH)
    category = ET_C2_FORMAT_DTYPE_MISMATCH;
  else if (x1_error->category == ET_C2_X1_DEVICE_MISMATCH)
    category = ET_C2_FORMAT_DEVICE_MISMATCH;
  return fail(p, category, code, outer_offset);
}

static int options_valid(const uint8_t *p) {
  uint32_t lr = u32(p), b1 = u32(p + 4u), b2 = u32(p + 8u);
  return bits_nonnegative(lr) && bits_nonnegative(b1) && b1 < 0x3f800000u &&
         bits_nonnegative(b2) && b2 < 0x3f800000u &&
         bits_positive(u32(p + 12u)) && bits_nonnegative(u32(p + 16u));
}

static int32_t cursor_validate(parser *p, uint64_t offset, uint32_t length,
                               const uint8_t *fingerprint, uint32_t fp_length) {
  const uint8_t *c = p->bytes + offset;
  uint64_t n, t, manifest_limit, shard_limit, total_limit, batch_limit;
  uint64_t window, rows, ordinal, elements, batch_bytes, total;
  uint32_t shuffle, packing, seed_present;
  if (length != 208u + fp_length || memcmp(c, "ESHKDCU1", 8u) != 0 ||
      u32(c + 12u) != 176u + fp_length || u32(c + 36u) != fp_length ||
      u64(c + 160u) != length)
    return ET_C2_FORMAT_CORRUPT_DATA;
  if (u16(c + 8u) != 1u || u16(c + 10u) != 0u || u32(c + 16u) != 0u ||
      u32(c + 20u) != 1u) return ET_C2_FORMAT_VERSION_MISMATCH;
  if (!zero_bytes(c + 168u, 8u)) return ET_C2_FORMAT_CORRUPT_DATA;
  shuffle = u32(c + 24u); packing = u32(c + 28u); seed_present = u32(c + 32u);
  if (packing > 1u || seed_present > 1u || shuffle > 1u || shuffle != seed_present ||
      (seed_present == 0u && u64(c + 96u) != 0u) ||
      memcmp(c + 176u, fingerprint, fp_length) != 0 ||
      !utf8_valid(c + 176u, fp_length)) return ET_C2_FORMAT_CORRUPT_DATA;
  n = u64(c + 48u); t = u64(c + 56u); manifest_limit = u64(c + 64u);
  shard_limit = u64(c + 72u); total_limit = u64(c + 80u);
  batch_limit = u64(c + 88u); window = u64(c + 104u); rows = u64(c + 112u);
  ordinal = u64(c + 120u);
  if (u64(c + 40u) == 0u || u64(c + 40u) > I64_MAX_U ||
      n == 0u || n > I64_MAX_U || t == 0u || t > I64_MAX_U ||
      manifest_limit == 0u || manifest_limit > I64_MAX_U ||
      shard_limit == 0u || shard_limit > I64_MAX_U || total_limit > I64_MAX_U ||
      batch_limit == 0u || batch_limit > I64_MAX_U ||
      window == 0u || window > I64_MAX_U || rows > I64_MAX_U || ordinal > I64_MAX_U ||
      (seed_present && u64(c + 96u) > I64_MAX_U) ||
      n > I64_MAX_U / t) return ET_C2_FORMAT_CORRUPT_DATA;
  elements = n * t;
  if (elements > I64_MAX_U / 17u) return ET_C2_FORMAT_CORRUPT_DATA;
  batch_bytes = elements * 17u;
  if (batch_bytes > batch_limit || manifest_limit > I64_MAX_U - shard_limit)
    return ET_C2_FORMAT_CORRUPT_DATA;
  total = manifest_limit + shard_limit;
  if (total > I64_MAX_U - batch_bytes) return ET_C2_FORMAT_CORRUPT_DATA;
  total += batch_bytes;
  if ((window < rows ? window : rows) > I64_MAX_U / 8u ||
      total > I64_MAX_U - UINT64_C(8) * (window < rows ? window : rows))
    return ET_C2_FORMAT_CORRUPT_DATA;
  return digest_matches(cursor_domain, sizeof(cursor_domain) - 1u, c,
                        176u + fp_length, c + 176u + fp_length) ?
             ET_C2_FORMAT_OK : ET_C2_FORMAT_CORRUPT_DATA;
}

static int canonical_parameter(const parser *p, const c1_info *c1,
                               uint32_t ordinal, c1_record *record,
                               uint32_t *entry_index) {
  uint32_t i, found = 0u;
  for (i = 0; i < c1->entries; ++i) {
    c1_record candidate;
    c1_record_at(p, c1, i, &candidate);
    if (candidate.kind == 1u && !c1_alias_member(p, c1, i, 0)) {
      if (found == ordinal) { *record = candidate; *entry_index = i; return 1; }
      ++found;
    }
  }
  return 0;
}

static int32_t validate_optimizer(parser *p, uint64_t metadata,
                                  uint64_t metadata_bytes, uint64_t payload,
                                  uint64_t payload_bytes, const c1_info *c1,
                                  uint32_t groups, uint64_t completed,
                                  uint64_t *maximum_tensor_bytes) {
  const uint8_t *h = p->bytes + metadata;
  uint32_t count = c1->unique_parameters, i;
  uint64_t config_bytes, table_bytes, group_table_bytes, at, table, expected = 0u;
  if (metadata_bytes < O2_HEADER_BYTES + 19u + O2_CONFIG_BYTES ||
      memcmp(h, optimizer_magic, 8u) != 0)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, metadata);
  if (u16(h + 8u) != 1u || u16(h + 10u) != 0u || u64(h + 16u) != 0u ||
      u16(h + 40u) != 2u || u16(h + 42u) != 0u)
    return fail(p, ET_C2_FORMAT_VERSION_MISMATCH, ET_C2_FORMAT_CODE_VERSION,
                metadata + 8u);
  if (u32(h + 12u) != 128u || u32(h + 24u) != 1u || u32(h + 44u) != 19u ||
      u32(h + 48u) != groups || u32(h + 52u) != count ||
      u32(h + 56u) != 2u * count || u32(h + 60u) != 0u ||
      u64(h + 64u) != completed || u64(h + 88u) != payload_bytes ||
      u64(h + 96u) != 147u || !zero_bytes(h + 120u, 8u))
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER,
                metadata + 12u);
  if (u32(h + 28u) != 3u)
    return fail(p, ET_C2_FORMAT_DTYPE_MISMATCH, ET_C2_FORMAT_CODE_OPTIMIZER,
                metadata + 28u);
  if (u32(h + 32u) != 1u)
    return fail(p, ET_C2_FORMAT_DEVICE_MISMATCH, ET_C2_FORMAT_CODE_OPTIMIZER,
                metadata + 32u);
  if (u32(h + 36u) != 1u)
    return fail(p, ET_C2_FORMAT_NONCONTIGUOUS, ET_C2_FORMAT_CODE_OPTIMIZER,
                metadata + 36u);
  if (memcmp(h + 128u, provider_id, 19u) != 0) {
    size_t provider_index;
    int grammar = 1;
    for (provider_index = 0u; provider_index < 19u; ++provider_index) {
      const uint8_t c = h[128u + provider_index];
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
        grammar = 0;
    }
    return fail(p, grammar ? ET_C2_FORMAT_UNSUPPORTED : ET_C2_FORMAT_CORRUPT_DATA,
                ET_C2_FORMAT_CODE_IDENTITY, metadata + 128u);
  }
  config_bytes = u64(h + 72u); table_bytes = u64(h + 80u);
  if (!add_u64(147u, config_bytes, &table) || u64(h + 104u) != table ||
      !add_u64(table, table_bytes, &at) || u64(h + 112u) != at || at != metadata_bytes)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, metadata + 72u);
  { const uint8_t *c = h + 147u;
    uint32_t clip = u32(c + 4u), schedule = u32(c + 12u), ratio = u32(c + 16u);
    group_table_bytes = u64(c + 48u);
    if (u32(c) != 64u || !zero_bytes(c + 20u, 4u) ||
        u32(c + 40u) != groups || u32(c + 44u) != count ||
        !zero_bytes(c + 56u, 8u) || config_bytes != 64u + group_table_bytes ||
        (clip == 0u ? u32(c + 8u) != 0u :
         (clip != 1u || !bits_positive(u32(c + 8u)))) ||
        (schedule == 0u ? (ratio != 0x3f800000u || u64(c + 24u) != 0u ||
                           u64(c + 32u) != 0u) :
         (schedule != 1u || u64(c + 32u) == 0u || u64(c + 32u) > I64_MAX_U ||
          u64(c + 24u) >= u64(c + 32u) || !bits_nonnegative(ratio) ||
          ratio > 0x3f800000u)))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER,
                  metadata + 147u);
    at = metadata + 147u + 64u;
  }
  { uint32_t previous_first = 0u; int have_first = 0;
    for (i = 0; i < groups; ++i) {
      uint64_t record_bytes;
      uint32_t members, j, first;
      if (!span_fits(at, 40u, metadata + table))
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, at);
      record_bytes = u64(p->bytes + at); members = u32(p->bytes + at + 8u);
      if (members == 0u || record_bytes != 40u + UINT64_C(4) * members ||
          !span_fits(at, record_bytes, metadata + table) ||
          !zero_bytes(p->bytes + at + 12u, 4u) ||
          !zero_bytes(p->bytes + at + 36u, 4u) ||
          !options_valid(p->bytes + at + 16u))
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, at);
      first = u32(p->bytes + at + 40u);
      if (have_first && first <= previous_first)
        return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ORDER, at + 40u);
      for (j = 0; j < members; ++j) {
        uint32_t index = u32(p->bytes + at + 40u + UINT64_C(4) * j);
        if (index >= count ||
            (j != 0u && index <= u32(p->bytes + at + 36u + UINT64_C(4) * j)))
          return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ORDER,
                      at + 40u + UINT64_C(4) * j);
      }
      previous_first = first; have_first = 1;
      at += record_bytes;
    }
    if (at != metadata + table)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, at);
    { uint32_t wanted;
      for (wanted = 0u; wanted < count; ++wanted) {
        uint64_t scan = metadata + 147u + 64u;
        uint32_t occurrences = 0u, g;
        for (g = 0u; g < groups; ++g) {
          uint32_t members = u32(p->bytes + scan + 8u), j;
          for (j = 0u; j < members; ++j)
            if (u32(p->bytes + scan + 40u + UINT64_C(4) * j) == wanted)
              ++occurrences;
          scan += u64(p->bytes + scan);
        }
        if (occurrences != 1u)
          return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER,
                      metadata + 147u + 64u);
      }
    }
  }
  at = metadata + table;
  for (i = 0; i < count; ++i) {
    const uint8_t *r;
    c1_record model;
    uint32_t model_index, path_bytes;
    uint16_t segments, rank, d;
    uint64_t record_bytes, extents_bytes, expect_record, elements, moment_bytes;
    uint64_t avg, sq;
    uint8_t digest[32], zeros[64] = {0};
    sha256_state state;
    uint8_t index_kind[8];
    if (!span_fits(at, 120u, metadata + metadata_bytes) ||
        !canonical_parameter(p, c1, i, &model, &model_index))
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, at);
    r = p->bytes + at; record_bytes = u64(r); path_bytes = u32(r + 12u);
    segments = u16(r + 16u); rank = u16(r + 18u);
    if (rank > 64u || !mul_u64(rank, 8u, &extents_bytes) ||
        !add_u64(120u, extents_bytes, &expect_record) ||
        !add_u64(expect_record, path_bytes, &expect_record) ||
        record_bytes != expect_record || !span_fits(at, record_bytes, metadata + metadata_bytes) ||
        u32(r + 8u) != model_index || !zero_bytes(r + 20u, 4u) ||
        segments != model.segment_count || rank != model.rank ||
        path_bytes != model.path_bytes || !path_validate(p, at + 120u + extents_bytes,
                                                         path_bytes, segments) ||
        path_compare(p->bytes, model.path_offset, model.segment_count,
                     at + 120u + extents_bytes, segments) != 0)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OPTIMIZER, at);
    elements = u64(r + 24u); moment_bytes = u64(r + 32u);
    if (elements != model.element_count || !mul_u64(elements, 4u, &expect_record) ||
        moment_bytes != expect_record)
      return fail(p, ET_C2_FORMAT_SHAPE_MISMATCH, ET_C2_FORMAT_CODE_SHAPE, at + 24u);
    for (d = 0; d < rank; ++d)
      if (u64(r + 120u + UINT64_C(8) * d) !=
          u64(p->bytes + model.start + 80u + UINT64_C(8) * d))
        return fail(p, ET_C2_FORMAT_SHAPE_MISMATCH, ET_C2_FORMAT_CODE_SHAPE,
                    at + 120u + UINT64_C(8) * d);
    avg = u64(r + 40u); sq = u64(r + 48u);
    if (avg != expected || !add_u64(avg, moment_bytes, &expect_record) ||
        sq != expect_record || !add_u64(sq, moment_bytes, &expected) ||
        expected > payload_bytes || moment_bytes > ET_C2_WIRE_MAX_TENSOR_BYTES)
      return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, at + 40u);
    if (moment_bytes > *maximum_tensor_bytes) *maximum_tensor_bytes = moment_bytes;
    index_kind[0] = (uint8_t)i; index_kind[1] = (uint8_t)(i >> 8u);
    index_kind[2] = (uint8_t)(i >> 16u); index_kind[3] = (uint8_t)(i >> 24u);
    { uint32_t kind;
      for (kind = 1u; kind <= 2u; ++kind) {
        uint64_t relative = kind == 1u ? avg : sq, q;
        index_kind[4] = (uint8_t)kind; index_kind[5] = index_kind[6] = index_kind[7] = 0u;
        sha256_init(&state);
        sha256_update(&state, moment_domain, sizeof(moment_domain) - 1u);
        sha256_update(&state, index_kind, sizeof(index_kind));
        sha256_update(&state, r, 56u); sha256_update(&state, zeros, sizeof(zeros));
        sha256_update(&state, r + 120u, (size_t)(record_bytes - 120u));
        sha256_update(&state, p->bytes + payload + relative, (size_t)moment_bytes);
        sha256_final(&state, digest);
        if (memcmp(digest, r + (kind == 1u ? 56u : 88u), 32u) != 0)
          return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CHECKSUM,
                      at + (kind == 1u ? 56u : 88u));
        for (q = 0; q < moment_bytes; q += 4u) {
          uint32_t bits = u32(p->bytes + payload + relative + q);
          if (!bits_finite(bits) ||
              (kind == 2u && !bits_nonnegative_or_negative_zero(bits)))
            return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_MOMENT,
                        payload + relative + q);
        }
      }
    }
    at += record_bytes;
  }
  if (at != metadata + metadata_bytes || expected != payload_bytes)
    return fail(p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, at);
  return ET_C2_FORMAT_OK;
}

static int printable_library_grammar(const uint8_t *p) {
  unsigned components = 0u, nonempty = 0u;
  size_t i;
  for (i = 0; i < 57u; ++i) {
    if (p[i] == 0u) { if (!nonempty) return 0; ++components; nonempty = 0u; }
    else { if (p[i] < 0x20u || p[i] > 0x7eu) return 0; nonempty = 1u; }
  }
  return components == 3u && !nonempty;
}
static int compiler_grammar(const uint8_t *p) {
  static const uint8_t prefix[] = "Eshkol Compiler v";
  size_t i = sizeof(prefix) - 1u, version_start = i;
  if (memcmp(p, prefix, i) != 0) return 0;
  while (i < 30u && p[i] != 0u) {
    uint8_t c = p[i++];
    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
          (c >= 'a' && c <= 'z') || c == '.' || c == '+' || c == '-')) return 0;
  }
  if (i == version_start || i >= 30u || p[i++] != 0u || i + 41u != 71u) return 0;
  { size_t j; for (j = 0; j < 40u; ++j)
      if (!((p[i + j] >= '0' && p[i + j] <= '9') ||
            (p[i + j] >= 'a' && p[i + j] <= 'f'))) return 0; }
  return p[i + 40u] == 0u;
}

int32_t et_c2_checkpoint_parse_v1(
    const uint8_t *bytes, size_t byte_count,
    const et_c2_checkpoint_limits_v1 *limits, uint32_t mode,
    et_c2_checkpoint_view_v1 *view,
    et_c2_checkpoint_format_error_v1 *error) {
  parser p;
  et_c2_checkpoint_view_v1 result;
  c1_info c1;
  et_c2_x1_projection_v1 x1 = {0};
  et_c2_x1_error_v1 x1_error = {0};
  const uint8_t *h;
  uint64_t metadata_end, payload_end, unsigned_end, at, optimizer_metadata;
  uint32_t cursor_bytes, epoch_bytes, fp_bytes, groups;
  int32_t status;
  p.bytes = bytes; p.size = byte_count; p.limits = limits; p.mode = mode; p.error = error;
  if (error != NULL) {
    if (error->struct_size != ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE) return ET_C2_FORMAT_INVALID_ARGUMENT;
    error->category = error->code = 0u; error->offset = 0u;
  }
  if (bytes == NULL || limits == NULL || view == NULL ||
      limits->struct_size != ET_C2_CHECKPOINT_LIMITS_V1_SIZE ||
      view->struct_size != ET_C2_CHECKPOINT_VIEW_V1_SIZE ||
      (mode != ET_C2_FORMAT_INSPECT && mode != ET_C2_FORMAT_LOAD) ||
      limits->maximum_file_bytes == 0u ||
      limits->maximum_file_bytes > ET_C2_WIRE_MAX_FILE_BYTES ||
      limits->maximum_metadata_bytes == 0u ||
      limits->maximum_metadata_bytes > ET_C2_WIRE_MAX_METADATA_BYTES ||
      limits->maximum_tensor_bytes == 0u ||
      limits->maximum_tensor_bytes > ET_C2_WIRE_MAX_TENSOR_BYTES ||
      limits->maximum_tensors == 0u || limits->maximum_tensors > ET_C2_WIRE_MAX_TENSORS ||
      limits->enforce_operational_profile > 1u)
    return fail(&p, ET_C2_FORMAT_INVALID_ARGUMENT, ET_C2_FORMAT_CODE_ARGUMENT, 0u);
  if (byte_count < 288u)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_TRUNCATED, byte_count);
  h = bytes;
  if (memcmp(h, c2_magic, 16u) != 0)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_MAGIC, 0u);
  if (u16(h + 16u) != 1u || u16(h + 18u) != 0u || u64(h + 32u) != 0u)
    return fail(&p, ET_C2_FORMAT_VERSION_MISMATCH, ET_C2_FORMAT_CODE_VERSION, 16u);
  if (u32(h + 20u) != 256u)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_FIXED_FIELD, 20u);
  if (u32(h + 24u) != 1u || u32(h + 28u) != 1u)
    return fail(&p, ET_C2_FORMAT_VERSION_MISMATCH, ET_C2_FORMAT_CODE_VERSION, 24u);
  if (!zero_bytes(h + 240u, 16u))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_RESERVED, 240u);
  memset(&result, 0, sizeof(result)); result.struct_size = sizeof(result);
  result.bytes = bytes; result.file_bytes = u64(h + 40u);
  result.outer_metadata_offset = u64(h + 48u); result.outer_metadata_bytes = u64(h + 56u);
  result.physical_payload_offset = u64(h + 64u); result.physical_payload_bytes = u64(h + 72u);
  result.model_tensor_count = u32(h + 80u); result.unique_parameter_count = u32(h + 84u);
  result.optimizer_tensor_count = u32(h + 88u); result.total_tensor_count = u32(h + 92u);
  result.model_container_bytes = u64(h + 96u); result.optimizer_metadata_bytes = u64(h + 104u);
  result.optimizer_payload_bytes = u64(h + 112u); result.x1_bytes = u64(h + 120u);
  cursor_bytes = u32(h + 128u); epoch_bytes = u32(h + 132u); fp_bytes = u32(h + 136u);
  groups = u32(h + 156u); result.tokenizer_fingerprint_bytes = fp_bytes;
  result.optimizer_group_count = groups; result.total_tokens = u64(h + 184u);
  result.completed_updates = u64(h + 192u); result.epochs = u64(h + 200u);
  result.rng_key_bits = u64(h + 216u); result.rng_counter_low_bits = u64(h + 224u);
  result.rng_counter_high_bits = u64(h + 232u);
  if (result.file_bytes != byte_count || result.file_bytes > ET_C2_WIRE_MAX_FILE_BYTES ||
      result.outer_metadata_offset != 256u ||
      !add_u64(256u, result.outer_metadata_bytes, &metadata_end) ||
      result.physical_payload_offset != metadata_end ||
      !add_u64(result.physical_payload_offset, result.physical_payload_bytes, &payload_end) ||
      !add_u64(payload_end, 32u, &unsigned_end) || unsigned_end != result.file_bytes ||
      result.outer_metadata_bytes > ET_C2_WIRE_MAX_METADATA_BYTES ||
      result.model_tensor_count > 4096u || result.unique_parameter_count == 0u ||
      result.unique_parameter_count > 1365u ||
      result.optimizer_tensor_count != 2u * result.unique_parameter_count ||
      result.total_tensor_count != result.model_tensor_count + result.optimizer_tensor_count ||
      result.total_tensor_count > 6826u || groups == 0u || groups > 1365u ||
      cursor_bytes != epoch_bytes || (cursor_bytes != 303u && cursor_bytes != 304u) ||
      ((fp_bytes == 95u) ? cursor_bytes != 303u :
       (fp_bytes == 96u ? cursor_bytes != 304u : 1)) ||
      u32(h + 140u) != 93u || u32(h + 144u) != 19u ||
      u32(h + 148u) != 57u || u32(h + 152u) != 71u ||
      result.x1_bytes > ET_C2_X1_MAX_CANONICAL_BYTES ||
      result.total_tokens > I64_MAX_U ||
      result.completed_updates > I64_MAX_U || result.epochs > I64_MAX_U ||
      u64(h + 208u) != 1u)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, 40u);
  { static const uint16_t versions[12] = {1,0,2,0,1,0,1,0,1,0,1,0};
    uint32_t i; for (i = 0; i < 12u; ++i)
      if (u16(h + 160u + 2u * i) != versions[i])
        return fail(&p, ET_C2_FORMAT_VERSION_MISMATCH, ET_C2_FORMAT_CODE_VERSION,
                    160u + 2u * i); }
  if (result.file_bytes > limits->maximum_file_bytes ||
      result.outer_metadata_bytes > limits->maximum_metadata_bytes ||
      result.total_tensor_count > limits->maximum_tensors)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_LIMIT, 40u);
  if (limits->enforce_operational_profile &&
      (result.file_bytes > ET_C2_PROFILE_MAX_FILE_BYTES ||
       result.outer_metadata_bytes > ET_C2_PROFILE_MAX_METADATA_BYTES ||
       result.total_tensor_count > ET_C2_PROFILE_MAX_TENSORS))
    return fail(&p, ET_C2_FORMAT_UNSUPPORTED, ET_C2_FORMAT_CODE_PROFILE, 40u);
  if (!digest_matches(c2_domain, sizeof(c2_domain) - 1u, bytes,
                      byte_count - 32u, bytes + byte_count - 32u))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CHECKSUM,
                byte_count - 32u);
  at = 256u;
  { uint64_t required = (uint64_t)fp_bytes + UINT64_C(93 + 19 + 57 + 71);
    if (!add_u64(required, result.x1_bytes, &required) ||
        !add_u64(required, cursor_bytes, &required) ||
        !add_u64(required, epoch_bytes, &required) ||
        !add_u64(required, result.optimizer_metadata_bytes, &required) ||
        required != result.outer_metadata_bytes || result.x1_bytes == 0u)
      return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, 56u);
  }
  if (!fingerprint_valid(bytes + at, fp_bytes, 1))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY, at);
  at += fp_bytes;
  if (!fingerprint_valid(bytes + at, 93u, 0))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY, at);
  at += 93u;
  if (memcmp(bytes + at, provider_id, 19u) != 0) {
    size_t i; int grammar = 1;
    for (i = 0; i < 19u; ++i)
      if (!((bytes[at + i] >= 'a' && bytes[at + i] <= 'z') ||
            (bytes[at + i] >= '0' && bytes[at + i] <= '9') || bytes[at + i] == '-')) grammar = 0;
    return fail(&p, grammar ? ET_C2_FORMAT_UNSUPPORTED : ET_C2_FORMAT_CORRUPT_DATA,
                ET_C2_FORMAT_CODE_IDENTITY, at);
  }
  at += 19u;
  if (memcmp(bytes + at, library_id, 57u) != 0)
    return fail(&p, printable_library_grammar(bytes + at) ? ET_C2_FORMAT_VERSION_MISMATCH :
                ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY, at);
  at += 57u;
  if (memcmp(bytes + at, compiler_id, 71u) != 0)
    return fail(&p, compiler_grammar(bytes + at) ?
                (mode == ET_C2_FORMAT_LOAD ? ET_C2_FORMAT_DETERMINISM_UNAVAILABLE :
                                             ET_C2_FORMAT_UNSUPPORTED) :
                ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_IDENTITY, at);
  at += 71u; result.x1_offset = at;
  if (!span_fits(at, result.x1_bytes, metadata_end))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_UTF8, at);
  x1.struct_size = sizeof(x1);
  x1_error.struct_size = sizeof(x1_error);
  status = et_c2_private_x1_canonical_inspect_v1(
      bytes + at, (size_t)result.x1_bytes, bytes + 256u + fp_bytes, 93u,
      &x1, &x1_error);
  if (status != ET_C2_X1_OK)
    return fail_x1(&p, &x1_error, at);
  at += result.x1_bytes; result.current_cursor_offset = at;
  if (!span_fits(at, cursor_bytes, metadata_end))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CURSOR, at);
  status = cursor_validate(&p, at, cursor_bytes, bytes + 256u, fp_bytes);
  if (status != ET_C2_FORMAT_OK)
    return fail(&p, (uint32_t)status, ET_C2_FORMAT_CODE_CURSOR, at);
  at += cursor_bytes; result.epoch_cursor_offset = at;
  if (!span_fits(at, epoch_bytes, metadata_end))
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CURSOR, at);
  status = cursor_validate(&p, at, epoch_bytes, bytes + 256u, fp_bytes);
  if (status != ET_C2_FORMAT_OK)
    return fail(&p, (uint32_t)status, ET_C2_FORMAT_CODE_CURSOR, at);
  { const uint8_t *current = bytes + result.current_cursor_offset;
    const uint8_t *epoch = bytes + result.epoch_cursor_offset;
    uint64_t rows = u64(current + 112u), current_at = u64(current + 120u),
             epoch_at = u64(epoch + 120u);
    if (memcmp(current, epoch, 120u) != 0 ||
        memcmp(current + 128u, epoch + 128u, 48u + fp_bytes) != 0 ||
        rows == 0u || epoch_at >= rows || epoch_at > current_at || current_at > rows ||
        u64(current + 40u) != x1.vocabulary_size ||
        result.rng_key_bits != x1.seed)
      return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_CURSOR,
                  result.epoch_cursor_offset);
  }
  at += epoch_bytes; optimizer_metadata = at; result.optimizer_metadata_offset = at;
  if (!add_u64(at, result.optimizer_metadata_bytes, &at) || at != metadata_end)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, optimizer_metadata);
  result.model_container_offset = result.physical_payload_offset;
  if (!add_u64(result.model_container_bytes, result.optimizer_payload_bytes, &at) ||
      at != result.physical_payload_bytes)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_SPAN, 96u);
  result.optimizer_payload_offset = result.model_container_offset + result.model_container_bytes;
  status = validate_c1(&p, result.model_container_offset, result.model_container_bytes,
                       result.outer_metadata_bytes, &c1);
  if (status != ET_C2_FORMAT_OK) return status;
  if (c1.entries != result.model_tensor_count ||
      c1.unique_parameters != result.unique_parameter_count)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_ALIAS,
                result.model_container_offset + 80u);
  status = validate_optimizer(&p, optimizer_metadata, result.optimizer_metadata_bytes,
                              result.optimizer_payload_offset,
                              result.optimizer_payload_bytes, &c1, groups,
                              result.completed_updates, &c1.maximum_tensor_bytes);
  if (status != ET_C2_FORMAT_OK) return status;
  if (c1.maximum_tensor_bytes > limits->maximum_tensor_bytes)
    return fail(&p, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_LIMIT, 112u);
  if (limits->enforce_operational_profile &&
      c1.maximum_tensor_bytes > ET_C2_PROFILE_MAX_TENSOR_BYTES)
    return fail(&p, ET_C2_FORMAT_UNSUPPORTED, ET_C2_FORMAT_CODE_PROFILE, 112u);
  *view = result;
  return ET_C2_FORMAT_OK;
}
