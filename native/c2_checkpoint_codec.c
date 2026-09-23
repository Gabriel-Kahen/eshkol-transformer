#include "c2_checkpoint_codec.h"

#include <limits.h>
#include <string.h>

#define C2_HEADER_BYTES ((size_t)256u)
#define C1_HEADER_BYTES ((size_t)128u)
#define C1_RECORD_BYTES ((size_t)80u)
#define O2_HEADER_BYTES ((size_t)128u)
#define O2_RECORD_BYTES ((size_t)120u)
#define SHA_BYTES ((size_t)32u)
#define I64_MAX_U UINT64_C(9223372036854775807)

static const uint8_t c2_magic[16] = {0x89, 0x45, 0x53, 0x48, 0x4b, 0x54,
                                     0x52, 0x4e, 0x53, 0x54, 0x41, 0x54,
                                     0x45, 0x0d, 0x0a, 0x00};
static const uint8_t c1_magic[16] = {0x89, 0x45, 0x53, 0x48, 0x4b, 0x4f,
                                     0x4c, 0x43, 0x4b, 0x50, 0x54, 0x0d,
                                     0x0a, 0x1a, 0x0a, 0x00};
static const uint8_t optimizer_magic[8] = {'E', 'S', 'H', 'K',
                                           'O', 'P', 'T', '1'};
static const uint8_t provider[] = "i2-dense-cpu-f32-v1";
static const uint8_t library_id[] = "eshkol-transformer\0"
                                    "0.1.0-draft\0"
                                    "eshkol-training-state:1.0\0";
static const uint8_t compiler_id[] =
    "Eshkol Compiler v1.3.4-evolve\0"
    "222cad3aac68ddf48d09c1cdf322fa4c4e7b8296\0";
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

typedef struct encode_plan {
  size_t outer_metadata, model_metadata, payload, file;
  uint32_t model_count, unique_count, group_count;
  uint64_t completed_updates;
} encode_plan;

static uint16_t get16(const uint8_t *p) {
  return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8u));
}
static uint32_t get32(const uint8_t *p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8u) | ((uint32_t)p[2] << 16u) |
         ((uint32_t)p[3] << 24u);
}
static uint64_t get64(const uint8_t *p) {
  uint64_t v = 0u;
  unsigned i;
  for (i = 0u; i < 8u; ++i)
    v |= (uint64_t)p[i] << (8u * i);
  return v;
}
static void put16(uint8_t *p, uint16_t v) {
  p[0] = (uint8_t)v;
  p[1] = (uint8_t)(v >> 8u);
}
static void put32(uint8_t *p, uint32_t v) {
  unsigned i;
  for (i = 0u; i < 4u; ++i)
    p[i] = (uint8_t)(v >> (8u * i));
}
static void put64(uint8_t *p, uint64_t v) {
  unsigned i;
  for (i = 0u; i < 8u; ++i)
    p[i] = (uint8_t)(v >> (8u * i));
}
static int add_size(size_t a, size_t b, size_t *out) {
  if (a > SIZE_MAX - b)
    return 0;
  *out = a + b;
  return 1;
}
static int size_from_u64(uint64_t value, size_t *out) {
  if (value > SIZE_MAX)
    return 0;
  *out = (size_t)value;
  return 1;
}
static int span(size_t at, size_t n, size_t end) {
  return at <= end && n <= end - at;
}
static int overlaps(const uint8_t *a, size_t an, const uint8_t *b, size_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  if (an == 0u || bn == 0u)
    return 0;
  if (an > UINTPTR_MAX - x || bn > UINTPTR_MAX - y)
    return 1;
  return x < y + bn && y < x + an;
}
static int request_overlap(const et_c2_checkpoint_encode_request_v1 *r,
                           const uint8_t *p, size_t n) {
  if (r == NULL)
    return 0;
  if (overlaps(p, n, (const uint8_t *)r, sizeof(*r)))
    return 1;
  if (r->struct_size != ET_C2_CHECKPOINT_ENCODE_REQUEST_V1_SIZE)
    return 0;
  return overlaps(p, n, r->tokenizer_fingerprint,
                  r->tokenizer_fingerprint_bytes) ||
         overlaps(p, n, r->x1_canonical, r->x1_canonical_bytes) ||
         overlaps(p, n, r->current_cursor, r->current_cursor_bytes) ||
         overlaps(p, n, r->epoch_cursor, r->epoch_cursor_bytes) ||
         overlaps(p, n, r->model_c1, r->model_c1_bytes) ||
         overlaps(p, n, r->optimizer_metadata, r->optimizer_metadata_bytes) ||
         overlaps(p, n, r->optimizer_payload, r->optimizer_payload_bytes);
}
static int zeros(const uint8_t *p, size_t n) {
  size_t i;
  for (i = 0u; i < n; ++i)
    if (p[i] != 0u)
      return 0;
  return 1;
}
static int32_t fail(et_c2_checkpoint_format_error_v1 *error, uint32_t category,
                    uint32_t code, size_t offset) {
  if (error != NULL) {
    error->category = category;
    error->code = code;
    error->offset = (uint64_t)offset;
  }
  return (int32_t)category;
}

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}
static void sha_block(sha256_state *s, const uint8_t block[64]) {
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
  uint32_t w[64], a, b, c, d, e, f, g, h;
  size_t i;
  for (i = 0u; i < 16u; ++i) {
    size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | block[j + 3u];
  }
  for (; i < 64u; ++i) {
    uint32_t x =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    uint32_t y =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + x + w[i - 7u] + y;
  }
  a = s->h[0];
  b = s->h[1];
  c = s->h[2];
  d = s->h[3];
  e = s->h[4];
  f = s->h[5];
  g = s->h[6];
  h = s->h[7];
  for (i = 0u; i < 64u; ++i) {
    uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u),
             ch = (e & f) ^ ((~e) & g), t1 = h + s1 + ch + k[i] + w[i],
             s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u),
             maj = (a & b) ^ (a & c) ^ (b & c), t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  s->h[0] += a;
  s->h[1] += b;
  s->h[2] += c;
  s->h[3] += d;
  s->h[4] += e;
  s->h[5] += f;
  s->h[6] += g;
  s->h[7] += h;
}
static void sha_init(sha256_state *s) {
  static const uint32_t h[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                0x1f83d9abu, 0x5be0cd19u};
  memcpy(s->h, h, sizeof(h));
  s->length = 0u;
  s->used = 0u;
}
static void sha_update(sha256_state *s, const uint8_t *p, size_t n) {
  s->length += (uint64_t)n;
  while (n) {
    size_t take = 64u - s->used;
    if (take > n)
      take = n;
    memcpy(s->block + s->used, p, take);
    s->used += take;
    p += take;
    n -= take;
    if (s->used == 64u) {
      sha_block(s, s->block);
      s->used = 0u;
    }
  }
}
static void sha_final(sha256_state *s, uint8_t out[32]) {
  uint64_t bits = s->length * 8u;
  size_t i;
  s->block[s->used++] = 0x80u;
  if (s->used > 56u) {
    memset(s->block + s->used, 0, 64u - s->used);
    sha_block(s, s->block);
    s->used = 0u;
  }
  memset(s->block + s->used, 0, 56u - s->used);
  for (i = 0u; i < 8u; ++i)
    s->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha_block(s, s->block);
  for (i = 0u; i < 8u; ++i) {
    out[4u * i] = (uint8_t)(s->h[i] >> 24u);
    out[4u * i + 1u] = (uint8_t)(s->h[i] >> 16u);
    out[4u * i + 2u] = (uint8_t)(s->h[i] >> 8u);
    out[4u * i + 3u] = (uint8_t)s->h[i];
  }
}
static void digest2(const uint8_t *domain, size_t dn, const uint8_t *p,
                    size_t n, uint8_t out[32]) {
  sha256_state s;
  sha_init(&s);
  sha_update(&s, domain, dn);
  sha_update(&s, p, n);
  sha_final(&s, out);
}

static int fingerprint(const uint8_t *p, size_t n) {
  static const char a[] = "sha256:eshkol-byte-tokenizer-v1:";
  static const char b[] = "sha256:eshkol-bpe-tokenizer-v1:";
  size_t i, pre;
  if (n == 96u && memcmp(p, a, sizeof(a) - 1u) == 0)
    pre = sizeof(a) - 1u;
  else if (n == 95u && memcmp(p, b, sizeof(b) - 1u) == 0)
    pre = sizeof(b) - 1u;
  else
    return 0;
  for (i = pre; i < n; ++i)
    if (!((p[i] >= '0' && p[i] <= '9') || (p[i] >= 'a' && p[i] <= 'f')))
      return 0;
  return 1;
}

static int c1_preflight(const uint8_t *p, size_t n, uint32_t *count,
                        size_t *metadata_out) {
  size_t metadata, payload_at, payload_n, at, i;
  uint8_t digest[32];
  if (n < C1_HEADER_BYTES + SHA_BYTES || memcmp(p, c1_magic, 16u) != 0 ||
      get16(p + 16u) != 1u || get16(p + 18u) != 0u || get32(p + 20u) != 128u ||
      get32(p + 24u) != 1u || get32(p + 28u) != 1u || get64(p + 32u) != 0u ||
      get64(p + 40u) != n || get64(p + 48u) != 128u ||
      get64(p + 56u) > SIZE_MAX || get64(p + 64u) > SIZE_MAX ||
      get64(p + 72u) > SIZE_MAX || get32(p + 88u) != 19u ||
      get16(p + 92u) != 1u || get16(p + 94u) != 0u || get16(p + 96u) != 2u ||
      get16(p + 98u) != 0u || !zeros(p + 100u, 28u))
    return 0;
  metadata = (size_t)get64(p + 56u);
  payload_at = (size_t)get64(p + 64u);
  payload_n = (size_t)get64(p + 72u);
  *count = get32(p + 80u);
  if (metadata > ET_C2_WIRE_MAX_METADATA_BYTES || *count > 4096u ||
      payload_at != 128u + metadata ||
      !span(payload_at, payload_n, n - SHA_BYTES) ||
      payload_at + payload_n != n - SHA_BYTES || metadata < 19u ||
      memcmp(p + 128u, provider, 19u) != 0)
    return 0;
  digest2(c1_domain, sizeof(c1_domain) - 1u, p, n - SHA_BYTES, digest);
  if (memcmp(digest, p + n - SHA_BYTES, SHA_BYTES) != 0)
    return 0;
  at = 147u;
  for (i = 0u; i < *count; ++i) {
    size_t rec, rank, path, relative, bytes, shape, end;
    sha256_state s;
    uint8_t got[32], z[32] = {0};
    if (!span(at, C1_RECORD_BYTES, payload_at) ||
        !size_from_u64(get64(p + at), &rec))
      return 0;
    rank = get16(p + at + 30u);
    path = get32(p + at + 24u);
    if (rank > 64u || !add_size(80u, 8u * rank, &shape) ||
        !add_size(shape, path, &end) || rec != end ||
        !span(at, rec, payload_at) ||
        get64(p + at + 16u) > ET_C2_WIRE_MAX_TENSOR_BYTES ||
        !size_from_u64(get64(p + at + 8u), &relative) ||
        !size_from_u64(get64(p + at + 16u), &bytes) ||
        !span(relative, bytes, payload_n))
      return 0;
    sha_init(&s);
    sha_update(&s, c1_tensor_domain, sizeof(c1_tensor_domain) - 1u);
    sha_update(&s, p + at, 48u);
    sha_update(&s, z, 32u);
    sha_update(&s, p + at + 80u, rec - 80u);
    sha_update(&s, p + payload_at + relative, bytes);
    sha_final(&s, got);
    if (memcmp(got, p + at + 48u, 32u) != 0)
      return 0;
    at += rec;
  }
  for (i = 0u; i < get32(p + 84u); ++i) {
    size_t members, member_bytes;
    if (!span(at, 8u, payload_at))
      return 0;
    members = get32(p + at);
    if (members < 2u || members > SIZE_MAX / 4u)
      return 0;
    member_bytes = 4u * members;
    if (!span(at + 8u, member_bytes, payload_at))
      return 0;
    at += 8u + member_bytes;
  }
  *metadata_out = metadata;
  return at == payload_at;
}

static int o2_preflight(const uint8_t *p, size_t n, const uint8_t *payload,
                        size_t payload_n, uint32_t *unique, uint32_t *groups,
                        uint64_t *completed) {
  size_t config, table, at, i;
  if (n < 211u || memcmp(p, optimizer_magic, 8u) != 0 || get16(p + 8u) != 1u ||
      get16(p + 10u) != 0u || get32(p + 12u) != 128u || get64(p + 16u) != 0u ||
      get32(p + 24u) != 1u || get32(p + 28u) != 3u || get32(p + 32u) != 1u ||
      get32(p + 36u) != 1u || get16(p + 40u) != 2u || get16(p + 42u) != 0u ||
      get32(p + 44u) != 19u || memcmp(p + 128u, provider, 19u) != 0 ||
      get64(p + 88u) != payload_n || get64(p + 96u) != 147u ||
      get64(p + 112u) != n || !zeros(p + 120u, 8u))
    return 0;
  *groups = get32(p + 48u);
  *unique = get32(p + 52u);
  *completed = get64(p + 64u);
  if (*groups == 0u || *groups > 1365u || *unique == 0u || *unique > 1365u ||
      get32(p + 56u) != 2u * *unique || *completed > I64_MAX_U)
    return 0;
  if (get64(p + 72u) > SIZE_MAX || get64(p + 104u) > SIZE_MAX)
    return 0;
  config = (size_t)get64(p + 72u);
  table = (size_t)get64(p + 104u);
  {
    size_t expected_table;
    if (!add_size(147u, config, &expected_table) || table != expected_table ||
        table > n || config < 64u)
      return 0;
  }
  at = 211u;
  for (i = 0u; i < *groups; ++i) {
    size_t rec, members, member_bytes, expected;
    if (!span(at, 40u, table) || !size_from_u64(get64(p + at), &rec))
      return 0;
    members = get32(p + at + 8u);
    if (members == 0u || members > SIZE_MAX / 4u)
      return 0;
    member_bytes = 4u * members;
    if (!add_size(40u, member_bytes, &expected) || rec != expected ||
        !span(at, rec, table))
      return 0;
    at += rec;
  }
  if (at != table)
    return 0;
  for (i = 0u; i < *unique; ++i) {
    size_t rec, rank, path, bytes, a, b, shape, expected;
    if (!span(at, O2_RECORD_BYTES, n) || !size_from_u64(get64(p + at), &rec))
      return 0;
    rank = get16(p + at + 18u);
    path = get32(p + at + 12u);
    if (rank > 64u || !add_size(120u, 8u * rank, &shape) ||
        !add_size(shape, path, &expected) || rec != expected ||
        !span(at, rec, n) ||
        get64(p + at + 32u) > ET_C2_WIRE_MAX_TENSOR_BYTES ||
        !size_from_u64(get64(p + at + 32u), &bytes) ||
        !size_from_u64(get64(p + at + 40u), &a) ||
        !size_from_u64(get64(p + at + 48u), &b) || !span(a, bytes, payload_n) ||
        !span(b, bytes, payload_n))
      return 0;
    at += rec;
  }
  return at == n && (payload_n == 0u || payload != NULL);
}

static int32_t plan_request(const et_c2_checkpoint_encode_request_v1 *r,
                            encode_plan *plan,
                            et_c2_checkpoint_format_error_v1 *error) {
  size_t fixed = 93u + 19u + 57u + 71u, physical;
  if (error != NULL &&
      error->struct_size != ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE)
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (r != NULL && error != NULL &&
      overlaps((const uint8_t *)error, sizeof(*error), (const uint8_t *)r,
               sizeof(*r)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (r == NULL || plan == NULL ||
      r->struct_size != ET_C2_CHECKPOINT_ENCODE_REQUEST_V1_SIZE)
    return fail(error, ET_C2_FORMAT_INVALID_ARGUMENT,
                ET_C2_FORMAT_CODE_ARGUMENT, 0u);
  if (error != NULL &&
      request_overlap(r, (const uint8_t *)error, sizeof(*error)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (error != NULL) {
    error->category = error->code = 0u;
    error->offset = 0u;
  }
  if (r->tokenizer_fingerprint == NULL || r->x1_canonical == NULL ||
      r->current_cursor == NULL || r->epoch_cursor == NULL ||
      r->model_c1 == NULL || r->optimizer_metadata == NULL ||
      (r->optimizer_payload_bytes != 0u && r->optimizer_payload == NULL) ||
      r->total_tokens > I64_MAX_U || r->epochs > I64_MAX_U)
    return fail(error, ET_C2_FORMAT_INVALID_ARGUMENT,
                ET_C2_FORMAT_CODE_ARGUMENT, 0u);
  if (!fingerprint(r->tokenizer_fingerprint, r->tokenizer_fingerprint_bytes) ||
      r->x1_canonical_bytes == 0u || r->x1_canonical_bytes > 16384u ||
      r->current_cursor_bytes != 208u + r->tokenizer_fingerprint_bytes ||
      r->epoch_cursor_bytes != r->current_cursor_bytes ||
      r->model_c1_bytes > ET_C2_WIRE_MAX_FILE_BYTES ||
      r->optimizer_metadata_bytes > ET_C2_WIRE_MAX_METADATA_BYTES ||
      r->optimizer_payload_bytes > ET_C2_WIRE_MAX_FILE_BYTES)
    return fail(error, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_FIXED_FIELD,
                0u);
  if (!c1_preflight(r->model_c1, r->model_c1_bytes, &plan->model_count,
                    &plan->model_metadata) ||
      !o2_preflight(r->optimizer_metadata, r->optimizer_metadata_bytes,
                    r->optimizer_payload, r->optimizer_payload_bytes,
                    &plan->unique_count, &plan->group_count,
                    &plan->completed_updates))
    return fail(error, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_FIXED_FIELD,
                0u);
  if (plan->model_count + 2u * plan->unique_count > ET_C2_WIRE_MAX_TENSORS)
    return fail(error, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_LIMIT, 0u);
  if (!add_size(fixed, r->tokenizer_fingerprint_bytes, &plan->outer_metadata) ||
      !add_size(plan->outer_metadata, r->x1_canonical_bytes,
                &plan->outer_metadata) ||
      !add_size(plan->outer_metadata, r->current_cursor_bytes,
                &plan->outer_metadata) ||
      !add_size(plan->outer_metadata, r->epoch_cursor_bytes,
                &plan->outer_metadata) ||
      !add_size(plan->outer_metadata, r->optimizer_metadata_bytes,
                &plan->outer_metadata) ||
      !add_size(r->model_c1_bytes, r->optimizer_payload_bytes, &physical) ||
      !add_size(C2_HEADER_BYTES, plan->outer_metadata, &plan->payload) ||
      !add_size(plan->payload, physical, &plan->file) ||
      !add_size(plan->file, SHA_BYTES, &plan->file) ||
      plan->file > ET_C2_WIRE_MAX_FILE_BYTES ||
      plan->outer_metadata > ET_C2_WIRE_MAX_METADATA_BYTES ||
      plan->model_metadata >
          ET_C2_WIRE_MAX_METADATA_BYTES - plan->outer_metadata)
    return fail(error, ET_C2_FORMAT_CORRUPT_DATA, ET_C2_FORMAT_CODE_OVERFLOW,
                0u);
  return ET_C2_FORMAT_OK;
}

int32_t
et_c2_checkpoint_encode_measure_v1(const et_c2_checkpoint_encode_request_v1 *r,
                                   size_t *file_bytes,
                                   et_c2_checkpoint_format_error_v1 *error) {
  encode_plan p;
  int32_t s;
  if (error != NULL &&
      error->struct_size != ET_C2_CHECKPOINT_FORMAT_ERROR_V1_SIZE)
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (error != NULL &&
      request_overlap(r, (const uint8_t *)error, sizeof(*error)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (file_bytes == NULL)
    return fail(error, ET_C2_FORMAT_INVALID_ARGUMENT,
                ET_C2_FORMAT_CODE_ARGUMENT, 0u);
  if ((error != NULL &&
       overlaps((const uint8_t *)file_bytes, sizeof(*file_bytes),
                (const uint8_t *)error, sizeof(*error))) ||
      request_overlap(r, (const uint8_t *)file_bytes, sizeof(*file_bytes)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  s = plan_request(r, &p, error);
  if (s == ET_C2_FORMAT_OK)
    *file_bytes = p.file;
  return s;
}

int32_t et_c2_checkpoint_encode_v1(const et_c2_checkpoint_encode_request_v1 *r,
                                   uint8_t *d, size_t dn,
                                   et_c2_checkpoint_format_error_v1 *error) {
  encode_plan p;
  int32_t status;
  size_t at, model_at, o2_at, current_at, epoch_at, i;
  uint8_t digest[32];
  sha256_state sh;
  if (d != NULL && r != NULL && overlaps(d, dn, (const uint8_t *)r, sizeof(*r)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  if (d != NULL && error != NULL &&
      overlaps(d, dn, (const uint8_t *)error, sizeof(*error)))
    return ET_C2_FORMAT_INVALID_ARGUMENT;
  status = plan_request(r, &p, error);
  if (status != ET_C2_FORMAT_OK)
    return status;
  if (d == NULL || dn != p.file ||
      overlaps(d, dn, r->tokenizer_fingerprint,
               r->tokenizer_fingerprint_bytes) ||
      overlaps(d, dn, r->x1_canonical, r->x1_canonical_bytes) ||
      overlaps(d, dn, r->current_cursor, r->current_cursor_bytes) ||
      overlaps(d, dn, r->epoch_cursor, r->epoch_cursor_bytes) ||
      overlaps(d, dn, r->model_c1, r->model_c1_bytes) ||
      overlaps(d, dn, r->optimizer_metadata, r->optimizer_metadata_bytes) ||
      overlaps(d, dn, r->optimizer_payload, r->optimizer_payload_bytes))
    return fail(error, ET_C2_FORMAT_INVALID_ARGUMENT,
                ET_C2_FORMAT_CODE_ARGUMENT, 0u);
  memset(d, 0, C2_HEADER_BYTES);
  memcpy(d, c2_magic, 16u);
  put16(d + 16u, 1u);
  put32(d + 20u, 256u);
  put32(d + 24u, 1u);
  put32(d + 28u, 1u);
  put64(d + 40u, p.file);
  put64(d + 48u, 256u);
  put64(d + 56u, p.outer_metadata);
  put64(d + 64u, p.payload);
  put64(d + 72u, r->model_c1_bytes + r->optimizer_payload_bytes);
  put32(d + 80u, p.model_count);
  put32(d + 84u, p.unique_count);
  put32(d + 88u, 2u * p.unique_count);
  put32(d + 92u, p.model_count + 2u * p.unique_count);
  put64(d + 96u, r->model_c1_bytes);
  put64(d + 104u, r->optimizer_metadata_bytes);
  put64(d + 112u, r->optimizer_payload_bytes);
  put64(d + 120u, r->x1_canonical_bytes);
  put32(d + 128u, (uint32_t)r->current_cursor_bytes);
  put32(d + 132u, (uint32_t)r->epoch_cursor_bytes);
  put32(d + 136u, (uint32_t)r->tokenizer_fingerprint_bytes);
  put32(d + 140u, 93u);
  put32(d + 144u, 19u);
  put32(d + 148u, 57u);
  put32(d + 152u, 71u);
  put32(d + 156u, p.group_count);
  {
    static const uint16_t v[12] = {1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    for (i = 0u; i < 12u; ++i)
      put16(d + 160u + 2u * i, v[i]);
  }
  put64(d + 184u, r->total_tokens);
  put64(d + 192u, p.completed_updates);
  put64(d + 200u, r->epochs);
  put64(d + 208u, 1u);
  put64(d + 216u, r->rng_key_bits);
  put64(d + 224u, r->rng_counter_low_bits);
  put64(d + 232u, r->rng_counter_high_bits);
  at = 256u;
  memcpy(d + at, r->tokenizer_fingerprint, r->tokenizer_fingerprint_bytes);
  at += r->tokenizer_fingerprint_bytes;
  memcpy(d + at, "sha256:eshkol-config-json-v1:", 29u);
  digest2((const uint8_t *)"", 0u, r->x1_canonical, r->x1_canonical_bytes,
          digest);
  for (i = 0u; i < 32u; ++i) {
    static const char h[] = "0123456789abcdef";
    d[at + 29u + 2u * i] = (uint8_t)h[digest[i] >> 4u];
    d[at + 30u + 2u * i] = (uint8_t)h[digest[i] & 15u];
  }
  at += 93u;
  memcpy(d + at, provider, 19u);
  at += 19u;
  memcpy(d + at, library_id, 57u);
  at += 57u;
  memcpy(d + at, compiler_id, 71u);
  at += 71u;
  memcpy(d + at, r->x1_canonical, r->x1_canonical_bytes);
  at += r->x1_canonical_bytes;
  current_at = at;
  memcpy(d + at, r->current_cursor, r->current_cursor_bytes);
  at += r->current_cursor_bytes;
  epoch_at = at;
  memcpy(d + at, r->epoch_cursor, r->epoch_cursor_bytes);
  at += r->epoch_cursor_bytes;
  o2_at = at;
  memcpy(d + at, r->optimizer_metadata, r->optimizer_metadata_bytes);
  model_at = p.payload;
  memcpy(d + model_at, r->model_c1, r->model_c1_bytes);
  if (r->optimizer_payload_bytes != 0u)
    memcpy(d + model_at + r->model_c1_bytes, r->optimizer_payload,
           r->optimizer_payload_bytes);
  digest2(cursor_domain, sizeof(cursor_domain) - 1u, d + current_at,
          r->current_cursor_bytes - SHA_BYTES, digest);
  memcpy(d + current_at + r->current_cursor_bytes - SHA_BYTES, digest,
         SHA_BYTES);
  digest2(cursor_domain, sizeof(cursor_domain) - 1u, d + epoch_at,
          r->epoch_cursor_bytes - SHA_BYTES, digest);
  memcpy(d + epoch_at + r->epoch_cursor_bytes - SHA_BYTES, digest, SHA_BYTES);
  {
    size_t record_at = model_at + 147u;
    size_t model_payload = model_at + (size_t)get64(d + model_at + 64u);
    for (i = 0u; i < p.model_count; ++i) {
      size_t rec = (size_t)get64(d + record_at),
             relative = (size_t)get64(d + record_at + 8u),
             bytes = (size_t)get64(d + record_at + 16u);
      uint8_t z[32] = {0};
      sha_init(&sh);
      sha_update(&sh, c1_tensor_domain, sizeof(c1_tensor_domain) - 1u);
      sha_update(&sh, d + record_at, 48u);
      sha_update(&sh, z, 32u);
      sha_update(&sh, d + record_at + 80u, rec - 80u);
      sha_update(&sh, d + model_payload + relative, bytes);
      sha_final(&sh, d + record_at + 48u);
      record_at += rec;
    }
    digest2(c1_domain, sizeof(c1_domain) - 1u, d + model_at,
            r->model_c1_bytes - SHA_BYTES, digest);
    memcpy(d + model_at + r->model_c1_bytes - SHA_BYTES, digest, SHA_BYTES);
  }
  at = o2_at + (size_t)get64(d + o2_at + 104u);
  for (i = 0u; i < p.unique_count; ++i) {
    size_t rec = (size_t)get64(d + at), bytes = (size_t)get64(d + at + 32u),
           kind;
    for (kind = 1u; kind <= 2u; ++kind) {
      size_t relative = (size_t)get64(d + at + (kind == 1u ? 40u : 48u));
      uint8_t index_kind[8], z[64] = {0};
      put32(index_kind, (uint32_t)i);
      put32(index_kind + 4u, (uint32_t)kind);
      sha_init(&sh);
      sha_update(&sh, moment_domain, sizeof(moment_domain) - 1u);
      sha_update(&sh, index_kind, 8u);
      sha_update(&sh, d + at, 56u);
      sha_update(&sh, z, 64u);
      sha_update(&sh, d + at + 120u, rec - 120u);
      sha_update(&sh, d + model_at + r->model_c1_bytes + relative, bytes);
      sha_final(&sh, d + at + (kind == 1u ? 56u : 88u));
    }
    at += rec;
  }
  digest2(c2_domain, sizeof(c2_domain) - 1u, d, p.file - SHA_BYTES, digest);
  memcpy(d + p.file - SHA_BYTES, digest, SHA_BYTES);
  return ET_C2_FORMAT_OK;
}
