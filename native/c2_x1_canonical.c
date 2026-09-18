#include "c2_x1_canonical.h"

#include <limits.h>
#include <string.h>

typedef struct sha256_state {
  uint32_t h[8];
  uint64_t bits;
  uint8_t block[64];
  size_t used;
} sha256_state;
static uint32_t rotr(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}
static uint32_t be32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24u) | ((uint32_t)p[1] << 16u) |
         ((uint32_t)p[2] << 8u) | p[3];
}
static void put_be32(uint8_t *p, uint32_t x) {
  p[0] = (uint8_t)(x >> 24u);
  p[1] = (uint8_t)(x >> 16u);
  p[2] = (uint8_t)(x >> 8u);
  p[3] = (uint8_t)x;
}
static void sha_block(sha256_state *s, const uint8_t *p) {
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
  for (i = 0; i < 16; i++)
    w[i] = be32(p + 4u * i);
  for (i = 16; i < 64; i++) {
    uint32_t x = w[i - 15], y = w[i - 2];
    w[i] = w[i - 16] + (rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3)) + w[i - 7] +
           (rotr(y, 17) ^ rotr(y, 19) ^ (y >> 10));
  }
  a = s->h[0];
  b = s->h[1];
  c = s->h[2];
  d = s->h[3];
  e = s->h[4];
  f = s->h[5];
  g = s->h[6];
  h = s->h[7];
  for (i = 0; i < 64; i++) {
    uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) +
                  ((e & f) ^ ((~e) & g)) + k[i] + w[i];
    uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) +
                  ((a & b) ^ (a & c) ^ (b & c));
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
  static const uint32_t v[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                0x1f83d9abu, 0x5be0cd19u};
  memcpy(s->h, v, sizeof(v));
  s->bits = 0;
  s->used = 0;
}
static void sha_update(sha256_state *s, const uint8_t *p, size_t n) {
  s->bits += (uint64_t)n * 8u;
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
      s->used = 0;
    }
  }
}
static void sha_final(sha256_state *s, uint8_t out[32]) {
  size_t i;
  s->block[s->used++] = 0x80u;
  if (s->used > 56u) {
    memset(s->block + s->used, 0, 64u - s->used);
    sha_block(s, s->block);
    s->used = 0;
  }
  memset(s->block + s->used, 0, 56u - s->used);
  for (i = 0; i < 8; i++)
    s->block[63u - i] = (uint8_t)(s->bits >> (8u * i));
  sha_block(s, s->block);
  for (i = 0; i < 8; i++)
    put_be32(out + 4u * i, s->h[i]);
}

typedef struct cursor {
  const uint8_t *begin, *at, *end;
} cursor;
enum { P_DEFAULT = 1, P_DERIVED = 2, P_INPUT = 4, P_OVERRIDE = 8 };
static int take(cursor *c, const char *s) {
  size_t n = strlen(s);
  if ((size_t)(c->end - c->at) < n || memcmp(c->at, s, n) != 0)
    return 0;
  c->at += n;
  return 1;
}
static int number(cursor *c, uint64_t *v, int positive) {
  uint64_t x = 0;
  const uint8_t *start = c->at;
  if (start == c->end || *start < '0' || *start > '9')
    return 0;
  if (*start == '0' && start + 1 < c->end && start[1] >= '0' && start[1] <= '9')
    return 0;
  while (c->at < c->end && *c->at >= '0' && *c->at <= '9') {
    uint32_t d = (uint32_t)(*c->at - '0');
    if (x > ((uint64_t)INT64_MAX - d) / 10u)
      return 0;
    x = x * 10u + d;
    c->at++;
  }
  if ((positive && x == 0u) || c->at == start)
    return 0;
  *v = x;
  return 1;
}
static int provenance(cursor *c, uint32_t allowed, uint32_t *out) {
  struct item {
    const char *s;
    uint32_t bit;
  } x[] = {{"\"default\"", P_DEFAULT},
           {"\"derived\"", P_DERIVED},
           {"\"input\"", P_INPUT},
           {"\"override\"", P_OVERRIDE}};
  size_t i;
  for (i = 0; i < 4; i++) {
    cursor copy = *c;
    if ((allowed & x[i].bit) && take(&copy, x[i].s)) {
      *c = copy;
      *out = x[i].bit;
      return 1;
    }
  }
  return 0;
}
static int utf8(const uint8_t *s, size_t n) {
  size_t i = 0;
  while (i < n) {
    uint8_t a = s[i++];
    if (a < 0x80)
      continue;
    if (a < 0xc2 || a > 0xf4)
      return 0;
    if (a <= 0xdf) {
      if (i >= n || (s[i++] & 0xc0) != 0x80)
        return 0;
    } else if (a <= 0xef) {
      if (i + 1 >= n || (s[i] & 0xc0) != 0x80 || (s[i + 1] & 0xc0) != 0x80 ||
          (a == 0xe0 && s[i] < 0xa0) || (a == 0xed && s[i] >= 0xa0))
        return 0;
      i += 2;
    } else {
      if (i + 2 >= n || (s[i] & 0xc0) != 0x80 || (s[i + 1] & 0xc0) != 0x80 ||
          (s[i + 2] & 0xc0) != 0x80 || (a == 0xf0 && s[i] < 0x90) ||
          (a == 0xf4 && s[i] > 0x8f))
        return 0;
      i += 3;
    }
  }
  return 1;
}
static const uint8_t *find_bytes(const uint8_t *p, size_t n,
                                 const char *needle) {
  size_t m = strlen(needle), i;
  if (m > n)
    return NULL;
  for (i = 0; i <= n - m; i++)
    if (memcmp(p + i, needle, m) == 0)
      return p + i;
  return NULL;
}
static int typed_version_mismatch(const uint8_t *p, size_t n) {
  static const char *keys[] = {"\"config-schema-version\":",
                               "\"format-version\":"};
  size_t i;
  for (i = 0; i < 2; i++) {
    const uint8_t *f = find_bytes(p, n, keys[i]);
    if (f) {
      cursor c = {p, f + strlen(keys[i]), p + n};
      uint64_t a, b;
      if (take(&c, "[") && number(&c, &a, 0) && take(&c, ",") &&
          number(&c, &b, 0) && take(&c, "]") && (a != 1 || b != 0))
        return 1;
    }
  }
  {
    const char *key = "\"required-features\":";
    const uint8_t *f = find_bytes(p, n, key);
    if (f) {
      f += strlen(key);
      if ((size_t)(p + n - f) >= 4u && f[0] == '[' && f[1] == '\"')
        return 1;
    }
  }
  return 0;
}
static uint32_t domain_mismatch(const uint8_t *p, size_t n) {
  const uint8_t *r = find_bytes(p, n, "\"resolved\":{");
  const uint8_t *end = p + n, *f;
  if (!r)
    return ET_C2_X1_CORRUPT_DATA;
  f = find_bytes(r, (size_t)(end - r), "\"model.device\":");
  if (f) {
    f += strlen("\"model.device\":");
    if (f < end && *f == '\"') {
      const uint8_t *a = ++f;
      while (f < end && *f >= 0x20 && *f <= 0x7e && *f != '\"' && *f != '\\')
        f++;
      if (f < end && *f == '\"' &&
          !((size_t)(f - a) == 3 && memcmp(a, "cpu", 3) == 0))
        return ET_C2_X1_DEVICE_MISMATCH;
    }
  }
  f = find_bytes(r, (size_t)(end - r), "\"model.dtype\":");
  if (f) {
    f += strlen("\"model.dtype\":");
    if (f < end && *f == '\"') {
      const uint8_t *a = ++f;
      while (f < end && *f >= 0x20 && *f <= 0x7e && *f != '\"' && *f != '\\')
        f++;
      if (f < end && *f == '\"' &&
          !((size_t)(f - a) == 3 && memcmp(a, "f32", 3) == 0))
        return ET_C2_X1_DTYPE_MISMATCH;
    }
  }
  return ET_C2_X1_CORRUPT_DATA;
}
static int fingerprint(const uint8_t *fp, size_t n, const uint8_t *p,
                       size_t bytes) {
  static const char prefix[] = "sha256:eshkol-config-json-v1:";
  static const char hex[] = "0123456789abcdef";
  sha256_state s;
  uint8_t d[32];
  size_t i;
  if (n != 93u || memcmp(fp, prefix, sizeof(prefix) - 1u) != 0)
    return 0;
  for (i = sizeof(prefix) - 1u; i < n; i++)
    if (!((fp[i] >= '0' && fp[i] <= '9') || (fp[i] >= 'a' && fp[i] <= 'f')))
      return 0;
  sha_init(&s);
  sha_update(&s, p, bytes);
  sha_final(&s, d);
  for (i = 0; i < 32; i++)
    if (fp[29u + 2u * i] != hex[d[i] >> 4u] ||
        fp[30u + 2u * i] != hex[d[i] & 15u])
      return 0;
  return 1;
}
static int32_t report(et_c2_x1_error_v1 *e, uint32_t category, uint32_t code,
                      uint64_t off) {
  if (e) {
    e->category = category;
    e->code = code;
    e->offset = off;
  }
  return (int32_t)category;
}

static int overlap(const void *a, size_t an, const void *b, size_t bn) {
  uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
  if (an == 0 || bn == 0)
    return 0;
  if (x > UINTPTR_MAX - an || y > UINTPTR_MAX - bn)
    return 1;
  return x < y + bn && y < x + an;
}

int32_t et_c2_private_x1_canonical_inspect_v1(const uint8_t *p, size_t n,
                                              const uint8_t *fp, size_t fn,
                                              et_c2_x1_projection_v1 *out,
                                              et_c2_x1_error_v1 *err) {
  static const char prefix[] =
      "{\"canonicalization\":\"eshkol-config-json-v1\",\"checksum-algorithm\":"
      "\"sha256\",\"checksum-coverage\":\"whole-document-including-final-lf\","
      "\"config-schema-version\":[1,0],\"format\":\"eshkol-resolved-run\","
      "\"format-version\":[1,0],\"limits\":{\"integer-digits\":19,\"max-input-"
      "bytes\":16384,\"max-input-keys\":14,\"max-input-nesting-depth\":1},"
      "\"provenance\":{";
  static const char *keys[] = {"\"config-schema-major\":",
                               "\"config-schema-minor\":",
                               "\"model.context-length\":",
                               "\"model.device\":",
                               "\"model.dtype\":",
                               "\"model.head-size\":",
                               "\"model.hidden-size\":",
                               "\"model.kv-head-count\":",
                               "\"model.layer-count\":",
                               "\"model.query-head-count\":",
                               "\"model.vocabulary-size\":",
                               "\"run.deterministic\":",
                               "\"run.seed\":",
                               "\"training.accumulation-steps\":"};
  static const uint32_t allowed[] = {P_INPUT,
                                     P_INPUT,
                                     P_INPUT | P_OVERRIDE,
                                     P_DEFAULT | P_INPUT | P_OVERRIDE,
                                     P_DEFAULT | P_INPUT | P_OVERRIDE,
                                     P_DERIVED | P_INPUT,
                                     P_INPUT | P_OVERRIDE,
                                     P_DEFAULT | P_INPUT | P_OVERRIDE,
                                     P_INPUT | P_OVERRIDE,
                                     P_INPUT | P_OVERRIDE,
                                     P_INPUT | P_OVERRIDE,
                                     P_DEFAULT | P_INPUT | P_OVERRIDE,
                                     P_INPUT | P_OVERRIDE,
                                     P_DEFAULT | P_INPUT | P_OVERRIDE};
  cursor c;
  uint32_t prov[14];
  uint64_t context, head, hidden, kv, layers, query, vocab, seed, accum;
  size_t i;
  et_c2_x1_projection_v1 result;
  if (err && err->struct_size != sizeof(*err))
    return ET_C2_X1_INVALID_ARGUMENT;
  if (!p || !fp || !out || out->struct_size != sizeof(*out) ||
      overlap(p, n, out, sizeof(*out)) || overlap(fp, fn, out, sizeof(*out)) ||
      (err && (overlap(p, n, err, sizeof(*err)) ||
               overlap(fp, fn, err, sizeof(*err)) ||
               overlap(out, sizeof(*out), err, sizeof(*err)))))
    return report(err, ET_C2_X1_INVALID_ARGUMENT, ET_C2_X1_CODE_ARGUMENT, 0);
  if (n > ET_C2_X1_MAX_CANONICAL_BYTES)
    return report(err, ET_C2_X1_CORRUPT_DATA, ET_C2_X1_CODE_LIMIT, n);
  if (!utf8(p, n))
    return report(err, ET_C2_X1_CORRUPT_DATA, ET_C2_X1_CODE_UTF8, 0);
  if (!fingerprint(fp, fn, p, n))
    return report(err, ET_C2_X1_CORRUPT_DATA, ET_C2_X1_CODE_FINGERPRINT, 0);
  c.begin = p;
  c.at = p;
  c.end = p + n;
  if (!take(&c, prefix))
    goto malformed;
  for (i = 0; i < 14; i++)
    if (!take(&c, keys[i]) || !provenance(&c, allowed[i], &prov[i]) ||
        (i != 13 && !take(&c, ",")))
      goto malformed;
  if (!take(&c,
            "},\"required-features\":[],\"resolved\":{\"config-schema-major\":"
            "1,\"config-schema-minor\":0,\"model.context-length\":") ||
      !number(&c, &context, 1) ||
      !take(&c, ",\"model.device\":\"cpu\",\"model.dtype\":\"f32\",\"model."
                "head-size\":") ||
      !number(&c, &head, 1) || !take(&c, ",\"model.hidden-size\":") ||
      !number(&c, &hidden, 1) || !take(&c, ",\"model.kv-head-count\":") ||
      !number(&c, &kv, 1) || !take(&c, ",\"model.layer-count\":") ||
      !number(&c, &layers, 1) || !take(&c, ",\"model.query-head-count\":") ||
      !number(&c, &query, 1) || !take(&c, ",\"model.vocabulary-size\":") ||
      !number(&c, &vocab, 1) ||
      !take(&c, ",\"run.deterministic\":true,\"run.seed\":") ||
      !number(&c, &seed, 0) || !take(&c, ",\"training.accumulation-steps\":") ||
      !number(&c, &accum, 1) || !take(&c, "}}\n") || c.at != c.end)
    goto malformed;
  if (query > UINT64_MAX / head || hidden != query * head || kv == 0 ||
      query % kv != 0)
    return report(err, ET_C2_X1_SHAPE_MISMATCH, ET_C2_X1_CODE_DIMENSION,
                  (uint64_t)(c.at - c.begin));
  if ((prov[7] == P_DEFAULT && kv != query) ||
      (prov[13] == P_DEFAULT && accum != 1) ||
      (prov[5] == P_DERIVED && head != hidden / query))
    return report(err, ET_C2_X1_CORRUPT_DATA, ET_C2_X1_CODE_PROVENANCE, 0);
  memset(&result, 0, sizeof(result));
  result.struct_size = sizeof(result);
  result.vocabulary_size = vocab;
  result.context_length = context;
  result.hidden_size = hidden;
  result.layer_count = layers;
  result.query_head_count = query;
  result.kv_head_count = kv;
  result.head_size = head;
  result.seed = seed;
  result.accumulation_steps = accum;
  *out = result;
  if (err) {
    err->category = 0;
    err->code = 0;
    err->offset = 0;
  }
  return ET_C2_X1_OK;
malformed: {
  uint32_t cat = typed_version_mismatch(p, n) ? ET_C2_X1_VERSION_MISMATCH
                                              : domain_mismatch(p, n);
  uint32_t code = cat == ET_C2_X1_VERSION_MISMATCH ? ET_C2_X1_CODE_VERSION
                                                   : ET_C2_X1_CODE_CANONICAL;
  return report(err, cat, code, (uint64_t)(c.at - c.begin));
}
}

static int bytevector(void *header, size_t maximum, size_t exact,
                      const uint8_t **payload, size_t *bytes) {
  int64_t n;
  uintptr_t start;
  if (!header)
    return 0;
  memcpy(&n, header, sizeof(n));
  if (n < 0 || (uint64_t)n > maximum ||
      (exact != SIZE_MAX && (uint64_t)n != exact))
    return 0;
  start = (uintptr_t)header;
  if (start > UINTPTR_MAX - sizeof(n) ||
      (size_t)n > UINTPTR_MAX - start - sizeof(n))
    return 0;
  *payload = (const uint8_t *)(start + sizeof(n));
  *bytes = (size_t)n;
  return 1;
}
static void put_le64(uint8_t *p, uint64_t x) {
  size_t i;
  for (i = 0; i < 8; i++)
    p[i] = (uint8_t)(x >> (8u * i));
}
int64_t et_c2_private_x1_canonical_inspect_bytevectors_v1(void *ch, void *fh,
                                                          void *oh) {
  const uint8_t *c, *f, *o;
  size_t cn, fn, on;
  et_c2_x1_projection_v1 p;
  et_c2_x1_error_v1 e;
  uint8_t *out;
  uint64_t values[9];
  size_t i;
  int32_t status;
  if (!bytevector(ch, ET_C2_X1_MAX_CANONICAL_BYTES, SIZE_MAX, &c, &cn) ||
      !bytevector(fh, 93u, 93u, &f, &fn) ||
      !bytevector(oh, 72u, 72u, &o, &on) ||
      overlap(ch, sizeof(int64_t) + cn, oh, sizeof(int64_t) + on) ||
      overlap(fh, sizeof(int64_t) + fn, oh, sizeof(int64_t) + on))
    return ET_C2_X1_INVALID_ARGUMENT;
  memset(&p, 0, sizeof(p));
  memset(&e, 0, sizeof(e));
  p.struct_size = sizeof(p);
  e.struct_size = sizeof(e);
  status = et_c2_private_x1_canonical_inspect_v1(c, cn, f, fn, &p, &e);
  if (status != ET_C2_X1_OK)
    return status;
  values[0] = p.vocabulary_size;
  values[1] = p.context_length;
  values[2] = p.hidden_size;
  values[3] = p.layer_count;
  values[4] = p.query_head_count;
  values[5] = p.kv_head_count;
  values[6] = p.head_size;
  values[7] = p.seed;
  values[8] = p.accumulation_steps;
  out = (uint8_t *)(uintptr_t)o;
  for (i = 0; i < 9; i++)
    put_le64(out + 8u * i, values[i]);
  return 0;
}
