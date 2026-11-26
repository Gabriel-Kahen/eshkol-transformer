#include "eshkol_transformer/data4_pipeline_abi.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#else
#include <fenv.h>
#endif

_Static_assert(CHAR_BIT == 8 && sizeof(size_t) <= sizeof(uint64_t),
               "DATA4 pipeline requires 8-bit bytes and at most 64-bit size_t");
_Static_assert(FLT_RADIX == 2 && DBL_MANT_DIG == 53 && DBL_MAX_EXP == 1024 &&
                   sizeof(double) == sizeof(uint64_t),
               "DATA4 pipeline requires IEEE-754 binary64");

/* ---- Floating-point environment guard ---- */

static int fp_environment_supported(void) {
#if defined(__x86_64__)
  uint16_t x87;
  __asm__ volatile("fnstcw %0" : "=m"(x87));
  return (x87 & 0x0c3fu) == 0x003fu && (_mm_getcsr() & 0xffc0u) == 0x1f80u;
#else
  return fegetround() == FE_TONEAREST;
#endif
}

/* ---- Overflow, span, and aliasing helpers ---- */

static int mul_size_overflow(size_t a, size_t b, size_t *out) {
  if (a != 0u && b > SIZE_MAX / a) return 1;
  *out = a * b;
  return 0;
}

static int add_size_overflow(size_t a, size_t b, size_t *out) {
  if (b > SIZE_MAX - a) return 1;
  *out = a + b;
  return 0;
}

static int valid_span(const void *p, size_t alignment, size_t bytes) {
  if (p == NULL) return bytes == 0u;
  if (alignment > 1u && ((uintptr_t)p % alignment) != 0u) return 0;
  return (uintptr_t)p <= UINTPTR_MAX - bytes;
}

static int spans_overlap(const void *a, size_t a_bytes, const void *b,
                         size_t b_bytes) {
  if (a == NULL || b == NULL || a_bytes == 0u || b_bytes == 0u) return 0;
  const uintptr_t left = (uintptr_t)a;
  const uintptr_t right = (uintptr_t)b;
  if (left > UINTPTR_MAX - a_bytes || right > UINTPTR_MAX - b_bytes) return 1;
  return left < right + b_bytes && right < left + a_bytes;
}

/* ---- Standalone SHA-256 (FIPS 180-4) ---- */

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
  size_t i;
  for (i = 0u; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = state->h[0];
  uint32_t b = state->h[1];
  uint32_t c = state->h[2];
  uint32_t d = state->h[3];
  uint32_t e = state->h[4];
  uint32_t f = state->h[5];
  uint32_t g = state->h[6];
  uint32_t h = state->h[7];
  for (i = 0u; i < 64u; ++i) {
    const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
    const uint32_t ch = (e & f) ^ ((~e) & g);
    const uint32_t t1 = h + s1 + ch + k[i] + w[i];
    const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
    const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + maj;
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  state->h[0] += a;
  state->h[1] += b;
  state->h[2] += c;
  state->h[3] += d;
  state->h[4] += e;
  state->h[5] += f;
  state->h[6] += g;
  state->h[7] += h;
}

static void sha256_init(sha256_state *state) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
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
    state->used += take;
    bytes += take;
    length -= take;
    if (state->used == 64u) {
      sha256_block(state, state->block);
      state->used = 0u;
    }
  }
}

static void sha256_final(sha256_state *state, uint8_t digest[32]) {
  const uint64_t bits = state->length * UINT64_C(8);
  state->block[state->used++] = 0x80u;
  if (state->used > 56u) {
    memset(state->block + state->used, 0, 64u - state->used);
    sha256_block(state, state->block);
    state->used = 0u;
  }
  memset(state->block + state->used, 0, 56u - state->used);
  for (size_t i = 0u; i < 8u; ++i)
    state->block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(state, state->block);
  for (size_t i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state->h[i];
  }
}

static void sha256_raw(const uint8_t *bytes, size_t size, uint8_t digest[32]) {
  sha256_state state;
  sha256_init(&state);
  if (size != 0u) sha256_update(&state, bytes, size);
  sha256_final(&state, digest);
}

et_data4_status et_data4_sha256(const uint8_t *bytes, size_t byte_count,
                                uint8_t digest_out[ET_DATA4_SHA256_BYTES]) {
  if (digest_out == NULL || (byte_count != 0u && bytes == NULL))
    return ET_DATA4_ERR_ARGUMENT;
  if (byte_count > PTRDIFF_MAX || (uint64_t)byte_count > UINT64_MAX / 8u)
    return ET_DATA4_ERR_OVERFLOW;
  if (!valid_span(bytes, 1u, byte_count) ||
      !valid_span(digest_out, 1u, ET_DATA4_SHA256_BYTES))
    return ET_DATA4_ERR_BOUNDS;
  if (spans_overlap(bytes, byte_count, digest_out, ET_DATA4_SHA256_BYTES))
    return ET_DATA4_ERR_ALIASING;
  sha256_raw(bytes, byte_count, digest_out);
  return ET_DATA4_OK;
}

/* ---- SplitMix64 & FNV-1a 64-bit deterministic shingle hashing ---- */

static uint64_t splitmix64_mix(uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

static uint64_t hash_bytes64(const uint8_t *bytes, size_t len) {
  uint64_t h = UINT64_C(0xcbf29ce484222325);
  for (size_t i = 0u; i < len; ++i) {
    h ^= (uint64_t)bytes[i];
    h *= UINT64_C(0x100000001b3);
  }
  return splitmix64_mix(h);
}

static uint64_t load_u64_be(const uint8_t p[8]) {
  return ((uint64_t)p[0] << 56) | ((uint64_t)p[1] << 48) |
         ((uint64_t)p[2] << 40) | ((uint64_t)p[3] << 32) |
         ((uint64_t)p[4] << 24) | ((uint64_t)p[5] << 16) |
         ((uint64_t)p[6] << 8) | (uint64_t)p[7];
}

/* ---- Source ID & Config validation ---- */

static int valid_source_id_char(unsigned char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
}

static int valid_source_id_span(const char *s, size_t len) {
  if (s == NULL || len == 0u || len > ET_DATA4_SOURCE_ID_MAX) return 0;
  for (size_t i = 0u; i < len; ++i) {
    if (!valid_source_id_char((unsigned char)s[i])) return 0;
  }
  return 1;
}

static int valid_source_id_cstr(const char *s, size_t *len_out) {
  if (s == NULL) return 0;
  size_t len = 0u;
  while (len <= ET_DATA4_SOURCE_ID_MAX && s[len] != '\0') {
    if (!valid_source_id_char((unsigned char)s[len])) return 0;
    ++len;
  }
  if (len == 0u || len > ET_DATA4_SOURCE_ID_MAX) return 0;
  if (len_out != NULL) *len_out = len;
  return 1;
}

static int valid_ratio(double v) {
  return isfinite(v) && !signbit(v) && v <= 1.0;
}

static int valid_config(const et_data4_pipeline_config *cfg) {
  if (cfg == NULL) return 0;
  if (cfg->reserved0 != 0u) return 0;
  if (cfg->min_bytes > cfg->max_bytes) return 0;
  if (cfg->min_codepoints > cfg->max_codepoints) return 0;
  if (!valid_ratio(cfg->min_printable_ratio) ||
      !valid_ratio(cfg->min_alnum_ratio) ||
      !valid_ratio(cfg->max_dup_line_ratio) ||
      !valid_ratio(cfg->max_repeated_ngram_ratio))
    return 0;
  if (cfg->rep_ngram_size == 0u ||
      cfg->rep_ngram_size > ET_DATA4_MAX_NGRAM_SIZE)
    return 0;
  if (cfg->near_dup_enabled > 1u) return 0;
  if (cfg->near_dup_enabled == 1u) {
    if (cfg->dedup_ngram_size == 0u ||
        cfg->dedup_ngram_size > ET_DATA4_MAX_NGRAM_SIZE)
      return 0;
    if (cfg->minhash_num_perms == 0u ||
        cfg->minhash_num_perms > ET_DATA4_MINHASH_MAX_PERMS)
      return 0;
    if (cfg->minhash_bands == 0u || cfg->minhash_bands > cfg->minhash_num_perms ||
        (cfg->minhash_num_perms % cfg->minhash_bands) != 0u)
      return 0;
    if (!isfinite(cfg->near_dup_jaccard_threshold) ||
        cfg->near_dup_jaccard_threshold <= 0.0 ||
        cfg->near_dup_jaccard_threshold > 1.0)
      return 0;
  } else {
    if (cfg->dedup_ngram_size > ET_DATA4_MAX_NGRAM_SIZE ||
        cfg->minhash_num_perms > ET_DATA4_MINHASH_MAX_PERMS ||
        cfg->minhash_bands > ET_DATA4_MINHASH_MAX_PERMS)
      return 0;
    if (!valid_ratio(cfg->near_dup_jaccard_threshold)) return 0;
  }
  return 1;
}

et_data4_status et_data4_config_default(et_data4_pipeline_config *config) {
  if (config == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (!valid_span(config, _Alignof(et_data4_pipeline_config), sizeof(*config)))
    return ET_DATA4_ERR_ALIGNMENT;
  *config = (et_data4_pipeline_config){
      .min_bytes = 1u,
      .max_bytes = 1048576u,
      .min_codepoints = 1u,
      .max_codepoints = 1048576u,
      .min_printable_ratio = 0.90,
      .min_alnum_ratio = 0.50,
      .max_dup_line_ratio = 0.30,
      .max_repeated_ngram_ratio = 0.40,
      .rep_ngram_size = 5u,
      .near_dup_enabled = 1u,
      .dedup_ngram_size = 5u,
      .minhash_num_perms = 64u,
      .minhash_bands = 16u,
      .reserved0 = 0u,
      .minhash_seed = UINT64_C(0x4441544134534545),
      .near_dup_jaccard_threshold = 0.80,
  };
  return ET_DATA4_OK;
}

const char *et_data4_disposition_name(et_data4_disposition disposition) {
  switch (disposition) {
    case ET_DATA4_DISP_KEPT:
      return "kept";
    case ET_DATA4_DISP_REJECTED_UTF8:
      return "rejected_utf8";
    case ET_DATA4_DISP_REJECTED_LENGTH:
      return "rejected_length";
    case ET_DATA4_DISP_REJECTED_QUALITY:
      return "rejected_quality";
    case ET_DATA4_DISP_REJECTED_REPETITION:
      return "rejected_repetition";
    case ET_DATA4_DISP_DUPLICATE_EXACT:
      return "duplicate_exact";
    case ET_DATA4_DISP_DUPLICATE_NEAR:
      return "duplicate_near";
    default:
      return NULL;
  }
}

/* ---- Strict RFC 3629 UTF-8 validation & deterministic classification ---- */

static int is_printable_cp(uint32_t cp) {
  if (cp == 0x09u || cp == 0x0Au || cp == 0x0Du || cp == 0x20u) return 1;
  if (cp >= 0x21u && cp <= 0x7Eu) return 1;
  if (cp < 0xA0u) return 0;
  if (cp >= 0xFDD0u && cp <= 0xFDEFu) return 0;
  if ((cp & 0xFFFEu) == 0xFFFEu) return 0;
  return 1;
}

static int is_alnum_cp(uint32_t cp) {
  if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') ||
      (cp >= '0' && cp <= '9'))
    return 1;
  if ((cp >= 0x00C0u && cp <= 0x00D6u) || (cp >= 0x00D8u && cp <= 0x00F6u) ||
      (cp >= 0x00F8u && cp <= 0x024Fu))
    return 1;
  if ((cp >= 0x0370u && cp <= 0x03FFu) || (cp >= 0x0400u && cp <= 0x04FFu))
    return 1;
  if ((cp >= 0x3040u && cp <= 0x30FFu) || (cp >= 0x4E00u && cp <= 0x9FFFu) ||
      (cp >= 0xAC00u && cp <= 0xD7A3u))
    return 1;
  return 0;
}

et_data4_status et_data4_validate_utf8(const uint8_t *bytes, size_t byte_count,
                                       uint64_t *codepoint_count,
                                       uint64_t *printable_count,
                                       uint64_t *alnum_count) {
  if (byte_count != 0u && bytes == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (byte_count > PTRDIFF_MAX) return ET_DATA4_ERR_OVERFLOW;
  if (!valid_span(bytes, 1u, byte_count)) return ET_DATA4_ERR_BOUNDS;
  if ((codepoint_count != NULL &&
       !valid_span(codepoint_count, _Alignof(uint64_t), sizeof(uint64_t))) ||
      (printable_count != NULL &&
       !valid_span(printable_count, _Alignof(uint64_t), sizeof(uint64_t))) ||
      (alnum_count != NULL &&
       !valid_span(alnum_count, _Alignof(uint64_t), sizeof(uint64_t))))
    return ET_DATA4_ERR_ALIGNMENT;
  if ((codepoint_count != NULL &&
       spans_overlap(bytes, byte_count, codepoint_count, sizeof(uint64_t))) ||
      (printable_count != NULL &&
       spans_overlap(bytes, byte_count, printable_count, sizeof(uint64_t))) ||
      (alnum_count != NULL &&
       spans_overlap(bytes, byte_count, alnum_count, sizeof(uint64_t))) ||
      (codepoint_count != NULL && printable_count != NULL &&
       spans_overlap(codepoint_count, sizeof(uint64_t), printable_count,
                     sizeof(uint64_t))) ||
      (codepoint_count != NULL && alnum_count != NULL &&
       spans_overlap(codepoint_count, sizeof(uint64_t), alnum_count,
                     sizeof(uint64_t))) ||
      (printable_count != NULL && alnum_count != NULL &&
       spans_overlap(printable_count, sizeof(uint64_t), alnum_count,
                     sizeof(uint64_t))))
    return ET_DATA4_ERR_ALIASING;

  uint64_t cp_total = 0u;
  uint64_t print_total = 0u;
  uint64_t alnum_total = 0u;

  size_t i = 0u;
  while (i < byte_count) {
    const uint8_t b0 = bytes[i];
    uint32_t cp = 0u;
    if (b0 <= 0x7Fu) {
      cp = (uint32_t)b0;
      i += 1u;
    } else if (b0 >= 0xC2u && b0 <= 0xDFu) {
      if (i + 1u >= byte_count) return ET_DATA4_ERR_FORMAT;
      const uint8_t b1 = bytes[i + 1u];
      if ((b1 & 0xC0u) != 0x80u) return ET_DATA4_ERR_FORMAT;
      cp = (((uint32_t)b0 & 0x1Fu) << 6) | ((uint32_t)b1 & 0x3Fu);
      i += 2u;
    } else if (b0 >= 0xE0u && b0 <= 0xEFu) {
      if (i + 2u >= byte_count) return ET_DATA4_ERR_FORMAT;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u)
        return ET_DATA4_ERR_FORMAT;
      if (b0 == 0xE0u && b1 < 0xA0u) return ET_DATA4_ERR_FORMAT;
      if (b0 == 0xEDu && b1 >= 0xA0u) return ET_DATA4_ERR_FORMAT;
      cp = (((uint32_t)b0 & 0x0Fu) << 12) | (((uint32_t)b1 & 0x3Fu) << 6) |
           ((uint32_t)b2 & 0x3Fu);
      i += 3u;
    } else if (b0 >= 0xF0u && b0 <= 0xF4u) {
      if (i + 3u >= byte_count) return ET_DATA4_ERR_FORMAT;
      const uint8_t b1 = bytes[i + 1u];
      const uint8_t b2 = bytes[i + 2u];
      const uint8_t b3 = bytes[i + 3u];
      if ((b1 & 0xC0u) != 0x80u || (b2 & 0xC0u) != 0x80u ||
          (b3 & 0xC0u) != 0x80u)
        return ET_DATA4_ERR_FORMAT;
      if (b0 == 0xF0u && b1 < 0x90u) return ET_DATA4_ERR_FORMAT;
      if (b0 == 0xF4u && b1 > 0x8Fu) return ET_DATA4_ERR_FORMAT;
      cp = (((uint32_t)b0 & 0x07u) << 18) | (((uint32_t)b1 & 0x3Fu) << 12) |
           (((uint32_t)b2 & 0x3Fu) << 6) | ((uint32_t)b3 & 0x3Fu);
      i += 4u;
    } else {
      return ET_DATA4_ERR_FORMAT;
    }
    ++cp_total;
    if (is_printable_cp(cp)) ++print_total;
    if (is_alnum_cp(cp)) ++alnum_total;
  }

  if (codepoint_count != NULL) *codepoint_count = cp_total;
  if (printable_count != NULL) *printable_count = print_total;
  if (alnum_count != NULL) *alnum_count = alnum_total;
  return ET_DATA4_OK;
}

/* ---- Duplicate line & repeated n-gram measurement ---- */

typedef struct line_slice {
  const uint8_t *ptr;
  size_t len;
  uint64_t hash;
} line_slice;

static int compare_line_slice(const void *left, const void *right) {
  const line_slice *a = (const line_slice *)left;
  const line_slice *b = (const line_slice *)right;
  if (a->hash != b->hash) return a->hash < b->hash ? -1 : 1;
  if (a->len != b->len) return a->len < b->len ? -1 : 1;
  return a->len == 0u ? 0 : memcmp(a->ptr, b->ptr, a->len);
}

static et_data4_status measure_duplicate_lines(const uint8_t *bytes,
                                               size_t byte_count,
                                               uint64_t *total_lines_out,
                                               uint64_t *dup_lines_out) {
  if (byte_count == 0u) {
    *total_lines_out = 0u;
    *dup_lines_out = 0u;
    return ET_DATA4_OK;
  }
  size_t count = 0u;
  size_t pos = 0u;
  while (pos < byte_count) {
    const uint8_t *nl = (const uint8_t *)memchr(bytes + pos, '\n', byte_count - pos);
    ++count;
    if (nl == NULL) break;
    pos = (size_t)(nl - bytes) + 1u;
  }
  size_t alloc_bytes = 0u;
  if (mul_size_overflow(count, sizeof(line_slice), &alloc_bytes) ||
      alloc_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;
  line_slice *slices = (line_slice *)malloc(alloc_bytes);
  if (slices == NULL) return ET_DATA4_ERR_NOMEM;

  size_t idx = 0u;
  pos = 0u;
  while (pos < byte_count) {
    const uint8_t *nl = (const uint8_t *)memchr(bytes + pos, '\n', byte_count - pos);
    const size_t end = nl == NULL ? byte_count : (size_t)(nl - bytes);
    const size_t len = end - pos;
    slices[idx++] =
        (line_slice){bytes + pos, len, hash_bytes64(bytes + pos, len)};
    if (nl == NULL) break;
    pos = end + 1u;
  }

  qsort(slices, count, sizeof(*slices), compare_line_slice);
  size_t unique = 1u;
  for (size_t i = 1u; i < count; ++i) {
    if (compare_line_slice(&slices[i - 1u], &slices[i]) != 0) ++unique;
  }
  free(slices);

  *total_lines_out = (uint64_t)count;
  *dup_lines_out = (uint64_t)(count - unique);
  return ET_DATA4_OK;
}

/* ---- Canonical sorted unique byte n-gram shingle sets ---- */

typedef struct shingle_entry {
  uint64_t h0;
  const uint8_t *ptr;
  uint32_t len;
} shingle_entry;

static int compare_shingle_entry(const void *left, const void *right) {
  const shingle_entry *a = (const shingle_entry *)left;
  const shingle_entry *b = (const shingle_entry *)right;
  if (a->h0 != b->h0) return a->h0 < b->h0 ? -1 : 1;
  if (a->len != b->len) return a->len < b->len ? -1 : 1;
  return memcmp(a->ptr, b->ptr, a->len);
}

typedef struct shingle_set {
  shingle_entry *items;
  size_t count;
} shingle_set;

static void shingle_set_free(shingle_set *set) {
  free(set->items);
  set->items = NULL;
  set->count = 0u;
}

static et_data4_status build_shingle_set(const uint8_t *bytes,
                                         size_t byte_count,
                                         uint32_t ngram_size,
                                         shingle_set *out_set,
                                         uint64_t *total_windows_out) {
  out_set->items = NULL;
  out_set->count = 0u;
  if (byte_count < (size_t)ngram_size) {
    if (total_windows_out != NULL) *total_windows_out = 0u;
    return ET_DATA4_OK;
  }
  const size_t windows = byte_count - (size_t)ngram_size + 1u;
  if (total_windows_out != NULL) *total_windows_out = (uint64_t)windows;

  size_t alloc_bytes = 0u;
  if (mul_size_overflow(windows, sizeof(shingle_entry), &alloc_bytes) ||
      alloc_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;
  shingle_entry *entries = (shingle_entry *)malloc(alloc_bytes);
  if (entries == NULL) return ET_DATA4_ERR_NOMEM;

  for (size_t i = 0u; i < windows; ++i) {
    entries[i] = (shingle_entry){
        .h0 = hash_bytes64(bytes + i, (size_t)ngram_size),
        .ptr = bytes + i,
        .len = ngram_size,
    };
  }
  qsort(entries, windows, sizeof(*entries), compare_shingle_entry);

  size_t unique = 1u;
  for (size_t i = 1u; i < windows; ++i) {
    if (compare_shingle_entry(&entries[unique - 1u], &entries[i]) != 0)
      entries[unique++] = entries[i];
  }
  out_set->items = entries;
  out_set->count = unique;
  return ET_DATA4_OK;
}

static void exact_jaccard_sets(const shingle_set *a, const shingle_set *b,
                               uint64_t *num_out, uint64_t *den_out) {
  size_t i = 0u;
  size_t j = 0u;
  uint64_t inter = 0u;
  while (i < a->count && j < b->count) {
    const int cmp = compare_shingle_entry(&a->items[i], &b->items[j]);
    if (cmp == 0) {
      ++inter;
      ++i;
      ++j;
    } else if (cmp < 0) {
      ++i;
    } else {
      ++j;
    }
  }
  const uint64_t uni = (uint64_t)a->count + (uint64_t)b->count - inter;
  *num_out = inter;
  *den_out = uni;
}

static void compute_minhash_from_set(const shingle_set *set,
                                     uint32_t num_perms, uint64_t seed,
                                     uint64_t *sig_out) {
  for (uint32_t k = 0u; k < num_perms; ++k) sig_out[k] = UINT64_MAX;
  if (set->count == 0u) return;
  uint64_t a_coeff[ET_DATA4_MINHASH_MAX_PERMS];
  uint64_t b_coeff[ET_DATA4_MINHASH_MAX_PERMS];
  for (uint32_t k = 0u; k < num_perms; ++k) {
    a_coeff[k] = splitmix64_mix(seed ^ ((uint64_t)k * UINT64_C(2) + UINT64_C(1))) |
                 UINT64_C(1);
    b_coeff[k] = splitmix64_mix(seed ^ ((uint64_t)k * UINT64_C(2) + UINT64_C(2)));
  }
  for (size_t i = 0u; i < set->count; ++i) {
    const uint64_t h0 = set->items[i].h0;
    for (uint32_t k = 0u; k < num_perms; ++k) {
      const uint64_t permuted = splitmix64_mix(a_coeff[k] * h0 + b_coeff[k]);
      if (permuted < sig_out[k]) sig_out[k] = permuted;
    }
  }
}

et_data4_status et_data4_compute_minhash(const uint8_t *bytes,
                                         size_t byte_count,
                                         uint32_t ngram_size,
                                         uint32_t num_perms, uint64_t seed,
                                         uint64_t *signature_out,
                                         size_t signature_capacity,
                                         uint64_t *unique_ngram_count_out) {
  if ((byte_count != 0u && bytes == NULL) || signature_out == NULL ||
      ngram_size == 0u || ngram_size > ET_DATA4_MAX_NGRAM_SIZE ||
      num_perms == 0u || num_perms > ET_DATA4_MINHASH_MAX_PERMS)
    return ET_DATA4_ERR_ARGUMENT;
  if (signature_capacity < (size_t)num_perms) return ET_DATA4_ERR_BOUNDS;
  if (byte_count > PTRDIFF_MAX) return ET_DATA4_ERR_OVERFLOW;
  const size_t sig_bytes = (size_t)num_perms * sizeof(uint64_t);
  if (!valid_span(bytes, 1u, byte_count) ||
      !valid_span(signature_out, _Alignof(uint64_t), sig_bytes))
    return ET_DATA4_ERR_ALIGNMENT;
  if (unique_ngram_count_out != NULL &&
      !valid_span(unique_ngram_count_out, _Alignof(uint64_t),
                  sizeof(uint64_t)))
    return ET_DATA4_ERR_ALIGNMENT;
  if (spans_overlap(bytes, byte_count, signature_out, sig_bytes) ||
      (unique_ngram_count_out != NULL &&
       (spans_overlap(bytes, byte_count, unique_ngram_count_out,
                      sizeof(uint64_t)) ||
        spans_overlap(signature_out, sig_bytes, unique_ngram_count_out,
                      sizeof(uint64_t)))))
    return ET_DATA4_ERR_ALIASING;

  shingle_set set;
  et_data4_status st =
      build_shingle_set(bytes, byte_count, ngram_size, &set, NULL);
  if (st != ET_DATA4_OK) return st;
  compute_minhash_from_set(&set, num_perms, seed, signature_out);
  if (unique_ngram_count_out != NULL)
    *unique_ngram_count_out = (uint64_t)set.count;
  shingle_set_free(&set);
  return ET_DATA4_OK;
}

et_data4_status et_data4_exact_jaccard(const uint8_t *a_bytes, size_t a_count,
                                       const uint8_t *b_bytes, size_t b_count,
                                       uint32_t ngram_size, uint64_t *num_out,
                                       uint64_t *den_out) {
  if ((a_count != 0u && a_bytes == NULL) ||
      (b_count != 0u && b_bytes == NULL) || ngram_size == 0u ||
      ngram_size > ET_DATA4_MAX_NGRAM_SIZE || num_out == NULL ||
      den_out == NULL)
    return ET_DATA4_ERR_ARGUMENT;
  if (a_count > PTRDIFF_MAX || b_count > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;
  if (!valid_span(a_bytes, 1u, a_count) || !valid_span(b_bytes, 1u, b_count) ||
      !valid_span(num_out, _Alignof(uint64_t), sizeof(uint64_t)) ||
      !valid_span(den_out, _Alignof(uint64_t), sizeof(uint64_t)))
    return ET_DATA4_ERR_ALIGNMENT;
  if (spans_overlap(num_out, sizeof(uint64_t), den_out, sizeof(uint64_t)) ||
      spans_overlap(a_bytes, a_count, num_out, sizeof(uint64_t)) ||
      spans_overlap(a_bytes, a_count, den_out, sizeof(uint64_t)) ||
      spans_overlap(b_bytes, b_count, num_out, sizeof(uint64_t)) ||
      spans_overlap(b_bytes, b_count, den_out, sizeof(uint64_t)))
    return ET_DATA4_ERR_ALIASING;

  shingle_set set_a = {NULL, 0u};
  shingle_set set_b = {NULL, 0u};
  et_data4_status st = build_shingle_set(a_bytes, a_count, ngram_size, &set_a, NULL);
  if (st != ET_DATA4_OK) return st;
  st = build_shingle_set(b_bytes, b_count, ngram_size, &set_b, NULL);
  if (st != ET_DATA4_OK) {
    shingle_set_free(&set_a);
    return st;
  }
  exact_jaccard_sets(&set_a, &set_b, num_out, den_out);
  shingle_set_free(&set_a);
  shingle_set_free(&set_b);
  return ET_DATA4_OK;
}

/* ---- Source-ID uniqueness check via sort ---- */

typedef struct id_ref {
  const char *id;
  size_t len;
} id_ref;

static int compare_id_ref(const void *left, const void *right) {
  const id_ref *a = (const id_ref *)left;
  const id_ref *b = (const id_ref *)right;
  const size_t min_len = a->len < b->len ? a->len : b->len;
  const int cmp = memcmp(a->id, b->id, min_len);
  if (cmp != 0) return cmp;
  if (a->len != b->len) return a->len < b->len ? -1 : 1;
  return 0;
}

static et_data4_status check_unique_source_ids(const id_ref *refs,
                                               size_t count) {
  if (count <= 1u) return ET_DATA4_OK;
  size_t alloc_bytes = 0u;
  if (mul_size_overflow(count, sizeof(id_ref), &alloc_bytes) ||
      alloc_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;
  id_ref *sorted = (id_ref *)malloc(alloc_bytes);
  if (sorted == NULL) return ET_DATA4_ERR_NOMEM;
  memcpy(sorted, refs, alloc_bytes);
  qsort(sorted, count, sizeof(*sorted), compare_id_ref);
  et_data4_status st = ET_DATA4_OK;
  for (size_t i = 1u; i < count; ++i) {
    if (compare_id_ref(&sorted[i - 1u], &sorted[i]) == 0) {
      st = ET_DATA4_ERR_DUPLICATE_SOURCE_ID;
      break;
    }
  }
  free(sorted);
  return st;
}

/* ---- Hash tables for exact dedup & LSH bands ---- */

static size_t next_pow2_min16(size_t n) {
  size_t cap = 16u;
  while (cap < n && cap <= SIZE_MAX / 2u) cap <<= 1u;
  return cap;
}

typedef struct kept_record {
  size_t input_ordinal;
  const et_data4_document *doc;
  uint8_t sha256[ET_DATA4_SHA256_BYTES];
  shingle_set shingles;
  uint64_t *signature; /* length = num_perms, owned when near_dup_enabled */
} kept_record;

static uint64_t lsh_band_hash(uint32_t band, const uint64_t *band_sig,
                              uint32_t rows_per_band) {
  uint64_t bh = splitmix64_mix((uint64_t)band ^ UINT64_C(0x243f6a8885a308d3));
  for (uint32_t r = 0u; r < rows_per_band; ++r) {
    bh = splitmix64_mix(bh ^ band_sig[r]);
  }
  return bh;
}

static int compare_size_t(const void *left, const void *right) {
  const size_t a = *(const size_t *)left;
  const size_t b = *(const size_t *)right;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/* ---- Main Pipeline Execution ---- */

et_data4_status et_data4_pipeline_run(
    const et_data4_document *documents, size_t doc_count,
    const et_data4_pipeline_config *config, et_data4_doc_result *results_out,
    size_t results_capacity, et_data4_pipeline_summary *summary_out) {
  if (!fp_environment_supported()) return ET_DATA4_ERR_FP_ENV;
  if (config == NULL || summary_out == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (!valid_span(config, _Alignof(et_data4_pipeline_config), sizeof(*config)) ||
      !valid_span(summary_out, _Alignof(et_data4_pipeline_summary),
                  sizeof(*summary_out)))
    return ET_DATA4_ERR_ALIGNMENT;
  if (spans_overlap(config, sizeof(*config), summary_out, sizeof(*summary_out)))
    return ET_DATA4_ERR_ALIASING;
  if (!valid_config(config)) return ET_DATA4_ERR_ARGUMENT;

  if (doc_count == 0u) {
    if (results_out != NULL && results_capacity != 0u &&
        !valid_span(results_out, _Alignof(et_data4_doc_result), 0u))
      return ET_DATA4_ERR_ALIGNMENT;
    et_data4_pipeline_summary empty_summary;
    memset(&empty_summary, 0, sizeof(empty_summary));
    *summary_out = empty_summary;
    return ET_DATA4_OK;
  }

  if (documents == NULL || results_out == NULL) return ET_DATA4_ERR_ARGUMENT;
  if (results_capacity < doc_count) return ET_DATA4_ERR_BOUNDS;

  size_t docs_bytes = 0u;
  size_t results_bytes = 0u;
  if (mul_size_overflow(doc_count, sizeof(et_data4_document), &docs_bytes) ||
      mul_size_overflow(doc_count, sizeof(et_data4_doc_result), &results_bytes) ||
      docs_bytes > PTRDIFF_MAX || results_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;

  if (!valid_span(documents, _Alignof(et_data4_document), docs_bytes) ||
      !valid_span(results_out, _Alignof(et_data4_doc_result), results_bytes))
    return ET_DATA4_ERR_ALIGNMENT;

  if (spans_overlap(documents, docs_bytes, results_out, results_bytes) ||
      spans_overlap(documents, docs_bytes, summary_out, sizeof(*summary_out)) ||
      spans_overlap(config, sizeof(*config), results_out, results_bytes) ||
      spans_overlap(results_out, results_bytes, summary_out,
                    sizeof(*summary_out)))
    return ET_DATA4_ERR_ALIASING;

  /* Validate all source IDs, bytes spans, overflow, and aliasing upfront. */
  size_t id_refs_bytes = 0u;
  if (mul_size_overflow(doc_count, sizeof(id_ref), &id_refs_bytes) ||
      id_refs_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;
  id_ref *id_refs = (id_ref *)malloc(id_refs_bytes);
  if (id_refs == NULL) return ET_DATA4_ERR_NOMEM;

  uint64_t total_input_bytes = 0u;
  et_data4_status st = ET_DATA4_OK;
  for (size_t i = 0u; i < doc_count; ++i) {
    size_t id_len = 0u;
    if (!valid_source_id_cstr(documents[i].source_id, &id_len)) {
      st = ET_DATA4_ERR_INVALID_SOURCE_ID;
      break;
    }
    if (documents[i].byte_count != 0u && documents[i].bytes == NULL) {
      st = ET_DATA4_ERR_ARGUMENT;
      break;
    }
    if (documents[i].byte_count > PTRDIFF_MAX ||
        (uint64_t)documents[i].byte_count > UINT64_MAX / 8u) {
      st = ET_DATA4_ERR_OVERFLOW;
      break;
    }
    if ((uint64_t)documents[i].byte_count > UINT64_MAX - total_input_bytes) {
      st = ET_DATA4_ERR_OVERFLOW;
      break;
    }
    total_input_bytes += (uint64_t)documents[i].byte_count;
    if (!valid_span(documents[i].bytes, 1u, documents[i].byte_count)) {
      st = ET_DATA4_ERR_BOUNDS;
      break;
    }
    if (spans_overlap(documents[i].bytes, documents[i].byte_count, results_out,
                      results_bytes) ||
        spans_overlap(documents[i].bytes, documents[i].byte_count, summary_out,
                      sizeof(*summary_out)) ||
        spans_overlap(documents[i].source_id, id_len + 1u, results_out,
                      results_bytes) ||
        spans_overlap(documents[i].source_id, id_len + 1u, summary_out,
                      sizeof(*summary_out))) {
      st = ET_DATA4_ERR_ALIASING;
      break;
    }
    id_refs[i] = (id_ref){documents[i].source_id, id_len};
  }
  if (st == ET_DATA4_OK) st = check_unique_source_ids(id_refs, doc_count);
  free(id_refs);
  if (st != ET_DATA4_OK) return st;

  /* Allocate staging results, kept records, exact-dedup hash table, and LSH. */
  et_data4_doc_result *staged_results =
      (et_data4_doc_result *)calloc(doc_count, sizeof(et_data4_doc_result));
  kept_record *kept = (kept_record *)calloc(doc_count, sizeof(kept_record));

  size_t exact_cap = 0u;
  if (doc_count > SIZE_MAX / 2u) {
    free(staged_results);
    free(kept);
    return ET_DATA4_ERR_OVERFLOW;
  }
  exact_cap = next_pow2_min16(doc_count * 2u);
  if (exact_cap > PTRDIFF_MAX / sizeof(size_t)) {
    free(staged_results);
    free(kept);
    return ET_DATA4_ERR_OVERFLOW;
  }
  size_t *exact_table = (size_t *)malloc(exact_cap * sizeof(size_t));

  /* LSH structures (allocated only if near_dup_enabled == 1). */
  const int near_on = config->near_dup_enabled == 1u;
  const uint32_t bands = near_on ? config->minhash_bands : 0u;
  const uint32_t num_perms = near_on ? config->minhash_num_perms : 0u;
  const uint32_t rows_per_band = near_on ? (num_perms / bands) : 0u;
  size_t lsh_bucket_cap = 0u;
  size_t *lsh_heads = NULL; /* bands * lsh_bucket_cap */
  size_t *lsh_next = NULL;  /* bands * doc_count (next kept index or SIZE_MAX) */
  uint64_t *sig_pool = NULL; /* doc_count * num_perms */
  size_t *cand_buf = NULL;
  uint32_t *cand_seen = NULL;
  uint64_t *cur_sig = NULL;

  if (staged_results == NULL || kept == NULL || exact_table == NULL) {
    st = ET_DATA4_ERR_NOMEM;
  } else {
    for (size_t i = 0u; i < exact_cap; ++i) exact_table[i] = SIZE_MAX;
  }

  if (st == ET_DATA4_OK && near_on) {
    lsh_bucket_cap = exact_cap;
    size_t heads_count = 0u;
    size_t heads_bytes = 0u;
    size_t next_count = 0u;
    size_t next_bytes = 0u;
    size_t sig_count = 0u;
    size_t sig_bytes = 0u;
    if (mul_size_overflow((size_t)bands, lsh_bucket_cap, &heads_count) ||
        mul_size_overflow(heads_count, sizeof(size_t), &heads_bytes) ||
        mul_size_overflow((size_t)bands, doc_count, &next_count) ||
        mul_size_overflow(next_count, sizeof(size_t), &next_bytes) ||
        mul_size_overflow(doc_count, (size_t)num_perms, &sig_count) ||
        mul_size_overflow(sig_count, sizeof(uint64_t), &sig_bytes) ||
        heads_bytes > PTRDIFF_MAX || next_bytes > PTRDIFF_MAX ||
        sig_bytes > PTRDIFF_MAX) {
      st = ET_DATA4_ERR_OVERFLOW;
    } else {
      lsh_heads = (size_t *)malloc(heads_bytes);
      lsh_next = (size_t *)malloc(next_bytes);
      sig_pool = (uint64_t *)malloc(sig_bytes);
      cand_buf = (size_t *)malloc(doc_count * sizeof(size_t));
      cand_seen = (uint32_t *)calloc(doc_count, sizeof(uint32_t));
      cur_sig = (uint64_t *)malloc((size_t)num_perms * sizeof(uint64_t));
      if (lsh_heads == NULL || lsh_next == NULL || sig_pool == NULL ||
          cand_buf == NULL || cand_seen == NULL || cur_sig == NULL) {
        st = ET_DATA4_ERR_NOMEM;
      } else {
        for (size_t i = 0u; i < heads_count; ++i) lsh_heads[i] = SIZE_MAX;
        for (size_t i = 0u; i < next_count; ++i) lsh_next[i] = SIZE_MAX;
      }
    }
  }

  et_data4_pipeline_summary summary;
  memset(&summary, 0, sizeof(summary));
  summary.input_documents = (uint64_t)doc_count;
  summary.input_bytes = total_input_bytes;

  size_t kept_count = 0u;
  uint32_t epoch = 0u;

  for (size_t i = 0u; st == ET_DATA4_OK && i < doc_count; ++i) {
    const et_data4_document *doc = &documents[i];
    et_data4_doc_result *res = &staged_results[i];
    res->ordinal = (uint64_t)i;
    const size_t id_len = strlen(doc->source_id);
    memcpy(res->source_id, doc->source_id, id_len + 1u);
    res->byte_count = (uint64_t)doc->byte_count;
    sha256_raw(doc->bytes, doc->byte_count, res->sha256);

    /* Stage 1: Strict UTF-8 validation */
    uint64_t cp_count = 0u;
    uint64_t print_count = 0u;
    uint64_t alnum_count = 0u;
    const et_data4_status utf8_st = et_data4_validate_utf8(
        doc->bytes, doc->byte_count, &cp_count, &print_count, &alnum_count);
    if (utf8_st != ET_DATA4_OK) {
      res->disposition = ET_DATA4_DISP_REJECTED_UTF8;
      ++summary.rejected_utf8_documents;
      continue;
    }
    res->codepoint_count = cp_count;
    res->printable_count = print_count;
    res->alnum_count = alnum_count;

    /* Stage 2: Byte and codepoint length bounds */
    if (res->byte_count < config->min_bytes ||
        res->byte_count > config->max_bytes ||
        cp_count < config->min_codepoints ||
        cp_count > config->max_codepoints) {
      res->disposition = ET_DATA4_DISP_REJECTED_LENGTH;
      ++summary.rejected_length_documents;
      continue;
    }

    /* Stage 3: Printable and alphanumeric ratios */
    const double print_ratio =
        cp_count == 0u ? 0.0 : (double)print_count / (double)cp_count;
    const double alnum_ratio =
        cp_count == 0u ? 0.0 : (double)alnum_count / (double)cp_count;
    if (print_ratio < config->min_printable_ratio ||
        alnum_ratio < config->min_alnum_ratio) {
      res->disposition = ET_DATA4_DISP_REJECTED_QUALITY;
      ++summary.rejected_quality_documents;
      continue;
    }

    /* Stage 4: Duplicate-line and repeated-ngram ratios */
    uint64_t total_lines = 0u;
    uint64_t dup_lines = 0u;
    st = measure_duplicate_lines(doc->bytes, doc->byte_count, &total_lines,
                                 &dup_lines);
    if (st != ET_DATA4_OK) break;
    res->total_lines = total_lines;
    res->duplicate_lines = dup_lines;

    shingle_set rep_set = {NULL, 0u};
    uint64_t total_rep_ngrams = 0u;
    st = build_shingle_set(doc->bytes, doc->byte_count, config->rep_ngram_size,
                           &rep_set, &total_rep_ngrams);
    if (st != ET_DATA4_OK) break;
    const uint64_t repeated_ngrams =
        total_rep_ngrams > (uint64_t)rep_set.count
            ? total_rep_ngrams - (uint64_t)rep_set.count
            : 0u;
    res->total_rep_ngrams = total_rep_ngrams;
    res->repeated_ngrams = repeated_ngrams;

    const double dup_line_ratio =
        total_lines == 0u ? 0.0 : (double)dup_lines / (double)total_lines;
    const double rep_ngram_ratio =
        total_rep_ngrams == 0u
            ? 0.0
            : (double)repeated_ngrams / (double)total_rep_ngrams;
    if (dup_line_ratio > config->max_dup_line_ratio ||
        rep_ngram_ratio > config->max_repeated_ngram_ratio) {
      shingle_set_free(&rep_set);
      res->disposition = ET_DATA4_DISP_REJECTED_REPETITION;
      ++summary.rejected_repetition_documents;
      continue;
    }

    /* Stage 5: Exact SHA-256 deduplication (with exact byte verification) */
    const size_t exact_mask = exact_cap - 1u;
    size_t slot = (size_t)(load_u64_be(res->sha256) & (uint64_t)exact_mask);
    size_t exact_match = SIZE_MAX;
    while (exact_table[slot] != SIZE_MAX) {
      const size_t kidx = exact_table[slot];
      const kept_record *cand = &kept[kidx];
      if (memcmp(cand->sha256, res->sha256, ET_DATA4_SHA256_BYTES) == 0 &&
          cand->doc->byte_count == doc->byte_count &&
          (doc->byte_count == 0u ||
           memcmp(cand->doc->bytes, doc->bytes, doc->byte_count) == 0)) {
        exact_match = kidx;
        break;
      }
      slot = (slot + 1u) & exact_mask;
    }
    if (exact_match != SIZE_MAX) {
      shingle_set_free(&rep_set);
      const kept_record *canon = &kept[exact_match];
      res->disposition = ET_DATA4_DISP_DUPLICATE_EXACT;
      res->has_matched_canonical = 1u;
      res->matched_canonical_ordinal = (uint64_t)canon->input_ordinal;
      const size_t canon_id_len = strlen(canon->doc->source_id);
      memcpy(res->matched_source_id, canon->doc->source_id, canon_id_len + 1u);
      ++summary.duplicate_exact_documents;
      continue;
    }

    /* Stage 6: Seeded MinHash + LSH banding + exact Jaccard verification */
    shingle_set dedup_set = {NULL, 0u};
    int have_dedup_set = 0;
    if (near_on) {
      if (config->dedup_ngram_size == config->rep_ngram_size) {
        dedup_set = rep_set;
        rep_set.items = NULL;
        rep_set.count = 0u;
      } else {
        shingle_set_free(&rep_set);
        st = build_shingle_set(doc->bytes, doc->byte_count,
                               config->dedup_ngram_size, &dedup_set, NULL);
        if (st != ET_DATA4_OK) break;
      }
      have_dedup_set = 1;
      if (dedup_set.count != 0u) {
        compute_minhash_from_set(&dedup_set, num_perms, config->minhash_seed,
                                 cur_sig);
        ++epoch;
        if (epoch == 0u) {
          memset(cand_seen, 0, doc_count * sizeof(uint32_t));
          epoch = 1u;
        }
        size_t cand_count = 0u;
        const size_t lsh_mask = lsh_bucket_cap - 1u;
        for (uint32_t b = 0u; b < bands; ++b) {
          const uint64_t *band_sig = cur_sig + (size_t)b * (size_t)rows_per_band;
          const size_t bucket =
              (size_t)(lsh_band_hash(b, band_sig, rows_per_band) &
                       (uint64_t)lsh_mask);
          size_t node = lsh_heads[(size_t)b * lsh_bucket_cap + bucket];
          while (node != SIZE_MAX) {
            if (cand_seen[node] != epoch) {
              const uint64_t *cand_band =
                  kept[node].signature + (size_t)b * (size_t)rows_per_band;
              if (memcmp(band_sig, cand_band,
                         (size_t)rows_per_band * sizeof(uint64_t)) == 0) {
                cand_seen[node] = epoch;
                cand_buf[cand_count++] = node;
              }
            }
            node = lsh_next[(size_t)b * doc_count + node];
          }
        }
        if (cand_count > 1u)
          qsort(cand_buf, cand_count, sizeof(size_t), compare_size_t);

        size_t near_match = SIZE_MAX;
        uint64_t match_num = 0u;
        uint64_t match_den = 0u;
        for (size_t c = 0u; c < cand_count; ++c) {
          const size_t kidx = cand_buf[c];
          uint64_t jnum = 0u;
          uint64_t jden = 0u;
          exact_jaccard_sets(&dedup_set, &kept[kidx].shingles, &jnum, &jden);
          if (jden != 0u &&
              ((double)jnum / (double)jden) >=
                  config->near_dup_jaccard_threshold) {
            near_match = kidx;
            match_num = jnum;
            match_den = jden;
            break;
          }
        }
        if (near_match != SIZE_MAX) {
          shingle_set_free(&dedup_set);
          const kept_record *canon = &kept[near_match];
          res->disposition = ET_DATA4_DISP_DUPLICATE_NEAR;
          res->has_matched_canonical = 1u;
          res->matched_canonical_ordinal = (uint64_t)canon->input_ordinal;
          const size_t canon_id_len = strlen(canon->doc->source_id);
          memcpy(res->matched_source_id, canon->doc->source_id,
                 canon_id_len + 1u);
          res->jaccard_num = match_num;
          res->jaccard_den = match_den;
          ++summary.duplicate_near_documents;
          continue;
        }
      }
    } else {
      shingle_set_free(&rep_set);
    }

    /* Document is KEPT: insert into exact table and LSH band index. */
    res->disposition = ET_DATA4_DISP_KEPT;
    ++summary.kept_documents;
    summary.kept_bytes += res->byte_count;

    const size_t kidx = kept_count++;
    kept[kidx].input_ordinal = i;
    kept[kidx].doc = doc;
    memcpy(kept[kidx].sha256, res->sha256, ET_DATA4_SHA256_BYTES);
    exact_table[slot] = kidx;

    if (near_on && have_dedup_set) {
      kept[kidx].shingles = dedup_set;
      if (dedup_set.count != 0u) {
        uint64_t *dest_sig = sig_pool + kidx * (size_t)num_perms;
        memcpy(dest_sig, cur_sig, (size_t)num_perms * sizeof(uint64_t));
        kept[kidx].signature = dest_sig;
        const size_t lsh_mask = lsh_bucket_cap - 1u;
        for (uint32_t b = 0u; b < bands; ++b) {
          const uint64_t *band_sig = dest_sig + (size_t)b * (size_t)rows_per_band;
          const size_t bucket =
              (size_t)(lsh_band_hash(b, band_sig, rows_per_band) &
                       (uint64_t)lsh_mask);
          const size_t head_slot = (size_t)b * lsh_bucket_cap + bucket;
          lsh_next[(size_t)b * doc_count + kidx] = lsh_heads[head_slot];
          lsh_heads[head_slot] = kidx;
        }
      } else {
        kept[kidx].signature = NULL;
      }
    }
  }

  for (size_t k = 0u; k < kept_count; ++k) shingle_set_free(&kept[k].shingles);
  free(cur_sig);
  free(cand_seen);
  free(cand_buf);
  free(sig_pool);
  free(lsh_next);
  free(lsh_heads);
  free(exact_table);
  free(kept);

  if (st == ET_DATA4_OK) {
    memcpy(results_out, staged_results, results_bytes);
    *summary_out = summary;
  }
  free(staged_results);
  return st;
}

/* ---- Canonical Provenance TSV Report Formatting & Verification ---- */

static const char hex_digits[] = "0123456789abcdef";

static void format_digest_hex(const uint8_t digest[32], char out[65]) {
  for (size_t i = 0u; i < ET_DATA4_SHA256_BYTES; ++i) {
    out[2u * i] = hex_digits[digest[i] >> 4];
    out[2u * i + 1u] = hex_digits[digest[i] & 0x0Fu];
  }
  out[64] = '\0';
}

static uint64_t double_to_bits(double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

typedef struct str_builder {
  char *data;
  size_t len;
  size_t cap;
  et_data4_status err;
} str_builder;

static void sb_append(str_builder *sb, const char *s, size_t n) {
  if (sb->err != ET_DATA4_OK) return;
  size_t need = 0u;
  if (add_size_overflow(sb->len, n, &need) ||
      add_size_overflow(need, 1u, &need) || need > PTRDIFF_MAX) {
    sb->err = ET_DATA4_ERR_OVERFLOW;
    return;
  }
  if (need > sb->cap) {
    size_t next_cap = sb->cap == 0u ? 1024u : sb->cap;
    while (next_cap < need) {
      if (next_cap > SIZE_MAX / 2u) {
        next_cap = need;
        break;
      }
      next_cap *= 2u;
    }
    if (next_cap > PTRDIFF_MAX) next_cap = need;
    char *grown = (char *)realloc(sb->data, next_cap);
    if (grown == NULL) {
      sb->err = ET_DATA4_ERR_NOMEM;
      return;
    }
    sb->data = grown;
    sb->cap = next_cap;
  }
  if (n != 0u) memcpy(sb->data + sb->len, s, n);
  sb->len += n;
  sb->data[sb->len] = '\0';
}

static void sb_append_kv_u64(str_builder *sb, const char *key, uint64_t val) {
  char buf[128];
  const int n = snprintf(buf, sizeof(buf), "%s\t%" PRIu64 "\n", key, val);
  if (n < 0 || (size_t)n >= sizeof(buf)) {
    sb->err = ET_DATA4_ERR_FORMAT;
    return;
  }
  sb_append(sb, buf, (size_t)n);
}

static void sb_append_kv_u32(str_builder *sb, const char *key, uint32_t val) {
  sb_append_kv_u64(sb, key, (uint64_t)val);
}

static void sb_append_kv_f64_hex(str_builder *sb, const char *key, double val) {
  char buf[128];
  const int n =
      snprintf(buf, sizeof(buf), "%s\t%016" PRIx64 "\n", key, double_to_bits(val));
  if (n < 0 || (size_t)n >= sizeof(buf)) {
    sb->err = ET_DATA4_ERR_FORMAT;
    return;
  }
  sb_append(sb, buf, (size_t)n);
}

static et_data4_status validate_report_semantics(
    const et_data4_pipeline_config *config,
    const et_data4_pipeline_summary *summary,
    const et_data4_doc_result *results, size_t result_count) {
  if (!valid_config(config)) return ET_DATA4_ERR_FORMAT;
  if (summary->input_documents != (uint64_t)result_count)
    return ET_DATA4_ERR_FORMAT;

  uint64_t kept = 0u;
  uint64_t rej_utf8 = 0u;
  uint64_t rej_len = 0u;
  uint64_t rej_qual = 0u;
  uint64_t rej_rep = 0u;
  uint64_t dup_exact = 0u;
  uint64_t dup_near = 0u;
  uint64_t input_bytes = 0u;
  uint64_t kept_bytes = 0u;

  id_ref *id_refs = NULL;
  if (result_count != 0u) {
    size_t refs_bytes = 0u;
    if (mul_size_overflow(result_count, sizeof(id_ref), &refs_bytes) ||
        refs_bytes > PTRDIFF_MAX)
      return ET_DATA4_ERR_OVERFLOW;
    id_refs = (id_ref *)malloc(refs_bytes);
    if (id_refs == NULL) return ET_DATA4_ERR_NOMEM;
  }

  et_data4_status st = ET_DATA4_OK;
  for (size_t i = 0u; i < result_count; ++i) {
    const et_data4_doc_result *r = &results[i];
    size_t id_len = 0u;
    if (r->ordinal != (uint64_t)i ||
        !valid_source_id_cstr(r->source_id, &id_len) ||
        et_data4_disposition_name(r->disposition) == NULL) {
      st = ET_DATA4_ERR_FORMAT;
      break;
    }
    id_refs[i] = (id_ref){r->source_id, id_len};
    if (r->byte_count > UINT64_MAX - input_bytes) {
      st = ET_DATA4_ERR_OVERFLOW;
      break;
    }
    input_bytes += r->byte_count;

    if (r->disposition == ET_DATA4_DISP_REJECTED_UTF8) {
      if (r->codepoint_count != 0u) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
    }

    if (r->disposition == ET_DATA4_DISP_DUPLICATE_EXACT ||
        r->disposition == ET_DATA4_DISP_DUPLICATE_NEAR) {
      size_t match_len = 0u;
      if (r->has_matched_canonical != 1u ||
          !valid_source_id_cstr(r->matched_source_id, &match_len) ||
          r->matched_canonical_ordinal >= (uint64_t)i) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
      const et_data4_doc_result *canon = &results[r->matched_canonical_ordinal];
      if (canon->disposition != ET_DATA4_DISP_KEPT ||
          strcmp(canon->source_id, r->matched_source_id) != 0) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
      if (r->disposition == ET_DATA4_DISP_DUPLICATE_EXACT) {
        if (r->jaccard_num != 0u || r->jaccard_den != 0u ||
            canon->byte_count != r->byte_count ||
            memcmp(canon->sha256, r->sha256, ET_DATA4_SHA256_BYTES) != 0) {
          st = ET_DATA4_ERR_FORMAT;
          break;
        }
        ++dup_exact;
      } else {
        if (config->near_dup_enabled != 1u || r->jaccard_den == 0u ||
            r->jaccard_num > r->jaccard_den ||
            ((double)r->jaccard_num / (double)r->jaccard_den) <
                config->near_dup_jaccard_threshold) {
          st = ET_DATA4_ERR_FORMAT;
          break;
        }
        ++dup_near;
      }
    } else {
      if (r->has_matched_canonical != 0u || r->matched_source_id[0] != '\0' ||
          r->jaccard_num != 0u || r->jaccard_den != 0u) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
      switch (r->disposition) {
        case ET_DATA4_DISP_KEPT:
          ++kept;
          if (r->byte_count > UINT64_MAX - kept_bytes) {
            st = ET_DATA4_ERR_OVERFLOW;
            break;
          }
          kept_bytes += r->byte_count;
          break;
        case ET_DATA4_DISP_REJECTED_UTF8:
          ++rej_utf8;
          break;
        case ET_DATA4_DISP_REJECTED_LENGTH:
          ++rej_len;
          break;
        case ET_DATA4_DISP_REJECTED_QUALITY:
          ++rej_qual;
          break;
        case ET_DATA4_DISP_REJECTED_REPETITION:
          ++rej_rep;
          break;
        default:
          st = ET_DATA4_ERR_FORMAT;
          break;
      }
      if (st != ET_DATA4_OK) break;
    }
  }

  if (st == ET_DATA4_OK && result_count > 1u) {
    const et_data4_status dup_st = check_unique_source_ids(id_refs, result_count);
    if (dup_st == ET_DATA4_ERR_DUPLICATE_SOURCE_ID)
      st = ET_DATA4_ERR_DUPLICATE_SOURCE_ID;
    else if (dup_st != ET_DATA4_OK)
      st = dup_st;
  }
  free(id_refs);
  if (st != ET_DATA4_OK) return st;

  if (kept != summary->kept_documents ||
      rej_utf8 != summary->rejected_utf8_documents ||
      rej_len != summary->rejected_length_documents ||
      rej_qual != summary->rejected_quality_documents ||
      rej_rep != summary->rejected_repetition_documents ||
      dup_exact != summary->duplicate_exact_documents ||
      dup_near != summary->duplicate_near_documents ||
      input_bytes != summary->input_bytes ||
      kept_bytes != summary->kept_bytes)
    return ET_DATA4_ERR_FORMAT;

  return ET_DATA4_OK;
}

et_data4_status et_data4_report_format(
    const et_data4_pipeline_config *config,
    const et_data4_pipeline_summary *summary,
    const et_data4_doc_result *results, size_t result_count, char *out,
    size_t capacity, size_t *size_out,
    uint8_t report_sha256_out[ET_DATA4_SHA256_BYTES]) {
  if (!fp_environment_supported()) return ET_DATA4_ERR_FP_ENV;
  if (config == NULL || summary == NULL || size_out == NULL ||
      (out == NULL && capacity != 0u) ||
      (result_count != 0u && results == NULL))
    return ET_DATA4_ERR_ARGUMENT;

  size_t results_bytes = 0u;
  if (mul_size_overflow(result_count, sizeof(et_data4_doc_result),
                        &results_bytes) ||
      results_bytes > PTRDIFF_MAX)
    return ET_DATA4_ERR_OVERFLOW;

  if (!valid_span(config, _Alignof(et_data4_pipeline_config),
                  sizeof(*config)) ||
      !valid_span(summary, _Alignof(et_data4_pipeline_summary),
                  sizeof(*summary)) ||
      !valid_span(results, _Alignof(et_data4_doc_result), results_bytes) ||
      !valid_span(size_out, _Alignof(size_t), sizeof(*size_out)) ||
      (report_sha256_out != NULL &&
       !valid_span(report_sha256_out, 1u, ET_DATA4_SHA256_BYTES)))
    return ET_DATA4_ERR_ALIGNMENT;

  if (spans_overlap(size_out, sizeof(*size_out), config, sizeof(*config)) ||
      spans_overlap(size_out, sizeof(*size_out), summary, sizeof(*summary)) ||
      spans_overlap(size_out, sizeof(*size_out), results, results_bytes) ||
      (report_sha256_out != NULL &&
       (spans_overlap(report_sha256_out, ET_DATA4_SHA256_BYTES, config,
                      sizeof(*config)) ||
        spans_overlap(report_sha256_out, ET_DATA4_SHA256_BYTES, summary,
                      sizeof(*summary)) ||
        spans_overlap(report_sha256_out, ET_DATA4_SHA256_BYTES, results,
                      results_bytes) ||
        spans_overlap(size_out, sizeof(*size_out), report_sha256_out,
                      ET_DATA4_SHA256_BYTES))))
    return ET_DATA4_ERR_ALIASING;

  if (out != NULL) {
    if (!valid_span(out, 1u, capacity)) return ET_DATA4_ERR_BOUNDS;
    if (spans_overlap(out, capacity, config, sizeof(*config)) ||
        spans_overlap(out, capacity, summary, sizeof(*summary)) ||
        spans_overlap(out, capacity, results, results_bytes) ||
        spans_overlap(out, capacity, size_out, sizeof(*size_out)) ||
        (report_sha256_out != NULL &&
         spans_overlap(out, capacity, report_sha256_out,
                       ET_DATA4_SHA256_BYTES)))
      return ET_DATA4_ERR_ALIASING;
  }

  et_data4_status st =
      validate_report_semantics(config, summary, results, result_count);
  if (st != ET_DATA4_OK) return st;

  str_builder sb = {NULL, 0u, 0u, ET_DATA4_OK};
  const char header_prefix[] =
      "format\t" ET_DATA4_REPORT_FORMAT_ID "\nversion\t1\t0\n";
  sb_append(&sb, header_prefix, sizeof(header_prefix) - 1u);
  sb_append_kv_u64(&sb, "min-bytes", config->min_bytes);
  sb_append_kv_u64(&sb, "max-bytes", config->max_bytes);
  sb_append_kv_u64(&sb, "min-codepoints", config->min_codepoints);
  sb_append_kv_u64(&sb, "max-codepoints", config->max_codepoints);
  sb_append_kv_f64_hex(&sb, "min-printable-ratio-hex",
                       config->min_printable_ratio);
  sb_append_kv_f64_hex(&sb, "min-alnum-ratio-hex", config->min_alnum_ratio);
  sb_append_kv_f64_hex(&sb, "max-dup-line-ratio-hex",
                       config->max_dup_line_ratio);
  sb_append_kv_f64_hex(&sb, "max-repeated-ngram-ratio-hex",
                       config->max_repeated_ngram_ratio);
  sb_append_kv_u32(&sb, "rep-ngram-size", config->rep_ngram_size);
  sb_append_kv_u32(&sb, "near-dup-enabled", config->near_dup_enabled);
  sb_append_kv_u32(&sb, "dedup-ngram-size", config->dedup_ngram_size);
  sb_append_kv_u32(&sb, "minhash-num-perms", config->minhash_num_perms);
  sb_append_kv_u32(&sb, "minhash-bands", config->minhash_bands);
  sb_append_kv_u64(&sb, "minhash-seed", config->minhash_seed);
  sb_append_kv_f64_hex(&sb, "near-dup-jaccard-threshold-hex",
                       config->near_dup_jaccard_threshold);
  sb_append_kv_u64(&sb, "input-documents", summary->input_documents);
  sb_append_kv_u64(&sb, "kept-documents", summary->kept_documents);
  sb_append_kv_u64(&sb, "rejected-utf8-documents",
                   summary->rejected_utf8_documents);
  sb_append_kv_u64(&sb, "rejected-length-documents",
                   summary->rejected_length_documents);
  sb_append_kv_u64(&sb, "rejected-quality-documents",
                   summary->rejected_quality_documents);
  sb_append_kv_u64(&sb, "rejected-repetition-documents",
                   summary->rejected_repetition_documents);
  sb_append_kv_u64(&sb, "duplicate-exact-documents",
                   summary->duplicate_exact_documents);
  sb_append_kv_u64(&sb, "duplicate-near-documents",
                   summary->duplicate_near_documents);
  sb_append_kv_u64(&sb, "input-bytes", summary->input_bytes);
  sb_append_kv_u64(&sb, "kept-bytes", summary->kept_bytes);

  for (size_t i = 0u; sb.err == ET_DATA4_OK && i < result_count; ++i) {
    const et_data4_doc_result *r = &results[i];
    char digest_hex[65];
    format_digest_hex(r->sha256, digest_hex);
    const char *disp_name = et_data4_disposition_name(r->disposition);
    const char *matched =
        r->has_matched_canonical ? r->matched_source_id : "";
    char row[512];
    const int n = snprintf(
        row, sizeof(row),
        "document\t%" PRIu64 "\t%s\t%" PRIu64 "\t%" PRIu64 "\t%s\t%s\t%s\t%" PRIu64
        "\t%" PRIu64 "\n",
        r->ordinal, r->source_id, r->byte_count, r->codepoint_count, digest_hex,
        disp_name, matched, r->jaccard_num, r->jaccard_den);
    if (n < 0 || (size_t)n >= sizeof(row)) {
      sb.err = ET_DATA4_ERR_FORMAT;
      break;
    }
    sb_append(&sb, row, (size_t)n);
  }

  const char alg_line[] = "checksum-algorithm\tsha256\n";
  sb_append(&sb, alg_line, sizeof(alg_line) - 1u);

  uint8_t digest[ET_DATA4_SHA256_BYTES];
  if (sb.err == ET_DATA4_OK) {
    sha256_raw((const uint8_t *)sb.data, sb.len, digest);
    char digest_hex[65];
    format_digest_hex(digest, digest_hex);
    char csum_line[96];
    const int n =
        snprintf(csum_line, sizeof(csum_line), "checksum\t%s\n", digest_hex);
    if (n < 0 || (size_t)n >= sizeof(csum_line))
      sb.err = ET_DATA4_ERR_FORMAT;
    else
      sb_append(&sb, csum_line, (size_t)n);
  }

  if (sb.err != ET_DATA4_OK) {
    free(sb.data);
    return sb.err;
  }

  *size_out = sb.len;
  if (report_sha256_out != NULL)
    memcpy(report_sha256_out, digest, ET_DATA4_SHA256_BYTES);

  if (out == NULL) {
    free(sb.data);
    return ET_DATA4_OK;
  }
  if (capacity <= sb.len) {
    free(sb.data);
    return ET_DATA4_ERR_BOUNDS;
  }
  memcpy(out, sb.data, sb.len + 1u);
  free(sb.data);
  return ET_DATA4_OK;
}

/* ---- Strict Canonical TSV Parser & Verifier ---- */

typedef struct text_span {
  const char *data;
  size_t size;
} text_span;

static int take_line(text_span *input, text_span *line) {
  const char *nl = (const char *)memchr(input->data, '\n', input->size);
  if (nl == NULL) return 0;
  const size_t len = (size_t)(nl - input->data);
  *line = (text_span){input->data, len};
  input->data = nl + 1;
  input->size -= len + 1u;
  return 1;
}

static int next_tab_field(text_span *line, text_span *field, int *more) {
  const char *tab = (const char *)memchr(line->data, '\t', line->size);
  if (tab == NULL) {
    *field = *line;
    line->data += line->size;
    line->size = 0u;
    *more = 0;
    return 1;
  }
  const size_t len = (size_t)(tab - line->data);
  *field = (text_span){line->data, len};
  line->data = tab + 1;
  line->size -= len + 1u;
  *more = 1;
  return 1;
}

static int span_equals(text_span s, const char *lit) {
  const size_t n = strlen(lit);
  return s.size == n && (n == 0u || memcmp(s.data, lit, n) == 0);
}

static int take_kv_line(text_span *input, const char *key, text_span *val) {
  text_span line;
  if (!take_line(input, &line)) return 0;
  text_span k;
  int more = 0;
  next_tab_field(&line, &k, &more);
  if (!more || !span_equals(k, key)) return 0;
  text_span v;
  next_tab_field(&line, &v, &more);
  if (more) return 0;
  *val = v;
  return 1;
}

static int parse_u64_strict(text_span s, uint64_t *out) {
  if (s.size == 0u || s.size > 20u || (s.size > 1u && s.data[0] == '0'))
    return 0;
  uint64_t acc = 0u;
  for (size_t i = 0u; i < s.size; ++i) {
    const unsigned char c = (unsigned char)s.data[i];
    if (c < '0' || c > '9') return 0;
    const uint64_t digit = (uint64_t)(c - '0');
    if (acc > (UINT64_MAX - digit) / 10u) return 0;
    acc = acc * 10u + digit;
  }
  *out = acc;
  return 1;
}

static int parse_u32_strict(text_span s, uint32_t *out) {
  uint64_t v = 0u;
  if (!parse_u64_strict(s, &v) || v > UINT32_MAX) return 0;
  *out = (uint32_t)v;
  return 1;
}

static int hex_nibble(unsigned char c) {
  if (c >= '0' && c <= '9') return (int)(c - '0');
  if (c >= 'a' && c <= 'f') return (int)(c - 'a' + 10);
  return -1;
}

static int parse_f64_hex_strict(text_span s, double *out) {
  if (s.size != 16u) return 0;
  uint64_t bits = 0u;
  for (size_t i = 0u; i < 16u; ++i) {
    const int d = hex_nibble((unsigned char)s.data[i]);
    if (d < 0) return 0;
    bits = (bits << 4) | (uint64_t)d;
  }
  if ((bits & UINT64_C(0x7ff0000000000000)) == UINT64_C(0x7ff0000000000000))
    return 0;
  double v = 0.0;
  memcpy(&v, &bits, sizeof(v));
  *out = v;
  return 1;
}

static int parse_sha256_hex_strict(text_span s, uint8_t out[32]) {
  if (s.size != 64u) return 0;
  for (size_t i = 0u; i < 32u; ++i) {
    const int hi = hex_nibble((unsigned char)s.data[2u * i]);
    const int lo = hex_nibble((unsigned char)s.data[2u * i + 1u]);
    if (hi < 0 || lo < 0) return 0;
    out[i] = (uint8_t)((hi << 4) | lo);
  }
  return 1;
}

static int parse_disposition_strict(text_span s, et_data4_disposition *out) {
  if (span_equals(s, "kept")) {
    *out = ET_DATA4_DISP_KEPT;
    return 1;
  }
  if (span_equals(s, "rejected_utf8")) {
    *out = ET_DATA4_DISP_REJECTED_UTF8;
    return 1;
  }
  if (span_equals(s, "rejected_length")) {
    *out = ET_DATA4_DISP_REJECTED_LENGTH;
    return 1;
  }
  if (span_equals(s, "rejected_quality")) {
    *out = ET_DATA4_DISP_REJECTED_QUALITY;
    return 1;
  }
  if (span_equals(s, "rejected_repetition")) {
    *out = ET_DATA4_DISP_REJECTED_REPETITION;
    return 1;
  }
  if (span_equals(s, "duplicate_exact")) {
    *out = ET_DATA4_DISP_DUPLICATE_EXACT;
    return 1;
  }
  if (span_equals(s, "duplicate_near")) {
    *out = ET_DATA4_DISP_DUPLICATE_NEAR;
    return 1;
  }
  return 0;
}

static int all_canonical_tsv_ascii(const char *bytes, size_t size) {
  for (size_t i = 0u; i < size; ++i) {
    const unsigned char c = (unsigned char)bytes[i];
    if (c == '\t' || c == '\n') continue;
    if (c >= 0x20u && c <= 0x7Eu) continue;
    return 0;
  }
  return 1;
}

et_data4_status et_data4_report_verify(
    const char *report_bytes, size_t report_size,
    et_data4_pipeline_config *config_out,
    et_data4_pipeline_summary *summary_out, et_data4_doc_result *results_out,
    size_t results_capacity) {
  if (!fp_environment_supported()) return ET_DATA4_ERR_FP_ENV;
  if (report_bytes == NULL || report_size == 0u ||
      (results_out == NULL && results_capacity != 0u))
    return ET_DATA4_ERR_ARGUMENT;
  if (report_size > PTRDIFF_MAX || (uint64_t)report_size > UINT64_MAX / 8u)
    return ET_DATA4_ERR_OVERFLOW;
  if (!valid_span(report_bytes, 1u, report_size)) return ET_DATA4_ERR_BOUNDS;
  if ((config_out != NULL &&
       !valid_span(config_out, _Alignof(et_data4_pipeline_config),
                   sizeof(*config_out))) ||
      (summary_out != NULL &&
       !valid_span(summary_out, _Alignof(et_data4_pipeline_summary),
                   sizeof(*summary_out))))
    return ET_DATA4_ERR_ALIGNMENT;
  if ((config_out != NULL &&
       spans_overlap(report_bytes, report_size, config_out,
                     sizeof(*config_out))) ||
      (summary_out != NULL &&
       spans_overlap(report_bytes, report_size, summary_out,
                     sizeof(*summary_out))) ||
      (config_out != NULL && summary_out != NULL &&
       spans_overlap(config_out, sizeof(*config_out), summary_out,
                     sizeof(*summary_out))))
    return ET_DATA4_ERR_ALIASING;

  if (!all_canonical_tsv_ascii(report_bytes, report_size))
    return ET_DATA4_ERR_FORMAT;

  text_span input = {report_bytes, report_size};
  text_span val;

  if (!take_kv_line(&input, "format", &val) ||
      !span_equals(val, ET_DATA4_REPORT_FORMAT_ID))
    return ET_DATA4_ERR_FORMAT;

  /* Parse version\t<major>\t<minor>\n */
  text_span ver_line;
  if (!take_line(&input, &ver_line)) return ET_DATA4_ERR_FORMAT;
  text_span vf0, vf1, vf2;
  int more = 0;
  next_tab_field(&ver_line, &vf0, &more);
  if (!more || !span_equals(vf0, "version")) return ET_DATA4_ERR_FORMAT;
  next_tab_field(&ver_line, &vf1, &more);
  if (!more) return ET_DATA4_ERR_FORMAT;
  next_tab_field(&ver_line, &vf2, &more);
  if (more) return ET_DATA4_ERR_FORMAT;
  uint32_t ver_major = 0u;
  uint32_t ver_minor = 0u;
  if (!parse_u32_strict(vf1, &ver_major) || !parse_u32_strict(vf2, &ver_minor))
    return ET_DATA4_ERR_FORMAT;
  if (ver_major != ET_DATA4_REPORT_VERSION_MAJOR ||
      ver_minor != ET_DATA4_REPORT_VERSION_MINOR)
    return ET_DATA4_ERR_VERSION;

  et_data4_pipeline_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  et_data4_pipeline_summary sum;
  memset(&sum, 0, sizeof(sum));

  if (!take_kv_line(&input, "min-bytes", &val) ||
      !parse_u64_strict(val, &cfg.min_bytes) ||
      !take_kv_line(&input, "max-bytes", &val) ||
      !parse_u64_strict(val, &cfg.max_bytes) ||
      !take_kv_line(&input, "min-codepoints", &val) ||
      !parse_u64_strict(val, &cfg.min_codepoints) ||
      !take_kv_line(&input, "max-codepoints", &val) ||
      !parse_u64_strict(val, &cfg.max_codepoints) ||
      !take_kv_line(&input, "min-printable-ratio-hex", &val) ||
      !parse_f64_hex_strict(val, &cfg.min_printable_ratio) ||
      !take_kv_line(&input, "min-alnum-ratio-hex", &val) ||
      !parse_f64_hex_strict(val, &cfg.min_alnum_ratio) ||
      !take_kv_line(&input, "max-dup-line-ratio-hex", &val) ||
      !parse_f64_hex_strict(val, &cfg.max_dup_line_ratio) ||
      !take_kv_line(&input, "max-repeated-ngram-ratio-hex", &val) ||
      !parse_f64_hex_strict(val, &cfg.max_repeated_ngram_ratio) ||
      !take_kv_line(&input, "rep-ngram-size", &val) ||
      !parse_u32_strict(val, &cfg.rep_ngram_size) ||
      !take_kv_line(&input, "near-dup-enabled", &val) ||
      !parse_u32_strict(val, &cfg.near_dup_enabled) ||
      !take_kv_line(&input, "dedup-ngram-size", &val) ||
      !parse_u32_strict(val, &cfg.dedup_ngram_size) ||
      !take_kv_line(&input, "minhash-num-perms", &val) ||
      !parse_u32_strict(val, &cfg.minhash_num_perms) ||
      !take_kv_line(&input, "minhash-bands", &val) ||
      !parse_u32_strict(val, &cfg.minhash_bands) ||
      !take_kv_line(&input, "minhash-seed", &val) ||
      !parse_u64_strict(val, &cfg.minhash_seed) ||
      !take_kv_line(&input, "near-dup-jaccard-threshold-hex", &val) ||
      !parse_f64_hex_strict(val, &cfg.near_dup_jaccard_threshold) ||
      !take_kv_line(&input, "input-documents", &val) ||
      !parse_u64_strict(val, &sum.input_documents) ||
      !take_kv_line(&input, "kept-documents", &val) ||
      !parse_u64_strict(val, &sum.kept_documents) ||
      !take_kv_line(&input, "rejected-utf8-documents", &val) ||
      !parse_u64_strict(val, &sum.rejected_utf8_documents) ||
      !take_kv_line(&input, "rejected-length-documents", &val) ||
      !parse_u64_strict(val, &sum.rejected_length_documents) ||
      !take_kv_line(&input, "rejected-quality-documents", &val) ||
      !parse_u64_strict(val, &sum.rejected_quality_documents) ||
      !take_kv_line(&input, "rejected-repetition-documents", &val) ||
      !parse_u64_strict(val, &sum.rejected_repetition_documents) ||
      !take_kv_line(&input, "duplicate-exact-documents", &val) ||
      !parse_u64_strict(val, &sum.duplicate_exact_documents) ||
      !take_kv_line(&input, "duplicate-near-documents", &val) ||
      !parse_u64_strict(val, &sum.duplicate_near_documents) ||
      !take_kv_line(&input, "input-bytes", &val) ||
      !parse_u64_strict(val, &sum.input_bytes) ||
      !take_kv_line(&input, "kept-bytes", &val) ||
      !parse_u64_strict(val, &sum.kept_bytes))
    return ET_DATA4_ERR_FORMAT;

  /* Sanity cap: each document row takes at least 80 bytes in the report. */
  if (sum.input_documents > (uint64_t)(report_size / 64u) + 1u ||
      sum.input_documents > (uint64_t)(PTRDIFF_MAX / sizeof(et_data4_doc_result)))
    return ET_DATA4_ERR_FORMAT;

  const size_t doc_count = (size_t)sum.input_documents;
  if (results_out != NULL && results_capacity < doc_count)
    return ET_DATA4_ERR_BOUNDS;
  if (results_out != NULL && doc_count != 0u) {
    const size_t rbytes = doc_count * sizeof(et_data4_doc_result);
    if (!valid_span(results_out, _Alignof(et_data4_doc_result), rbytes))
      return ET_DATA4_ERR_ALIGNMENT;
    if (spans_overlap(report_bytes, report_size, results_out, rbytes) ||
        (config_out != NULL &&
         spans_overlap(config_out, sizeof(*config_out), results_out, rbytes)) ||
        (summary_out != NULL &&
         spans_overlap(summary_out, sizeof(*summary_out), results_out, rbytes)))
      return ET_DATA4_ERR_ALIASING;
  }

  et_data4_doc_result *parsed = NULL;
  if (doc_count != 0u) {
    parsed = (et_data4_doc_result *)calloc(doc_count, sizeof(et_data4_doc_result));
    if (parsed == NULL) return ET_DATA4_ERR_NOMEM;
  }

  et_data4_status st = ET_DATA4_OK;
  for (size_t i = 0u; i < doc_count; ++i) {
    text_span row_line;
    if (!take_line(&input, &row_line)) {
      st = ET_DATA4_ERR_FORMAT;
      break;
    }
    text_span cols[10];
    int has_more = 1;
    for (int c = 0; c < 10; ++c) {
      if (!has_more) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
      next_tab_field(&row_line, &cols[c], &has_more);
    }
    if (st != ET_DATA4_OK || has_more || !span_equals(cols[0], "document")) {
      st = ET_DATA4_ERR_FORMAT;
      break;
    }
    et_data4_doc_result *r = &parsed[i];
    if (!parse_u64_strict(cols[1], &r->ordinal) ||
        !valid_source_id_span(cols[2].data, cols[2].size) ||
        !parse_u64_strict(cols[3], &r->byte_count) ||
        !parse_u64_strict(cols[4], &r->codepoint_count) ||
        !parse_sha256_hex_strict(cols[5], r->sha256) ||
        !parse_disposition_strict(cols[6], &r->disposition) ||
        !parse_u64_strict(cols[8], &r->jaccard_num) ||
        !parse_u64_strict(cols[9], &r->jaccard_den)) {
      st = ET_DATA4_ERR_FORMAT;
      break;
    }
    memcpy(r->source_id, cols[2].data, cols[2].size);
    r->source_id[cols[2].size] = '\0';

    if (cols[7].size != 0u) {
      if (!valid_source_id_span(cols[7].data, cols[7].size)) {
        st = ET_DATA4_ERR_FORMAT;
        break;
      }
      r->has_matched_canonical = 1u;
      memcpy(r->matched_source_id, cols[7].data, cols[7].size);
      r->matched_source_id[cols[7].size] = '\0';
      /* Resolve canonical ordinal among earlier parsed rows. */
      r->matched_canonical_ordinal = UINT64_MAX;
      for (size_t prev = 0u; prev < i; ++prev) {
        if (strcmp(parsed[prev].source_id, r->matched_source_id) == 0) {
          r->matched_canonical_ordinal = (uint64_t)prev;
          break;
        }
      }
    }
  }

  if (st == ET_DATA4_OK) {
    if (!take_kv_line(&input, "checksum-algorithm", &val) ||
        !span_equals(val, "sha256")) {
      st = ET_DATA4_ERR_FORMAT;
    }
  }

  size_t payload_len = 0u;
  if (st == ET_DATA4_OK) {
    payload_len = (size_t)(input.data - report_bytes);
    if (!take_kv_line(&input, "checksum", &val) || input.size != 0u ||
        !parse_sha256_hex_strict(val, sum.report_sha256)) {
      st = ET_DATA4_ERR_FORMAT;
    }
  }

  /* Verify SHA-256 checksum BEFORE semantic cross-checks so any tamper of a
   * well-formed field deterministically reports ET_DATA4_ERR_CHECKSUM. */
  if (st == ET_DATA4_OK) {
    uint8_t expected[ET_DATA4_SHA256_BYTES];
    sha256_raw((const uint8_t *)report_bytes, payload_len, expected);
    if (memcmp(expected, sum.report_sha256, ET_DATA4_SHA256_BYTES) != 0)
      st = ET_DATA4_ERR_CHECKSUM;
  }

  if (st == ET_DATA4_OK) {
    st = validate_report_semantics(&cfg, &sum, parsed, doc_count);
    if (st == ET_DATA4_ERR_DUPLICATE_SOURCE_ID) st = ET_DATA4_ERR_FORMAT;
  }

  if (st == ET_DATA4_OK) {
    if (config_out != NULL) *config_out = cfg;
    if (summary_out != NULL) *summary_out = sum;
    if (results_out != NULL && doc_count != 0u)
      memcpy(results_out, parsed, doc_count * sizeof(et_data4_doc_result));
  }
  free(parsed);
  return st;
}
