#define _GNU_SOURCE

#include "eshkol_transformer/mod5_adamw_abi.h"

#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if !defined(__linux__) || !defined(O_CLOEXEC) || !defined(O_DIRECTORY) ||     \
    !defined(O_NOFOLLOW) || !defined(O_NONBLOCK)
#error "MOD5 AdamW training-state I/O requires Linux O_NOFOLLOW/O_DIRECTORY"
#endif

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4u && sizeof(double) == 8u,
               "MOD5 AdamW requires IEEE-754 binary32 and binary64");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 && FLT_MAX_EXP == 128 &&
                   FLT_EVAL_METHOD == 0,
               "MOD5 AdamW requires standard binary32 evaluation");

#define ET_MOD5_PI 3.14159265358979323846
#define ET_MOD5_TEMP_RANDOM_BYTES 16u
#define ET_MOD5_TEMP_ATTEMPTS 128u
#define ET_MOD5_TEMP_NAME_BYTES 45u
#define ET_MOD5_D1_SHARD_MAGIC "ESHKTSH1"
#define ET_MOD5_D1_SHARD_FIXED 80u
#define ET_MOD5_D1_MAX_TOKENS UINT64_C(1048576)

/* Maximum snapshot bytes: 224 + 128 + 12 * 16777216 + 32 = 201326976 bytes. */
#define ET_MOD5_SNAPSHOT_MAX_BYTES                                             \
  ((uint64_t)ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES +                             \
   (uint64_t)ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES +                            \
   UINT64_C(12) * ET_MOD5_ADAMW_MAX_PARAMS +                                   \
   (uint64_t)ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES)

struct et_mod5_adamw_state {
  uint64_t param_count;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float max_grad_norm;
  et_mod5_lr_schedule_config_v1 schedule;
  uint8_t model_config_hash[32];
  uint8_t full_config_hash[32];
  uint32_t tokenizer_id_len;
  char tokenizer_id[ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES + 1u];
  uint64_t step;
  uint64_t cumulative_tokens;
  et_mod5_dataset_cursor_v1 cursor;
  uint64_t rng_state[4];
  float *params;
  float *m;
  float *v;
};

/* --- Self-contained SHA-256 (zero misaligned pointer casts) --- */

typedef struct et_sha256_ctx {
  uint32_t h[8];
  uint64_t length;
  uint8_t block[64];
  size_t used;
} et_sha256_ctx;

static uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32u - n));
}

static uint64_t rotl64(uint64_t x, unsigned n) {
  return (x << n) | (x >> (64u - n));
}

static void sha256_block(et_sha256_ctx *ctx, const uint8_t block[64]) {
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
  a = ctx->h[0];
  b = ctx->h[1];
  c = ctx->h[2];
  d = ctx->h[3];
  e = ctx->h[4];
  f = ctx->h[5];
  g = ctx->h[6];
  h = ctx->h[7];
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
  ctx->h[0] += a;
  ctx->h[1] += b;
  ctx->h[2] += c;
  ctx->h[3] += d;
  ctx->h[4] += e;
  ctx->h[5] += f;
  ctx->h[6] += g;
  ctx->h[7] += h;
}

static void sha256_init(et_sha256_ctx *ctx) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  memcpy(ctx->h, initial, sizeof(initial));
  ctx->length = 0u;
  ctx->used = 0u;
}

static void sha256_update(et_sha256_ctx *ctx, const uint8_t *data, size_t len) {
  ctx->length += (uint64_t)len;
  while (len != 0u) {
    size_t take = 64u - ctx->used;
    if (take > len) {
      take = len;
    }
    memcpy(ctx->block + ctx->used, data, take);
    ctx->used += take;
    data += take;
    len -= take;
    if (ctx->used == 64u) {
      sha256_block(ctx, ctx->block);
      ctx->used = 0u;
    }
  }
}

static void sha256_final(et_sha256_ctx *ctx, uint8_t digest[32]) {
  const uint64_t bits = ctx->length * UINT64_C(8);
  size_t i;
  ctx->block[ctx->used++] = 0x80u;
  if (ctx->used > 56u) {
    memset(ctx->block + ctx->used, 0, 64u - ctx->used);
    sha256_block(ctx, ctx->block);
    ctx->used = 0u;
  }
  memset(ctx->block + ctx->used, 0, 56u - ctx->used);
  for (i = 0u; i < 8u; ++i) {
    ctx->block[63u - i] = (uint8_t)(bits >> (8u * i));
  }
  sha256_block(ctx, ctx->block);
  for (i = 0u; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(ctx->h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(ctx->h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(ctx->h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)ctx->h[i];
  }
}

static int ct_equal_32(const uint8_t a[32], const uint8_t b[32]) {
  uint8_t diff = 0u;
  for (size_t i = 0u; i < 32u; ++i) {
    diff |= (uint8_t)(a[i] ^ b[i]);
  }
  return diff == 0u;
}

static void digest_to_hex(const uint8_t digest[32], char out_hex[65]) {
  static const char hex[] = "0123456789abcdef";
  for (size_t i = 0u; i < 32u; ++i) {
    out_hex[i * 2u] = hex[digest[i] >> 4u];
    out_hex[i * 2u + 1u] = hex[digest[i] & 0x0fu];
  }
  out_hex[64] = '\0';
}

/* --- Strict little-endian byte marshaling (zero misaligned pointer UB) --- */

static uint32_t f32_to_bits(float v) {
  uint32_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static float bits_to_f32(uint32_t bits) {
  float v = 0.0f;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static uint64_t f64_to_bits(double v) {
  uint64_t bits = 0u;
  memcpy(&bits, &v, sizeof(bits));
  return bits;
}

static double bits_to_f64(uint64_t bits) {
  double v = 0.0;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static void put_u16_le(uint8_t *dst, uint16_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
}

static void put_u32_le(uint8_t *dst, uint32_t v) {
  dst[0] = (uint8_t)(v & 0xffu);
  dst[1] = (uint8_t)((v >> 8u) & 0xffu);
  dst[2] = (uint8_t)((v >> 16u) & 0xffu);
  dst[3] = (uint8_t)((v >> 24u) & 0xffu);
}

static void put_u64_le(uint8_t *dst, uint64_t v) {
  for (size_t i = 0u; i < 8u; ++i) {
    dst[i] = (uint8_t)((v >> (8u * i)) & 0xffu);
  }
}

static void put_f32_le(uint8_t *dst, float v) {
  put_u32_le(dst, f32_to_bits(v));
}

static void put_f64_le(uint8_t *dst, double v) {
  put_u64_le(dst, f64_to_bits(v));
}

static uint16_t get_u16_le(const uint8_t *src) {
  return (uint16_t)((uint16_t)src[0] | ((uint16_t)src[1] << 8u));
}

static uint32_t get_u32_le(const uint8_t *src) {
  return ((uint32_t)src[0]) | ((uint32_t)src[1] << 8u) |
         ((uint32_t)src[2] << 16u) | ((uint32_t)src[3] << 24u);
}

static uint64_t get_u64_le(const uint8_t *src) {
  uint64_t v = 0u;
  for (size_t i = 0u; i < 8u; ++i) {
    v |= ((uint64_t)src[i] << (8u * i));
  }
  return v;
}

static float get_f32_le(const uint8_t *src) {
  return bits_to_f32(get_u32_le(src));
}

static double get_f64_le(const uint8_t *src) {
  return bits_to_f64(get_u64_le(src));
}

/* --- FP determinism and validation helpers --- */

static et_mod5_adamw_status check_fp_env(void) {
  if (fegetround() != FE_TONEAREST) {
    return ET_MOD5_ADAMW_ERR_DETERMINISM;
  }
#if defined(__x86_64__)
  {
    const unsigned int csr = _mm_getcsr();
    /* Require round-to-nearest-even (bits 14:13 == 0), FTZ (bit 15) == 0,
     * and DAZ (bit 6) == 0. */
    if ((csr & 0x6000u) != 0u || (csr & 0x8040u) != 0u) {
      return ET_MOD5_ADAMW_ERR_DETERMINISM;
    }
  }
#endif
  return ET_MOD5_ADAMW_OK;
}

static int is_finite_f32(float v) { return isfinite(v) != 0; }
static int is_finite_f64(double v) { return isfinite(v) != 0; }

/* Canonicalize -0.0f to +0.0f for deterministic hashing/serialization. */
static float canonical_f32(float v) { return v == 0.0f ? 0.0f : v; }
static double canonical_f64(double v) { return v == 0.0 ? 0.0 : v; }

static et_mod5_adamw_status
validate_schedule(const et_mod5_lr_schedule_config_v1 *sched) {
  if (sched == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (sched->reserved != 0u) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (!is_finite_f64(sched->base_lr) || !is_finite_f64(sched->min_lr)) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  if (sched->base_lr < 0.0 || sched->min_lr < 0.0 ||
      sched->min_lr > sched->base_lr) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  switch (sched->kind) {
  case ET_MOD5_LR_CONSTANT:
    return ET_MOD5_ADAMW_OK;
  case ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY:
  case ET_MOD5_LR_LINEAR_WARMUP_COSINE_DECAY:
    if (sched->total_steps == 0u || sched->warmup_steps > sched->total_steps) {
      return ET_MOD5_ADAMW_ERR_INVALID_ARG;
    }
    return ET_MOD5_ADAMW_OK;
  default:
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
}

static et_mod5_adamw_status validate_tokenizer_id(const char *tok,
                                                  uint32_t *out_len) {
  size_t len = 0u;
  if (tok == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  while (len <= ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES && tok[len] != '\0') {
    ++len;
  }
  if (len == 0u || len > ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (out_len != NULL) {
    *out_len = (uint32_t)len;
  }
  return ET_MOD5_ADAMW_OK;
}

static et_mod5_adamw_status
validate_config(const et_mod5_adamw_config_v1 *cfg, uint32_t *out_tok_len) {
  et_mod5_adamw_status st;
  if (cfg == NULL || cfg->reserved != 0u) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (cfg->param_count == 0u || cfg->param_count > ET_MOD5_ADAMW_MAX_PARAMS) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (!is_finite_f32(cfg->beta1) || !is_finite_f32(cfg->beta2) ||
      !is_finite_f32(cfg->eps) || !is_finite_f32(cfg->weight_decay) ||
      !is_finite_f32(cfg->max_grad_norm)) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  if (cfg->beta1 < 0.0f || cfg->beta1 >= 1.0f || cfg->beta2 < 0.0f ||
      cfg->beta2 >= 1.0f || cfg->eps <= 0.0f || cfg->weight_decay < 0.0f ||
      cfg->max_grad_norm < 0.0f) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = validate_schedule(&cfg->schedule);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  return validate_tokenizer_id(cfg->tokenizer_id, out_tok_len);
}

static void compute_full_config_hash_parts(
    uint64_t param_count, float beta1, float beta2, float eps,
    float weight_decay, float max_grad_norm,
    const et_mod5_lr_schedule_config_v1 *sched,
    const uint8_t model_config_hash[32], const char *tokenizer_id,
    uint32_t tokenizer_id_len, uint8_t out_hash[32]) {
  static const uint8_t domain[8] = {'E', 'T', 'M', '5', 'C', 'F', 'G', '1'};
  uint8_t buf[108];
  et_sha256_ctx ctx;

  sha256_init(&ctx);
  sha256_update(&ctx, domain, sizeof(domain));
  sha256_update(&ctx, model_config_hash, 32u);

  put_u64_le(buf + 0u, param_count);
  put_f32_le(buf + 8u, canonical_f32(beta1));
  put_f32_le(buf + 12u, canonical_f32(beta2));
  put_f32_le(buf + 16u, canonical_f32(eps));
  put_f32_le(buf + 20u, canonical_f32(weight_decay));
  put_f32_le(buf + 24u, canonical_f32(max_grad_norm));
  put_u32_le(buf + 28u, sched->kind);
  put_f64_le(buf + 32u, canonical_f64(sched->base_lr));
  put_f64_le(buf + 40u, canonical_f64(sched->min_lr));
  put_u64_le(buf + 48u, sched->warmup_steps);
  put_u64_le(buf + 56u, sched->total_steps);
  put_u32_le(buf + 64u, tokenizer_id_len);
  sha256_update(&ctx, buf, 68u);
  sha256_update(&ctx, (const uint8_t *)tokenizer_id, (size_t)tokenizer_id_len);
  sha256_final(&ctx, out_hash);
}

static double ipow_f64(double base, uint64_t exp) {
  double result = 1.0;
  double factor = base;
  while (exp > 0u) {
    if ((exp & UINT64_C(1)) != 0u) {
      result *= factor;
    }
    exp >>= 1u;
    if (exp > 0u) {
      factor *= factor;
    }
  }
  return result;
}

static uint64_t xoshiro256ss_next(uint64_t s[4]) {
  const uint64_t result = rotl64(s[1] * UINT64_C(5), 7u) * UINT64_C(9);
  const uint64_t t = s[1] << 17u;
  s[2] ^= s[0];
  s[3] ^= s[1];
  s[1] ^= s[2];
  s[0] ^= s[3];
  s[2] ^= t;
  s[3] = rotl64(s[3], 45u);
  return result;
}

static void init_default_rng(const uint8_t seed_hash[32], uint64_t rng[4]) {
  static const uint8_t domain[8] = {'E', 'T', 'M', '5', 'R', 'N', 'G', '1'};
  uint8_t digest[32];
  et_sha256_ctx ctx;
  sha256_init(&ctx);
  sha256_update(&ctx, domain, sizeof(domain));
  sha256_update(&ctx, seed_hash, 32u);
  sha256_final(&ctx, digest);
  for (size_t i = 0u; i < 4u; ++i) {
    rng[i] = get_u64_le(digest + i * 8u);
  }
  /* Guarantee non-zero xoshiro256** state. */
  if ((rng[0] | rng[1] | rng[2] | rng[3]) == 0u) {
    rng[0] = UINT64_C(0x9e3779b97f4a7c15);
  }
}

/* --- Checked Linux filesystem I/O (atomic fsync+rename & O_NOFOLLOW read) --- */

typedef struct et_path_parts {
  char *parent;
  char *basename;
} et_path_parts;

static void free_path_parts(et_path_parts *parts) {
  free(parts->parent);
  free(parts->basename);
  parts->parent = NULL;
  parts->basename = NULL;
}

static int split_path(const char *path, et_path_parts *parts) {
  const char *slash;
  const char *base;
  size_t base_len;
  size_t parent_len;
  memset(parts, 0, sizeof(*parts));
  if (path == NULL || path[0] == '\0') {
    errno = EINVAL;
    return -1;
  }
  slash = strrchr(path, '/');
  base = slash == NULL ? path : slash + 1;
  base_len = strlen(base);
  if (base_len == 0u || (base_len == 1u && base[0] == '.') ||
      (base_len == 2u && base[0] == '.' && base[1] == '.')) {
    errno = EINVAL;
    return -1;
  }
  parts->basename = (char *)malloc(base_len + 1u);
  if (parts->basename == NULL) {
    return -1;
  }
  memcpy(parts->basename, base, base_len + 1u);
  if (slash == NULL) {
    parts->parent = (char *)malloc(2u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, ".", 2u);
    }
  } else if (slash == path) {
    parts->parent = (char *)malloc(2u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, "/", 2u);
    }
  } else {
    parent_len = (size_t)(slash - path);
    while (parent_len > 1u && path[parent_len - 1u] == '/') {
      --parent_len;
    }
    parts->parent = (char *)malloc(parent_len + 1u);
    if (parts->parent != NULL) {
      memcpy(parts->parent, path, parent_len);
      parts->parent[parent_len] = '\0';
    }
  }
  if (parts->parent == NULL) {
    free_path_parts(parts);
    errno = ENOMEM;
    return -1;
  }
  return 0;
}

static int open_parent_dir(const char *path, et_path_parts *parts) {
  int fd;
  if (split_path(path, parts) != 0) {
    return -1;
  }
  do {
    fd = open(parts->parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    free_path_parts(parts);
  }
  return fd;
}

static int write_all(int fd, const uint8_t *data, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = write(fd, data + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int read_all_exact(int fd, uint8_t *dst, size_t len) {
  size_t off = 0u;
  while (off < len) {
    const ssize_t n = read(fd, dst + off, len - off);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = ENODATA;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static int fsync_retry(int fd) {
  int rc;
  do {
    rc = fsync(fd);
  } while (rc < 0 && errno == EINTR);
  return rc;
}

static int fill_random_bytes(uint8_t out[ET_MOD5_TEMP_RANDOM_BYTES]) {
  size_t off = 0u;
  while (off < ET_MOD5_TEMP_RANDOM_BYTES) {
    const ssize_t n =
        getrandom(out + off, ET_MOD5_TEMP_RANDOM_BYTES - off, 0u);
    if (n > 0) {
      off += (size_t)n;
    } else if (n == 0) {
      errno = EIO;
      return -1;
    } else if (errno != EINTR) {
      return -1;
    }
  }
  return 0;
}

static void format_temp_name(char name[ET_MOD5_TEMP_NAME_BYTES],
                             const uint8_t rnd[ET_MOD5_TEMP_RANDOM_BYTES]) {
  static const char hex[] = "0123456789abcdef";
  memcpy(name, ".et-m5-", 7u);
  for (size_t i = 0u; i < ET_MOD5_TEMP_RANDOM_BYTES; ++i) {
    name[7u + i * 2u] = hex[rnd[i] >> 4u];
    name[8u + i * 2u] = hex[rnd[i] & 0x0fu];
  }
  memcpy(name + 39u, ".tmp", 5u);
}

static et_mod5_adamw_status atomic_write_file(const char *path,
                                              const uint8_t *data, size_t len) {
  et_path_parts parts;
  struct stat dst_st;
  int dirfd = -1;
  int tmpfd = -1;
  char tmpname[ET_MOD5_TEMP_NAME_BYTES];
  int created = 0;

  if (path == NULL || data == NULL || len == 0u) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  dirfd = open_parent_dir(path, &parts);
  if (dirfd < 0) {
    return ET_MOD5_ADAMW_ERR_IO;
  }
  /* Refuse replacing an existing symlink or non-regular file. */
  if (fstatat(dirfd, parts.basename, &dst_st, AT_SYMLINK_NOFOLLOW) == 0) {
    if (!S_ISREG(dst_st.st_mode)) {
      (void)close(dirfd);
      free_path_parts(&parts);
      return ET_MOD5_ADAMW_ERR_IO;
    }
  } else if (errno != ENOENT) {
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }

  for (uint32_t attempt = 0u; attempt < ET_MOD5_TEMP_ATTEMPTS; ++attempt) {
    uint8_t rnd[ET_MOD5_TEMP_RANDOM_BYTES];
    if (fill_random_bytes(rnd) != 0) {
      goto io_fail;
    }
    format_temp_name(tmpname, rnd);
    do {
      tmpfd = openat(dirfd, tmpname,
                     O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                     (mode_t)0600);
    } while (tmpfd < 0 && errno == EINTR);
    if (tmpfd >= 0) {
      created = 1;
      break;
    }
    if (errno != EEXIST) {
      goto io_fail;
    }
  }
  if (!created) {
    goto io_fail;
  }
  if (write_all(tmpfd, data, len) != 0) {
    goto io_fail;
  }
  if (fsync_retry(tmpfd) != 0) {
    goto io_fail;
  }
  if (close(tmpfd) != 0) {
    tmpfd = -1;
    goto io_fail;
  }
  tmpfd = -1;

  {
    int rc;
    do {
      rc = renameat(dirfd, tmpname, dirfd, parts.basename);
    } while (rc < 0 && errno == EINTR);
    if (rc != 0) {
      goto io_fail;
    }
  }
  created = 0;

  if (fsync_retry(dirfd) != 0) {
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  if (close(dirfd) != 0) {
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  free_path_parts(&parts);
  return ET_MOD5_ADAMW_OK;

io_fail:
  if (tmpfd >= 0) {
    (void)close(tmpfd);
  }
  if (created) {
    (void)unlinkat(dirfd, tmpname, 0);
  }
  if (dirfd >= 0) {
    (void)close(dirfd);
  }
  free_path_parts(&parts);
  return ET_MOD5_ADAMW_ERR_IO;
}

static et_mod5_adamw_status read_regular_nofollow(const char *path,
                                                  uint64_t max_bytes,
                                                  uint8_t **out_buf,
                                                  size_t *out_len) {
  et_path_parts parts;
  struct stat before;
  struct stat after;
  uint8_t *buf = NULL;
  uint8_t trailing = 0u;
  int dirfd = -1;
  int fd = -1;
  size_t file_size = 0u;

  if (path == NULL || out_buf == NULL || out_len == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  *out_buf = NULL;
  *out_len = 0u;

  dirfd = open_parent_dir(path, &parts);
  if (dirfd < 0) {
    return ET_MOD5_ADAMW_ERR_IO;
  }
  do {
    fd = openat(dirfd, parts.basename,
                O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  if (fstat(fd, &before) != 0 || !S_ISREG(before.st_mode) ||
      before.st_size < 0) {
    (void)close(fd);
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  if ((uint64_t)before.st_size > max_bytes ||
      (uint64_t)before.st_size > (uint64_t)SIZE_MAX) {
    (void)close(fd);
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }
  file_size = (size_t)before.st_size;
  if (file_size > 0u) {
    buf = (uint8_t *)malloc(file_size);
    if (buf == NULL) {
      (void)close(fd);
      (void)close(dirfd);
      free_path_parts(&parts);
      return ET_MOD5_ADAMW_ERR_ALLOC;
    }
    if (read_all_exact(fd, buf, file_size) != 0) {
      free(buf);
      (void)close(fd);
      (void)close(dirfd);
      free_path_parts(&parts);
      return ET_MOD5_ADAMW_ERR_IO;
    }
  }
  for (;;) {
    const ssize_t n = read(fd, &trailing, 1u);
    if (n == 0) {
      break;
    }
    if (n > 0) {
      free(buf);
      (void)close(fd);
      (void)close(dirfd);
      free_path_parts(&parts);
      return ET_MOD5_ADAMW_ERR_IO;
    }
    if (errno != EINTR) {
      free(buf);
      (void)close(fd);
      (void)close(dirfd);
      free_path_parts(&parts);
      return ET_MOD5_ADAMW_ERR_IO;
    }
  }
  if (fstat(fd, &after) != 0 || after.st_dev != before.st_dev ||
      after.st_ino != before.st_ino || after.st_size != before.st_size) {
    free(buf);
    (void)close(fd);
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  if (close(fd) != 0) {
    free(buf);
    (void)close(dirfd);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  if (close(dirfd) != 0) {
    free(buf);
    free_path_parts(&parts);
    return ET_MOD5_ADAMW_ERR_IO;
  }
  free_path_parts(&parts);
  *out_buf = buf;
  *out_len = file_size;
  return ET_MOD5_ADAMW_OK;
}

/* --- Core C11 / C++17 Public Functions --- */

et_mod5_adamw_status
et_mod5_adamw_compute_config_hash(const et_mod5_adamw_config_v1 *cfg,
                                  uint8_t out_hash[32]) {
  uint32_t tok_len = 0u;
  et_mod5_adamw_status st;
  if (out_hash == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = validate_config(cfg, &tok_len);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  compute_full_config_hash_parts(
      cfg->param_count, cfg->beta1, cfg->beta2, cfg->eps, cfg->weight_decay,
      cfg->max_grad_norm, &cfg->schedule, cfg->model_config_hash,
      cfg->tokenizer_id, tok_len, out_hash);
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status
et_mod5_adamw_compute_lr(const et_mod5_lr_schedule_config_v1 *sched,
                         uint64_t step, double *out_lr) {
  et_mod5_adamw_status st;
  double lr = 0.0;

  if (out_lr == NULL || step == 0u) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = check_fp_env();
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  st = validate_schedule(sched);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }

  if (sched->kind == ET_MOD5_LR_CONSTANT) {
    lr = sched->base_lr;
  } else if (sched->warmup_steps > 0u && step <= sched->warmup_steps) {
    const double ratio = (double)step / (double)sched->warmup_steps;
    lr = sched->base_lr * ratio;
  } else if (step >= sched->total_steps) {
    lr = sched->min_lr;
  } else {
    /* Guaranteed: warmup_steps < step < total_steps, so total > warmup. */
    const double progress = (double)(step - sched->warmup_steps) /
                            (double)(sched->total_steps - sched->warmup_steps);
    if (sched->kind == ET_MOD5_LR_LINEAR_WARMUP_LINEAR_DECAY) {
      lr = sched->min_lr + (sched->base_lr - sched->min_lr) * (1.0 - progress);
    } else {
      const double cosine_factor = 0.5 * (1.0 + cos(ET_MOD5_PI * progress));
      lr = sched->min_lr + (sched->base_lr - sched->min_lr) * cosine_factor;
    }
  }

  if (!is_finite_f64(lr) || lr < 0.0) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  *out_lr = canonical_f64(lr);
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_create(const et_mod5_adamw_config_v1 *cfg,
                                          et_mod5_adamw_state **out_state) {
  uint32_t tok_len = 0u;
  et_mod5_adamw_status st;
  et_mod5_adamw_state *state = NULL;
  size_t n = 0u;

  if (out_state == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  *out_state = NULL;

  st = check_fp_env();
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  st = validate_config(cfg, &tok_len);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  if (cfg->param_count > (uint64_t)(SIZE_MAX / sizeof(float))) {
    return ET_MOD5_ADAMW_ERR_OVERFLOW;
  }
  n = (size_t)cfg->param_count;

  state = (et_mod5_adamw_state *)calloc(1u, sizeof(*state));
  if (state == NULL) {
    return ET_MOD5_ADAMW_ERR_ALLOC;
  }
  state->params = (float *)calloc(n, sizeof(float));
  state->m = (float *)calloc(n, sizeof(float));
  state->v = (float *)calloc(n, sizeof(float));
  if (state->params == NULL || state->m == NULL || state->v == NULL) {
    et_mod5_adamw_destroy(state);
    return ET_MOD5_ADAMW_ERR_ALLOC;
  }

  state->param_count = cfg->param_count;
  state->beta1 = canonical_f32(cfg->beta1);
  state->beta2 = canonical_f32(cfg->beta2);
  state->eps = cfg->eps;
  state->weight_decay = canonical_f32(cfg->weight_decay);
  state->max_grad_norm = canonical_f32(cfg->max_grad_norm);
  state->schedule = cfg->schedule;
  state->schedule.base_lr = canonical_f64(cfg->schedule.base_lr);
  state->schedule.min_lr = canonical_f64(cfg->schedule.min_lr);
  memcpy(state->model_config_hash, cfg->model_config_hash, 32u);
  state->tokenizer_id_len = tok_len;
  memcpy(state->tokenizer_id, cfg->tokenizer_id, tok_len);
  state->tokenizer_id[tok_len] = '\0';
  compute_full_config_hash_parts(
      state->param_count, state->beta1, state->beta2, state->eps,
      state->weight_decay, state->max_grad_norm, &state->schedule,
      state->model_config_hash, state->tokenizer_id, state->tokenizer_id_len,
      state->full_config_hash);
  init_default_rng(state->full_config_hash, state->rng_state);

  *out_state = state;
  return ET_MOD5_ADAMW_OK;
}

void et_mod5_adamw_destroy(et_mod5_adamw_state *state) {
  if (state == NULL) {
    return;
  }
  free(state->params);
  free(state->m);
  free(state->v);
  free(state);
}

et_mod5_adamw_status et_mod5_adamw_set_params(et_mod5_adamw_state *state,
                                              const float *params,
                                              size_t count) {
  if (state == NULL || params == NULL ||
      (uint64_t)count != state->param_count) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  for (size_t i = 0u; i < count; ++i) {
    if (!is_finite_f32(params[i])) {
      return ET_MOD5_ADAMW_ERR_NONFINITE;
    }
  }
  memcpy(state->params, params, count * sizeof(float));
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_get_params(const et_mod5_adamw_state *state,
                                              float *out_params, size_t count) {
  if (state == NULL || out_params == NULL ||
      (uint64_t)count != state->param_count) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memcpy(out_params, state->params, count * sizeof(float));
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_get_moments(const et_mod5_adamw_state *state,
                                               float *out_m, float *out_v,
                                               size_t count) {
  if (state == NULL || (out_m == NULL && out_v == NULL) ||
      (uint64_t)count != state->param_count) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if (out_m != NULL) {
    memcpy(out_m, state->m, count * sizeof(float));
  }
  if (out_v != NULL) {
    memcpy(out_v, state->v, count * sizeof(float));
  }
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status
et_mod5_adamw_set_cursor(et_mod5_adamw_state *state,
                         const et_mod5_dataset_cursor_v1 *cursor) {
  if (state == NULL || cursor == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  state->cursor = *cursor;
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status
et_mod5_adamw_get_cursor(const et_mod5_adamw_state *state,
                         et_mod5_dataset_cursor_v1 *out_cursor) {
  if (state == NULL || out_cursor == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  *out_cursor = state->cursor;
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_set_rng(et_mod5_adamw_state *state,
                                           const uint64_t rng_state[4]) {
  if (state == NULL || rng_state == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  if ((rng_state[0] | rng_state[1] | rng_state[2] | rng_state[3]) == 0u) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memcpy(state->rng_state, rng_state, sizeof(state->rng_state));
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_get_rng(const et_mod5_adamw_state *state,
                                           uint64_t out_rng_state[4]) {
  if (state == NULL || out_rng_state == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memcpy(out_rng_state, state->rng_state, sizeof(state->rng_state));
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_rng_next_u64(et_mod5_adamw_state *state,
                                                uint64_t *out_value) {
  if (state == NULL || out_value == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  *out_value = xoshiro256ss_next(state->rng_state);
  return ET_MOD5_ADAMW_OK;
}

uint64_t et_mod5_adamw_get_step(const et_mod5_adamw_state *state) {
  return state == NULL ? 0u : state->step;
}

uint64_t et_mod5_adamw_get_cumulative_tokens(const et_mod5_adamw_state *state) {
  return state == NULL ? 0u : state->cumulative_tokens;
}

et_mod5_adamw_status et_mod5_adamw_step(et_mod5_adamw_state *state,
                                        const float *grads, size_t grad_count,
                                        uint64_t step_tokens,
                                        et_mod5_adamw_step_info_v1 *out_info) {
  et_mod5_adamw_status st;
  uint64_t next_step;
  double lr_d = 0.0;
  double sum_sq = 0.0;
  double grad_norm = 0.0;
  double clip_scale_d = 1.0;
  uint32_t clipped = 0u;
  double bc1_d;
  double bc2_d;
  float bc1;
  float bc2;
  float lr_f32;
  float one_minus_b1;
  float one_minus_b2;
  float decay_factor;
  size_t n;

  if (state == NULL || grads == NULL ||
      (uint64_t)grad_count != state->param_count) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = check_fp_env();
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  if (state->step == UINT64_MAX) {
    return ET_MOD5_ADAMW_ERR_OVERFLOW;
  }
  if (step_tokens > UINT64_MAX - state->cumulative_tokens) {
    return ET_MOD5_ADAMW_ERR_OVERFLOW;
  }

  next_step = state->step + UINT64_C(1);
  st = et_mod5_adamw_compute_lr(&state->schedule, next_step, &lr_d);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  lr_f32 = (float)lr_d;
  if (!is_finite_f32(lr_f32) || lr_f32 < 0.0f) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }

  /* Exact bias corrections 1 - beta1^t, 1 - beta2^t in binary64, then f32. */
  bc1_d = 1.0 - ipow_f64((double)state->beta1, next_step);
  bc2_d = 1.0 - ipow_f64((double)state->beta2, next_step);
  if (!is_finite_f64(bc1_d) || bc1_d <= 0.0 || !is_finite_f64(bc2_d) ||
      bc2_d <= 0.0) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  bc1 = (float)bc1_d;
  bc2 = (float)bc2_d;
  if (!is_finite_f32(bc1) || bc1 <= 0.0f || !is_finite_f32(bc2) ||
      bc2 <= 0.0f) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }

  one_minus_b1 = 1.0f - state->beta1;
  one_minus_b2 = 1.0f - state->beta2;
  decay_factor = 1.0f - lr_f32 * state->weight_decay;
  if (!is_finite_f32(decay_factor)) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }

  /* Pass 1: validate all live params, moments, grads, and accumulate global
   * L2 norm in binary64. Zero state mutation occurs here. */
  n = grad_count;
  for (size_t i = 0u; i < n; ++i) {
    const float p = state->params[i];
    const float mv = state->m[i];
    const float vv = state->v[i];
    const float g = grads[i];
    if (!is_finite_f32(p) || !is_finite_f32(mv) || !is_finite_f32(vv) ||
        vv < 0.0f || !is_finite_f32(g)) {
      return ET_MOD5_ADAMW_ERR_NONFINITE;
    }
    {
      const double gd = (double)g;
      sum_sq += gd * gd;
    }
  }
  if (!is_finite_f64(sum_sq)) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  grad_norm = sqrt(sum_sq);
  if (!is_finite_f64(grad_norm)) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }

  if (state->max_grad_norm > 0.0f && grad_norm > (double)state->max_grad_norm) {
    clip_scale_d = (double)state->max_grad_norm / grad_norm;
    if (!is_finite_f64(clip_scale_d) || clip_scale_d < 0.0) {
      return ET_MOD5_ADAMW_ERR_NONFINITE;
    }
    clipped = 1u;
  }

  /* Pass 2: preflight every element update before committing any write. */
  for (size_t i = 0u; i < n; ++i) {
    const float g =
        clipped ? (float)((double)grads[i] * clip_scale_d) : grads[i];
    const float g_sq = g * g;
    const float m_new = state->beta1 * state->m[i] + one_minus_b1 * g;
    const float v_new = state->beta2 * state->v[i] + one_minus_b2 * g_sq;
    const float m_hat = m_new / bc1;
    const float v_hat = v_new / bc2;
    const float denom = sqrtf(v_hat) + state->eps;
    const float p_decay = state->params[i] * decay_factor;
    const float p_new = p_decay - lr_f32 * (m_hat / denom);
    if (!is_finite_f32(g) || !is_finite_f32(g_sq) || !is_finite_f32(m_new) ||
        !is_finite_f32(v_new) || v_new < 0.0f || !is_finite_f32(m_hat) ||
        !is_finite_f32(v_hat) || !is_finite_f32(denom) || denom <= 0.0f ||
        !is_finite_f32(p_decay) || !is_finite_f32(p_new)) {
      return ET_MOD5_ADAMW_ERR_NONFINITE;
    }
  }

  /* Pass 3: atomic commit — all operations guaranteed finite. */
  for (size_t i = 0u; i < n; ++i) {
    const float g =
        clipped ? (float)((double)grads[i] * clip_scale_d) : grads[i];
    const float g_sq = g * g;
    const float m_new = state->beta1 * state->m[i] + one_minus_b1 * g;
    const float v_new = state->beta2 * state->v[i] + one_minus_b2 * g_sq;
    const float m_hat = m_new / bc1;
    const float v_hat = v_new / bc2;
    const float denom = sqrtf(v_hat) + state->eps;
    const float p_decay = state->params[i] * decay_factor;
    const float p_new = p_decay - lr_f32 * (m_hat / denom);
    state->m[i] = m_new;
    state->v[i] = v_new;
    state->params[i] = p_new;
  }

  state->step = next_step;
  state->cumulative_tokens += step_tokens;

  if (out_info != NULL) {
    out_info->step = next_step;
    out_info->lr = lr_d;
    out_info->grad_norm = grad_norm;
    out_info->clip_scale = clip_scale_d;
    out_info->clipped = clipped;
    out_info->reserved = 0u;
  }
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status
et_mod5_adamw_save_snapshot(const et_mod5_adamw_state *state,
                            const char *path) {
  uint64_t payload_bytes;
  uint64_t total_bytes;
  uint8_t *buf = NULL;
  size_t off;
  size_t n;
  et_sha256_ctx sha;
  et_mod5_adamw_status st;

  if (state == NULL || path == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  n = (size_t)state->param_count;
  for (size_t i = 0u; i < n; ++i) {
    if (!is_finite_f32(state->params[i]) || !is_finite_f32(state->m[i]) ||
        !is_finite_f32(state->v[i]) || state->v[i] < 0.0f) {
      return ET_MOD5_ADAMW_ERR_NONFINITE;
    }
  }

  payload_bytes = UINT64_C(12) * state->param_count;
  total_bytes = (uint64_t)ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES +
                (uint64_t)state->tokenizer_id_len + payload_bytes +
                (uint64_t)ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES;
  if (total_bytes > ET_MOD5_SNAPSHOT_MAX_BYTES ||
      total_bytes > (uint64_t)SIZE_MAX) {
    return ET_MOD5_ADAMW_ERR_OVERFLOW;
  }

  buf = (uint8_t *)malloc((size_t)total_bytes);
  if (buf == NULL) {
    return ET_MOD5_ADAMW_ERR_ALLOC;
  }

  /* Fixed 224-byte header */
  memcpy(buf + 0u, ET_MOD5_ADAMW_SNAPSHOT_MAGIC,
         ET_MOD5_ADAMW_SNAPSHOT_MAGIC_BYTES);
  put_u16_le(buf + 8u, (uint16_t)ET_MOD5_ADAMW_SNAPSHOT_VERSION_MAJOR);
  put_u16_le(buf + 10u, (uint16_t)ET_MOD5_ADAMW_SNAPSHOT_VERSION_MINOR);
  put_u32_le(buf + 12u, 0u); /* flags */
  put_u64_le(buf + 16u, state->param_count);
  put_u64_le(buf + 24u, state->step);
  put_u64_le(buf + 32u, state->cumulative_tokens);
  put_u64_le(buf + 40u, state->cursor.shard_index);
  put_u64_le(buf + 48u, state->cursor.token_offset);
  put_u64_le(buf + 56u, state->cursor.epoch);
  for (size_t i = 0u; i < 4u; ++i) {
    put_u64_le(buf + 64u + i * 8u, state->rng_state[i]);
  }
  put_f32_le(buf + 96u, state->beta1);
  put_f32_le(buf + 100u, state->beta2);
  put_f32_le(buf + 104u, state->eps);
  put_f32_le(buf + 108u, state->weight_decay);
  put_f32_le(buf + 112u, state->max_grad_norm);
  put_u32_le(buf + 116u, state->schedule.kind);
  put_f64_le(buf + 120u, state->schedule.base_lr);
  put_f64_le(buf + 128u, state->schedule.min_lr);
  put_u64_le(buf + 136u, state->schedule.warmup_steps);
  put_u64_le(buf + 144u, state->schedule.total_steps);
  memcpy(buf + 152u, state->model_config_hash, 32u);
  memcpy(buf + 184u, state->full_config_hash, 32u);
  put_u32_le(buf + 216u, state->tokenizer_id_len);
  put_u32_le(buf + 220u, 0u); /* reserved */

  off = ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES;
  memcpy(buf + off, state->tokenizer_id, (size_t)state->tokenizer_id_len);
  off += (size_t)state->tokenizer_id_len;

  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(buf + off + i * 4u, state->params[i]);
  }
  off += n * 4u;
  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(buf + off + i * 4u, state->m[i]);
  }
  off += n * 4u;
  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(buf + off + i * 4u, state->v[i]);
  }
  off += n * 4u;

  sha256_init(&sha);
  sha256_update(&sha, buf, off);
  sha256_final(&sha, buf + off);

  st = atomic_write_file(path, buf, (size_t)total_bytes);
  free(buf);
  return st;
}

et_mod5_adamw_status et_mod5_adamw_load_snapshot(et_mod5_adamw_state *state,
                                                 const char *path) {
  uint8_t *buf = NULL;
  size_t file_size = 0u;
  et_mod5_adamw_status st;
  uint16_t ver_major;
  uint16_t ver_minor;
  uint32_t flags;
  uint32_t reserved;
  uint64_t param_count;
  uint64_t step;
  uint64_t cumulative_tokens;
  et_mod5_dataset_cursor_v1 cursor;
  uint64_t rng_state[4];
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float max_grad_norm;
  et_mod5_lr_schedule_config_v1 sched;
  uint8_t model_config_hash[32];
  uint8_t stored_full_hash[32];
  uint8_t recomputed_full_hash[32];
  uint32_t tok_len;
  char tok_buf[ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES + 1u];
  uint64_t expected_size;
  uint8_t computed_digest[32];
  et_sha256_ctx sha;
  float *tmp_params = NULL;
  float *tmp_m = NULL;
  float *tmp_v = NULL;
  size_t off;
  size_t n;

  if (state == NULL || path == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }

  st = read_regular_nofollow(path, ET_MOD5_SNAPSHOT_MAX_BYTES, &buf,
                             &file_size);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }

  if (file_size < ET_MOD5_ADAMW_SNAPSHOT_MAGIC_BYTES) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }
  if (memcmp(buf, ET_MOD5_ADAMW_SNAPSHOT_MAGIC,
             ET_MOD5_ADAMW_SNAPSHOT_MAGIC_BYTES) != 0) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_BAD_MAGIC;
  }
  if (file_size < 12u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }
  ver_major = get_u16_le(buf + 8u);
  ver_minor = get_u16_le(buf + 10u);
  if (ver_major != ET_MOD5_ADAMW_SNAPSHOT_VERSION_MAJOR ||
      ver_minor != ET_MOD5_ADAMW_SNAPSHOT_VERSION_MINOR) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_VERSION_MISMATCH;
  }
  if (file_size < ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES +
                      ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }

  flags = get_u32_le(buf + 12u);
  param_count = get_u64_le(buf + 16u);
  tok_len = get_u32_le(buf + 216u);
  reserved = get_u32_le(buf + 220u);

  if (param_count == 0u || param_count > ET_MOD5_ADAMW_MAX_PARAMS ||
      tok_len == 0u || tok_len > ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }
  expected_size = (uint64_t)ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES +
                  (uint64_t)tok_len + UINT64_C(12) * param_count +
                  (uint64_t)ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES;
  if ((uint64_t)file_size != expected_size) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }

  /* Verify trailing SHA-256 over all preceding bytes before trusting payload. */
  sha256_init(&sha);
  sha256_update(&sha, buf, file_size - ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES);
  sha256_final(&sha, computed_digest);
  if (!ct_equal_32(computed_digest,
                   buf + file_size - ET_MOD5_ADAMW_SNAPSHOT_CHECKSUM_BYTES)) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH;
  }

  if (flags != 0u || reserved != 0u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }

  step = get_u64_le(buf + 24u);
  cumulative_tokens = get_u64_le(buf + 32u);
  cursor.shard_index = get_u64_le(buf + 40u);
  cursor.token_offset = get_u64_le(buf + 48u);
  cursor.epoch = get_u64_le(buf + 56u);
  for (size_t i = 0u; i < 4u; ++i) {
    rng_state[i] = get_u64_le(buf + 64u + i * 8u);
  }
  if ((rng_state[0] | rng_state[1] | rng_state[2] | rng_state[3]) == 0u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }

  beta1 = get_f32_le(buf + 96u);
  beta2 = get_f32_le(buf + 100u);
  eps = get_f32_le(buf + 104u);
  weight_decay = get_f32_le(buf + 108u);
  max_grad_norm = get_f32_le(buf + 112u);
  memset(&sched, 0, sizeof(sched));
  sched.kind = get_u32_le(buf + 116u);
  sched.base_lr = get_f64_le(buf + 120u);
  sched.min_lr = get_f64_le(buf + 128u);
  sched.warmup_steps = get_u64_le(buf + 136u);
  sched.total_steps = get_u64_le(buf + 144u);
  memcpy(model_config_hash, buf + 152u, 32u);
  memcpy(stored_full_hash, buf + 184u, 32u);

  off = ET_MOD5_ADAMW_SNAPSHOT_HEADER_BYTES;
  memcpy(tok_buf, buf + off, (size_t)tok_len);
  tok_buf[tok_len] = '\0';
  for (size_t i = 0u; i < (size_t)tok_len; ++i) {
    if (tok_buf[i] == '\0') {
      free(buf);
      return ET_MOD5_ADAMW_ERR_INVALID_ARG;
    }
  }
  off += (size_t)tok_len;

  {
    et_mod5_adamw_config_v1 parsed_cfg;
    memset(&parsed_cfg, 0, sizeof(parsed_cfg));
    parsed_cfg.param_count = param_count;
    parsed_cfg.beta1 = beta1;
    parsed_cfg.beta2 = beta2;
    parsed_cfg.eps = eps;
    parsed_cfg.weight_decay = weight_decay;
    parsed_cfg.max_grad_norm = max_grad_norm;
    parsed_cfg.schedule = sched;
    memcpy(parsed_cfg.model_config_hash, model_config_hash, 32u);
    parsed_cfg.tokenizer_id = tok_buf;
    st = validate_config(&parsed_cfg, NULL);
    if (st != ET_MOD5_ADAMW_OK) {
      free(buf);
      return st;
    }
  }

  compute_full_config_hash_parts(param_count, beta1, beta2, eps, weight_decay,
                                 max_grad_norm, &sched, model_config_hash,
                                 tok_buf, tok_len, recomputed_full_hash);
  if (!ct_equal_32(stored_full_hash, recomputed_full_hash)) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH;
  }

  /* Verify complete configuration compatibility against live trainer. */
  if (param_count != state->param_count ||
      !ct_equal_32(model_config_hash, state->model_config_hash) ||
      !ct_equal_32(stored_full_hash, state->full_config_hash) ||
      tok_len != state->tokenizer_id_len ||
      memcmp(tok_buf, state->tokenizer_id, (size_t)tok_len) != 0 ||
      f32_to_bits(canonical_f32(beta1)) != f32_to_bits(state->beta1) ||
      f32_to_bits(canonical_f32(beta2)) != f32_to_bits(state->beta2) ||
      f32_to_bits(eps) != f32_to_bits(state->eps) ||
      f32_to_bits(canonical_f32(weight_decay)) !=
          f32_to_bits(state->weight_decay) ||
      f32_to_bits(canonical_f32(max_grad_norm)) !=
          f32_to_bits(state->max_grad_norm) ||
      sched.kind != state->schedule.kind ||
      f64_to_bits(canonical_f64(sched.base_lr)) !=
          f64_to_bits(state->schedule.base_lr) ||
      f64_to_bits(canonical_f64(sched.min_lr)) !=
          f64_to_bits(state->schedule.min_lr) ||
      sched.warmup_steps != state->schedule.warmup_steps ||
      sched.total_steps != state->schedule.total_steps) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH;
  }

  /* Stage decoded arrays and validate finite-only values before mutation. */
  n = (size_t)param_count;
  tmp_params = (float *)malloc(n * sizeof(float));
  tmp_m = (float *)malloc(n * sizeof(float));
  tmp_v = (float *)malloc(n * sizeof(float));
  if (tmp_params == NULL || tmp_m == NULL || tmp_v == NULL) {
    free(tmp_params);
    free(tmp_m);
    free(tmp_v);
    free(buf);
    return ET_MOD5_ADAMW_ERR_ALLOC;
  }

  for (size_t i = 0u; i < n; ++i) {
    tmp_params[i] = get_f32_le(buf + off + i * 4u);
    if (!is_finite_f32(tmp_params[i])) {
      goto nonfinite_fail;
    }
  }
  off += n * 4u;
  for (size_t i = 0u; i < n; ++i) {
    tmp_m[i] = get_f32_le(buf + off + i * 4u);
    if (!is_finite_f32(tmp_m[i])) {
      goto nonfinite_fail;
    }
  }
  off += n * 4u;
  for (size_t i = 0u; i < n; ++i) {
    tmp_v[i] = get_f32_le(buf + off + i * 4u);
    if (!is_finite_f32(tmp_v[i]) || tmp_v[i] < 0.0f) {
      goto nonfinite_fail;
    }
  }

  /* Atomic commit to live trainer state. */
  memcpy(state->params, tmp_params, n * sizeof(float));
  memcpy(state->m, tmp_m, n * sizeof(float));
  memcpy(state->v, tmp_v, n * sizeof(float));
  state->step = step;
  state->cumulative_tokens = cumulative_tokens;
  state->cursor = cursor;
  memcpy(state->rng_state, rng_state, sizeof(state->rng_state));

  free(tmp_params);
  free(tmp_m);
  free(tmp_v);
  free(buf);
  return ET_MOD5_ADAMW_OK;

nonfinite_fail:
  free(tmp_params);
  free(tmp_m);
  free(tmp_v);
  free(buf);
  return ET_MOD5_ADAMW_ERR_NONFINITE;
}

et_mod5_adamw_status et_mod5_adamw_sha256_bytes(const uint8_t *data, size_t len,
                                                uint8_t out_digest[32]) {
  et_sha256_ctx ctx;
  if (out_digest == NULL || (data == NULL && len != 0u)) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  sha256_init(&ctx);
  if (len > 0u) {
    sha256_update(&ctx, data, len);
  }
  sha256_final(&ctx, out_digest);
  return ET_MOD5_ADAMW_OK;
}

et_mod5_adamw_status et_mod5_adamw_sha256_file(const char *path,
                                               uint8_t out_digest[32]) {
  uint8_t *buf = NULL;
  size_t len = 0u;
  et_mod5_adamw_status st;
  if (path == NULL || out_digest == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = read_regular_nofollow(path, ET_MOD5_SNAPSHOT_MAX_BYTES, &buf, &len);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  st = et_mod5_adamw_sha256_bytes(buf, len, out_digest);
  free(buf);
  return st;
}

/* --- Compiled Eshkol AOT Bridge Functions --- */

static float f64_to_checked_f32(double value, int *ok) {
  if (!is_finite_f64(value) || value > (double)FLT_MAX ||
      value < -(double)FLT_MAX) {
    *ok = 0;
    return 0.0f;
  }
  *ok = 1;
  return (float)value;
}

int32_t et_mod5_adamw_aot_create(int64_t param_count, double beta1,
                                 double beta2, double eps, double weight_decay,
                                 double max_grad_norm, int32_t sched_kind,
                                 double base_lr, double min_lr,
                                 int64_t warmup_steps, int64_t total_steps,
                                 const char *tokenizer_id,
                                 const char *model_config_tag,
                                 et_mod5_adamw_state **out_state) {
  et_mod5_adamw_config_v1 cfg;
  int ok1 = 0, ok2 = 0, ok3 = 0, ok4 = 0, ok5 = 0;

  if (out_state == NULL || param_count <= 0 || warmup_steps < 0 ||
      total_steps < 0 || sched_kind < 0 || model_config_tag == NULL ||
      model_config_tag[0] == '\0') {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memset(&cfg, 0, sizeof(cfg));
  cfg.param_count = (uint64_t)param_count;
  cfg.beta1 = f64_to_checked_f32(beta1, &ok1);
  cfg.beta2 = f64_to_checked_f32(beta2, &ok2);
  cfg.eps = f64_to_checked_f32(eps, &ok3);
  cfg.weight_decay = f64_to_checked_f32(weight_decay, &ok4);
  cfg.max_grad_norm = f64_to_checked_f32(max_grad_norm, &ok5);
  if (!ok1 || !ok2 || !ok3 || !ok4 || !ok5) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  cfg.schedule.kind = (uint32_t)sched_kind;
  cfg.schedule.base_lr = base_lr;
  cfg.schedule.min_lr = min_lr;
  cfg.schedule.warmup_steps = (uint64_t)warmup_steps;
  cfg.schedule.total_steps = (uint64_t)total_steps;
  cfg.tokenizer_id = tokenizer_id;

  (void)et_mod5_adamw_sha256_bytes((const uint8_t *)model_config_tag,
                                   strlen(model_config_tag),
                                   cfg.model_config_hash);
  return et_mod5_adamw_create(&cfg, out_state);
}

void et_mod5_adamw_aot_destroy(et_mod5_adamw_state *state) {
  et_mod5_adamw_destroy(state);
}

int32_t et_mod5_adamw_aot_set_param_f64(et_mod5_adamw_state *state,
                                        int64_t index, double value) {
  int ok = 0;
  float v;
  if (state == NULL || index < 0 || (uint64_t)index >= state->param_count) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  v = f64_to_checked_f32(value, &ok);
  if (!ok) {
    return ET_MOD5_ADAMW_ERR_NONFINITE;
  }
  state->params[(size_t)index] = v;
  return ET_MOD5_ADAMW_OK;
}

double et_mod5_adamw_aot_get_param_f64(const et_mod5_adamw_state *state,
                                       int64_t index) {
  if (state == NULL || index < 0 || (uint64_t)index >= state->param_count) {
    return (double)NAN;
  }
  return (double)state->params[(size_t)index];
}

int64_t et_mod5_adamw_aot_get_param_bits(const et_mod5_adamw_state *state,
                                         int64_t index) {
  if (state == NULL || index < 0 || (uint64_t)index >= state->param_count) {
    return -1;
  }
  return (int64_t)(uint64_t)f32_to_bits(state->params[(size_t)index]);
}

int64_t et_mod5_adamw_aot_get_m_bits(const et_mod5_adamw_state *state,
                                     int64_t index) {
  if (state == NULL || index < 0 || (uint64_t)index >= state->param_count) {
    return -1;
  }
  return (int64_t)(uint64_t)f32_to_bits(state->m[(size_t)index]);
}

int64_t et_mod5_adamw_aot_get_v_bits(const et_mod5_adamw_state *state,
                                     int64_t index) {
  if (state == NULL || index < 0 || (uint64_t)index >= state->param_count) {
    return -1;
  }
  return (int64_t)(uint64_t)f32_to_bits(state->v[(size_t)index]);
}

int32_t et_mod5_adamw_aot_step(et_mod5_adamw_state *state, const float *grads,
                               int64_t grad_count, int64_t step_tokens,
                               double *out_lr, double *out_grad_norm,
                               int32_t *out_clipped) {
  et_mod5_adamw_step_info_v1 info;
  et_mod5_adamw_status st;
  if (grad_count < 0 || step_tokens < 0) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memset(&info, 0, sizeof(info));
  st = et_mod5_adamw_step(state, grads, (size_t)grad_count,
                          (uint64_t)step_tokens, &info);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  if (out_lr != NULL) {
    *out_lr = info.lr;
  }
  if (out_grad_norm != NULL) {
    *out_grad_norm = info.grad_norm;
  }
  if (out_clipped != NULL) {
    *out_clipped = (int32_t)info.clipped;
  }
  return ET_MOD5_ADAMW_OK;
}

int32_t et_mod5_adamw_aot_save(const et_mod5_adamw_state *state,
                               const char *path) {
  return et_mod5_adamw_save_snapshot(state, path);
}

int32_t et_mod5_adamw_aot_load(et_mod5_adamw_state *state, const char *path) {
  return et_mod5_adamw_load_snapshot(state, path);
}

int32_t et_mod5_adamw_aot_set_cursor(et_mod5_adamw_state *state,
                                     int64_t shard_index, int64_t token_offset,
                                     int64_t epoch) {
  et_mod5_dataset_cursor_v1 cursor;
  if (shard_index < 0 || token_offset < 0 || epoch < 0) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  cursor.shard_index = (uint64_t)shard_index;
  cursor.token_offset = (uint64_t)token_offset;
  cursor.epoch = (uint64_t)epoch;
  return et_mod5_adamw_set_cursor(state, &cursor);
}

int64_t et_mod5_adamw_aot_get_cursor_shard(const et_mod5_adamw_state *state) {
  return state == NULL ? -1 : (int64_t)state->cursor.shard_index;
}

int64_t et_mod5_adamw_aot_get_cursor_offset(const et_mod5_adamw_state *state) {
  return state == NULL ? -1 : (int64_t)state->cursor.token_offset;
}

int64_t et_mod5_adamw_aot_get_cursor_epoch(const et_mod5_adamw_state *state) {
  return state == NULL ? -1 : (int64_t)state->cursor.epoch;
}

int32_t et_mod5_adamw_aot_set_rng(et_mod5_adamw_state *state, int64_t s0,
                                  int64_t s1, int64_t s2, int64_t s3) {
  const uint64_t rng[4] = {(uint64_t)s0, (uint64_t)s1, (uint64_t)s2,
                           (uint64_t)s3};
  return et_mod5_adamw_set_rng(state, rng);
}

int64_t et_mod5_adamw_aot_get_rng_word(const et_mod5_adamw_state *state,
                                       int32_t lane) {
  if (state == NULL || lane < 0 || lane >= 4) {
    return -1;
  }
  return (int64_t)state->rng_state[lane];
}

int64_t et_mod5_adamw_aot_rng_next(et_mod5_adamw_state *state) {
  uint64_t val = 0u;
  if (et_mod5_adamw_rng_next_u64(state, &val) != ET_MOD5_ADAMW_OK) {
    return -1;
  }
  /* Mask to positive 63-bit signed integer for convenient Eshkol i64 display. */
  return (int64_t)(val & UINT64_C(0x7fffffffffffffff));
}

int64_t et_mod5_adamw_aot_get_step(const et_mod5_adamw_state *state) {
  return state == NULL ? -1 : (int64_t)state->step;
}

int64_t et_mod5_adamw_aot_get_tokens(const et_mod5_adamw_state *state) {
  return state == NULL ? -1 : (int64_t)state->cumulative_tokens;
}

int32_t et_mod5_adamw_aot_compute_lr(int32_t sched_kind, double base_lr,
                                     double min_lr, int64_t warmup_steps,
                                     int64_t total_steps, int64_t step,
                                     double *out_lr) {
  et_mod5_lr_schedule_config_v1 sched;
  if (sched_kind < 0 || warmup_steps < 0 || total_steps < 0 || step <= 0) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  memset(&sched, 0, sizeof(sched));
  sched.kind = (uint32_t)sched_kind;
  sched.base_lr = base_lr;
  sched.min_lr = min_lr;
  sched.warmup_steps = (uint64_t)warmup_steps;
  sched.total_steps = (uint64_t)total_steps;
  return et_mod5_adamw_compute_lr(&sched, (uint64_t)step, out_lr);
}

double et_mod5_adamw_aot_load_f64(const double *cell) {
  double value = (double)NAN;
  if (cell != NULL) {
    memcpy(&value, cell, sizeof(value));
  }
  return value;
}

int32_t et_mod5_adamw_aot_state_sha256_hex(const et_mod5_adamw_state *state,
                                           char *out_hex, int64_t capacity) {
  static const uint8_t domain[8] = {'E', 'T', 'M', '5', 'S', 'T', 'A', '1'};
  uint8_t meta[72];
  uint8_t word[4];
  uint8_t digest[32];
  et_sha256_ctx ctx;
  size_t n;

  if (state == NULL || out_hex == NULL || capacity < 65) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  sha256_init(&ctx);
  sha256_update(&ctx, domain, sizeof(domain));
  sha256_update(&ctx, state->full_config_hash, 32u);
  put_u64_le(meta + 0u, state->step);
  put_u64_le(meta + 8u, state->cumulative_tokens);
  put_u64_le(meta + 16u, state->cursor.shard_index);
  put_u64_le(meta + 24u, state->cursor.token_offset);
  put_u64_le(meta + 32u, state->cursor.epoch);
  put_u64_le(meta + 40u, state->rng_state[0]);
  put_u64_le(meta + 48u, state->rng_state[1]);
  put_u64_le(meta + 56u, state->rng_state[2]);
  put_u64_le(meta + 64u, state->rng_state[3]);
  sha256_update(&ctx, meta, sizeof(meta));

  n = (size_t)state->param_count;
  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(word, state->params[i]);
    sha256_update(&ctx, word, 4u);
  }
  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(word, state->m[i]);
    sha256_update(&ctx, word, 4u);
  }
  for (size_t i = 0u; i < n; ++i) {
    put_f32_le(word, state->v[i]);
    sha256_update(&ctx, word, 4u);
  }
  sha256_final(&ctx, digest);
  digest_to_hex(digest, out_hex);
  return ET_MOD5_ADAMW_OK;
}

int32_t et_mod5_adamw_aot_file_sha256_hex(const char *path, char *out_hex,
                                          int64_t capacity) {
  uint8_t digest[32];
  et_mod5_adamw_status st;
  if (path == NULL || out_hex == NULL || capacity < 65) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = et_mod5_adamw_sha256_file(path, digest);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  digest_to_hex(digest, out_hex);
  return ET_MOD5_ADAMW_OK;
}

/* --- Validated D1 v1.0 .ets token shard write/read helpers --- */

int32_t et_mod5_adamw_aot_write_d1_shard(const char *path, int64_t shard_index,
                                         int64_t vocab_size,
                                         const char *tokenizer_id,
                                         const int64_t *tokens,
                                         int64_t token_count) {
  uint32_t fp_len = 0u;
  uint32_t header_bytes;
  uint64_t payload_bytes;
  uint64_t file_bytes;
  uint8_t *buf = NULL;
  et_sha256_ctx sha;
  et_mod5_adamw_status st;

  if (path == NULL || shard_index < 0 || vocab_size <= 0 || tokens == NULL ||
      token_count <= 0 || (uint64_t)token_count > ET_MOD5_D1_MAX_TOKENS) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = validate_tokenizer_id(tokenizer_id, &fp_len);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  for (int64_t i = 0; i < token_count; ++i) {
    if (tokens[i] < 0 || tokens[i] >= vocab_size) {
      return ET_MOD5_ADAMW_ERR_INVALID_ARG;
    }
  }

  header_bytes = ET_MOD5_D1_SHARD_FIXED + fp_len;
  payload_bytes = (uint64_t)token_count * UINT64_C(8);
  file_bytes = (uint64_t)header_bytes + payload_bytes + UINT64_C(32);

  buf = (uint8_t *)calloc(1u, (size_t)file_bytes);
  if (buf == NULL) {
    return ET_MOD5_ADAMW_ERR_ALLOC;
  }

  memcpy(buf + 0u, ET_MOD5_D1_SHARD_MAGIC, 8u);
  put_u16_le(buf + 8u, 1u);  /* major */
  put_u16_le(buf + 10u, 0u); /* minor */
  put_u32_le(buf + 12u, header_bytes);
  put_u32_le(buf + 16u, 0u); /* required_features */
  put_u32_le(buf + 20u, 1u); /* token_encoding = u64-le */
  put_u32_le(buf + 24u, 1u); /* checksum = sha256 */
  put_u32_le(buf + 28u, 0u); /* reserved */
  put_u64_le(buf + 32u, (uint64_t)shard_index);
  put_u64_le(buf + 40u, (uint64_t)token_count);
  put_u64_le(buf + 48u, payload_bytes);
  put_u64_le(buf + 56u, (uint64_t)vocab_size);
  put_u32_le(buf + 64u, fp_len);
  put_u32_le(buf + 68u, 0u); /* reserved */
  put_u64_le(buf + 72u, file_bytes);
  memcpy(buf + ET_MOD5_D1_SHARD_FIXED, tokenizer_id, (size_t)fp_len);

  for (int64_t i = 0; i < token_count; ++i) {
    put_u64_le(buf + header_bytes + (size_t)i * 8u, (uint64_t)tokens[i]);
  }

  sha256_init(&sha);
  sha256_update(&sha, buf, (size_t)(file_bytes - UINT64_C(32)));
  sha256_final(&sha, buf + file_bytes - UINT64_C(32));

  st = atomic_write_file(path, buf, (size_t)file_bytes);
  free(buf);
  return st;
}

int32_t et_mod5_adamw_aot_read_d1_shard(
    const char *path, int64_t expected_shard_index, int64_t expected_vocab_size,
    const char *expected_tokenizer_id, int64_t *out_tokens, int64_t max_tokens,
    int64_t *out_token_count) {
  uint8_t *buf = NULL;
  size_t file_size = 0u;
  uint32_t expected_fp_len = 0u;
  uint32_t header_bytes;
  uint32_t fp_len;
  uint64_t shard_index;
  uint64_t token_count;
  uint64_t payload_bytes;
  uint64_t vocab_size;
  uint64_t declared_file_bytes;
  uint8_t computed[32];
  et_sha256_ctx sha;
  et_mod5_adamw_status st;

  if (path == NULL || expected_shard_index < 0 || expected_vocab_size <= 0 ||
      out_tokens == NULL || max_tokens <= 0 || out_token_count == NULL) {
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }
  st = validate_tokenizer_id(expected_tokenizer_id, &expected_fp_len);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }

  st = read_regular_nofollow(path, ET_MOD5_SNAPSHOT_MAX_BYTES, &buf,
                             &file_size);
  if (st != ET_MOD5_ADAMW_OK) {
    return st;
  }
  if (file_size < ET_MOD5_D1_SHARD_FIXED + 32u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }
  if (memcmp(buf, ET_MOD5_D1_SHARD_MAGIC, 8u) != 0) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_BAD_MAGIC;
  }
  if (get_u16_le(buf + 8u) != 1u || get_u16_le(buf + 10u) != 0u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_VERSION_MISMATCH;
  }
  if (get_u32_le(buf + 16u) != 0u || get_u32_le(buf + 20u) != 1u ||
      get_u32_le(buf + 24u) != 1u || get_u32_le(buf + 28u) != 0u ||
      get_u32_le(buf + 68u) != 0u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }

  header_bytes = get_u32_le(buf + 12u);
  shard_index = get_u64_le(buf + 32u);
  token_count = get_u64_le(buf + 40u);
  payload_bytes = get_u64_le(buf + 48u);
  vocab_size = get_u64_le(buf + 56u);
  fp_len = get_u32_le(buf + 64u);
  declared_file_bytes = get_u64_le(buf + 72u);

  if (fp_len > ET_MOD5_ADAMW_MAX_TOKENIZER_ID_BYTES ||
      header_bytes != ET_MOD5_D1_SHARD_FIXED + fp_len ||
      token_count == 0u || token_count > ET_MOD5_D1_MAX_TOKENS ||
      payload_bytes != token_count * UINT64_C(8) ||
      declared_file_bytes != (uint64_t)file_size ||
      (uint64_t)file_size != (uint64_t)header_bytes + payload_bytes + 32u) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_TRUNCATED;
  }

  sha256_init(&sha);
  sha256_update(&sha, buf, file_size - 32u);
  sha256_final(&sha, computed);
  if (!ct_equal_32(computed, buf + file_size - 32u)) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_CHECKSUM_MISMATCH;
  }

  if (shard_index != (uint64_t)expected_shard_index ||
      vocab_size != (uint64_t)expected_vocab_size ||
      fp_len != expected_fp_len ||
      memcmp(buf + ET_MOD5_D1_SHARD_FIXED, expected_tokenizer_id,
             (size_t)fp_len) != 0) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_CONFIG_MISMATCH;
  }
  if (token_count > (uint64_t)max_tokens) {
    free(buf);
    return ET_MOD5_ADAMW_ERR_INVALID_ARG;
  }

  for (uint64_t i = 0u; i < token_count; ++i) {
    const uint64_t tok = get_u64_le(buf + header_bytes + (size_t)i * 8u);
    if (tok >= vocab_size) {
      free(buf);
      return ET_MOD5_ADAMW_ERR_INVALID_ARG;
    }
    out_tokens[i] = (int64_t)tok;
  }
  *out_token_count = (int64_t)token_count;
  free(buf);
  return ET_MOD5_ADAMW_OK;
}
