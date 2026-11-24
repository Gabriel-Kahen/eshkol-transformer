#define _POSIX_C_SOURCE 200809L

#include "wave4_composed_eshkol_bridge.h"

#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#pragma STDC FP_CONTRACT OFF

enum {
  WAVE4_X = 8,
  WAVE4_PER_LAYER = 104,
  WAVE4_MAX_P = 8 * WAVE4_PER_LAYER,
  HANDSHAKE_MAGIC = 0x57344331u
};

struct et_wave4_composed_aot_context {
  et_wave4_composed_config_v1 config;
  et_wave4_composed_scaler_config_v1 scaler_cfg;
  et_wave4_composed_memory_v1 plan;
  et_wave4_composed_state_v1 state;
  et_wave4_composed_batch_v1 batch;
  et_wave4_composed_workspace_v1 workspace;
  et_wave4_composed_result_v1 last_result;
  et_eval5_manifest last_manifest;
  et_dist5_comm_t *comm;
  float *master;
  float *momentum;
  float *input;
  float *target;
  float *overflow_target;
  uint32_t rank;
  uint32_t world;
  int timeout_ms;
  int overflow_inject;
  int bootstrapped;
  int has_step;
  int has_eval;
  int32_t last_status;
};

/* Standalone SHA-256 core matching c2_checkpoint_format.c / eval5_benchmark.c. */
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
  for (size_t i = 0; i < 16u; ++i) {
    const size_t j = i * 4u;
    w[i] = ((uint32_t)block[j] << 24u) | ((uint32_t)block[j + 1u] << 16u) |
           ((uint32_t)block[j + 2u] << 8u) | (uint32_t)block[j + 3u];
  }
  for (size_t i = 16u; i < 64u; ++i) {
    const uint32_t s0 =
        rotr32(w[i - 15u], 7u) ^ rotr32(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
    const uint32_t s1 =
        rotr32(w[i - 2u], 17u) ^ rotr32(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
    w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
  }
  uint32_t a = state->h[0], b = state->h[1], c = state->h[2], d = state->h[3];
  uint32_t e = state->h[4], f = state->h[5], g = state->h[6], h = state->h[7];
  for (size_t i = 0; i < 64u; ++i) {
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

static void sha256_compute(const void *data, size_t size,
                           uint8_t digest[ET_EVAL5_SHA256_BYTES]) {
  static const uint32_t initial[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                                      0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                                      0x1f83d9abu, 0x5be0cd19u};
  sha256_state state;
  memcpy(state.h, initial, sizeof(initial));
  state.length = (uint64_t)size;
  state.used = 0u;

  const uint8_t *bytes = (const uint8_t *)data;
  while (size != 0u) {
    size_t take = 64u - state.used;
    if (take > size)
      take = size;
    memcpy(state.block + state.used, bytes, take);
    state.used += take;
    bytes += take;
    size -= take;
    if (state.used == 64u) {
      sha256_block(&state, state.block);
      state.used = 0u;
    }
  }

  const uint64_t bits = state.length * UINT64_C(8);
  state.block[state.used++] = 0x80u;
  if (state.used > 56u) {
    memset(state.block + state.used, 0, 64u - state.used);
    sha256_block(&state, state.block);
    state.used = 0u;
  }
  memset(state.block + state.used, 0, 56u - state.used);
  for (size_t i = 0; i < 8u; ++i)
    state.block[63u - i] = (uint8_t)(bits >> (8u * i));
  sha256_block(&state, state.block);
  for (size_t i = 0; i < 8u; ++i) {
    digest[i * 4u] = (uint8_t)(state.h[i] >> 24u);
    digest[i * 4u + 1u] = (uint8_t)(state.h[i] >> 16u);
    digest[i * 4u + 2u] = (uint8_t)(state.h[i] >> 8u);
    digest[i * 4u + 3u] = (uint8_t)state.h[i];
  }
}

static void encode_hex(const uint8_t digest[ET_EVAL5_SHA256_BYTES],
                       char out[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES]) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < ET_EVAL5_SHA256_BYTES; ++i) {
    out[i * 2u] = digits[(digest[i] >> 4u) & 0xfu];
    out[i * 2u + 1u] = digits[digest[i] & 0xfu];
  }
  out[ET_EVAL5_SHA256_BYTES * 2u] = '\0';
}

static uint64_t f64_bits(double value) {
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static uint32_t f32_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int is_nan_f64(double value) {
  const uint64_t bits = f64_bits(value);
  return (bits & UINT64_C(0x7ff0000000000000)) ==
             UINT64_C(0x7ff0000000000000) &&
         (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

static int is_finite_f64(double value) {
  return (f64_bits(value) & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}

static float f32_from_f64(double value) {
  if (is_nan_f64(value))
    return (float)NAN;
  if (value > (double)FLT_MAX)
    return (float)INFINITY;
  if (value < -(double)FLT_MAX)
    return -(float)INFINITY;
  return (float)value;
}

static int valid_index(int64_t index, size_t count) {
  return index >= 0 && (uint64_t)index < (uint64_t)count;
}

static int invertible_scale(float scale) {
  const float inverse = 1.0f / scale;
  return isfinite(inverse) && inverse > 0.0f;
}

static void put_u32_le(uint8_t out[4], uint32_t value) {
  out[0] = (uint8_t)(value & 0xffu);
  out[1] = (uint8_t)((value >> 8u) & 0xffu);
  out[2] = (uint8_t)((value >> 16u) & 0xffu);
  out[3] = (uint8_t)((value >> 24u) & 0xffu);
}

static uint32_t get_u32_le(const uint8_t in[4]) {
  return (uint32_t)in[0] | ((uint32_t)in[1] << 8u) | ((uint32_t)in[2] << 16u) |
         ((uint32_t)in[3] << 24u);
}

static int64_t now_ms(void) {
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return -1;
  return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int remaining_ms(int64_t deadline) {
  const int64_t now = now_ms();
  if (now < 0 || now >= deadline)
    return 0;
  const int64_t diff = deadline - now;
  return diff > (int64_t)INT_MAX ? INT_MAX : (int)diff;
}

static int32_t dist_err(et_dist5_status_t status) {
  return status == ET_DIST5_OK
             ? (int32_t)ET_WAVE4_COMPOSED_OK
             : (int32_t)(ET_WAVE4_COMPOSED_DIST5_ERROR_BASE + status);
}

static int32_t send_all(int fd, const uint8_t *buf, size_t len,
                        int64_t deadline) {
  size_t done = 0u;
  while (done < len) {
    const int rem = remaining_ms(deadline);
    if (rem <= 0)
      return dist_err(ET_DIST5_ERR_TIMEOUT);
    struct pollfd pfd = {.fd = fd, .events = POLLOUT, .revents = 0};
    const int pr = poll(&pfd, 1u, rem);
    if (pr < 0) {
      if (errno == EINTR)
        continue;
      return dist_err(ET_DIST5_ERR_IO);
    }
    if (pr == 0)
      return dist_err(ET_DIST5_ERR_TIMEOUT);
    const ssize_t n = send(fd, buf + done, len - done, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return dist_err(ET_DIST5_ERR_PEER_DISCONNECTED);
    }
    if (n == 0)
      return dist_err(ET_DIST5_ERR_PEER_DISCONNECTED);
    done += (size_t)n;
  }
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

static int32_t recv_all(int fd, uint8_t *buf, size_t len, int64_t deadline) {
  size_t done = 0u;
  while (done < len) {
    const int rem = remaining_ms(deadline);
    if (rem <= 0)
      return dist_err(ET_DIST5_ERR_TIMEOUT);
    struct pollfd pfd = {.fd = fd, .events = POLLIN, .revents = 0};
    const int pr = poll(&pfd, 1u, rem);
    if (pr < 0) {
      if (errno == EINTR)
        continue;
      return dist_err(ET_DIST5_ERR_IO);
    }
    if (pr == 0)
      return dist_err(ET_DIST5_ERR_TIMEOUT);
    const ssize_t n = recv(fd, buf + done, len - done, 0);
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
      return dist_err(ET_DIST5_ERR_PEER_DISCONNECTED);
    }
    if (n == 0)
      return dist_err(ET_DIST5_ERR_PEER_DISCONNECTED);
    done += (size_t)n;
  }
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

static int32_t connect_comm(const char *sock_dir, uint32_t rank, uint32_t world,
                            int timeout_ms, et_dist5_comm_t **out_comm) {
  if (world == 1u) {
    return dist_err(et_dist5_comm_create(ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR,
                                         0u, 1u, NULL, 0u, out_comm));
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  const int path_len =
      snprintf(addr.sun_path, sizeof(addr.sun_path), "%s/rank0.sock", sock_dir);
  if (path_len <= 0 || (size_t)path_len >= sizeof(addr.sun_path))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;

  const int64_t start = now_ms();
  if (start < 0)
    return dist_err(ET_DIST5_ERR_IO);
  const int64_t deadline = start + (int64_t)timeout_ms;

  if (rank == 0u) {
    int peer_fds[ET_DIST5_MAX_WORLD_SIZE];
    for (uint32_t i = 0u; i < world; ++i)
      peer_fds[i] = -1;

    const int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0)
      return dist_err(ET_DIST5_ERR_IO);

    (void)unlink(addr.sun_path);
    int32_t status = (int32_t)ET_WAVE4_COMPOSED_OK;
    if (bind(listen_fd, (const struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, (int)world) != 0) {
      status = dist_err(ET_DIST5_ERR_IO);
    }

    uint32_t connected = 0u;
    const uint32_t needed = world - 1u;
    while (status == (int32_t)ET_WAVE4_COMPOSED_OK && connected < needed) {
      const int rem = remaining_ms(deadline);
      if (rem <= 0) {
        status = dist_err(ET_DIST5_ERR_TIMEOUT);
        break;
      }
      struct pollfd pfd = {.fd = listen_fd, .events = POLLIN, .revents = 0};
      const int pr = poll(&pfd, 1u, rem);
      if (pr < 0) {
        if (errno == EINTR)
          continue;
        status = dist_err(ET_DIST5_ERR_IO);
        break;
      }
      if (pr == 0) {
        status = dist_err(ET_DIST5_ERR_TIMEOUT);
        break;
      }
      const int fd = accept(listen_fd, NULL, NULL);
      if (fd < 0) {
        if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
          continue;
        status = dist_err(ET_DIST5_ERR_IO);
        break;
      }

      uint8_t hello[8];
      status = recv_all(fd, hello, sizeof(hello), deadline);
      if (status != (int32_t)ET_WAVE4_COMPOSED_OK) {
        close(fd);
        break;
      }
      const uint32_t magic = get_u32_le(hello);
      const uint32_t peer_rank = get_u32_le(hello + 4u);
      if (magic != HANDSHAKE_MAGIC || peer_rank == 0u || peer_rank >= world ||
          peer_fds[peer_rank - 1u] != -1) {
        close(fd);
        status = dist_err(ET_DIST5_ERR_MISMATCH);
        break;
      }
      uint8_t ack[4];
      put_u32_le(ack, HANDSHAKE_MAGIC);
      status = send_all(fd, ack, sizeof(ack), deadline);
      if (status != (int32_t)ET_WAVE4_COMPOSED_OK) {
        close(fd);
        break;
      }
      peer_fds[peer_rank - 1u] = fd;
      ++connected;
    }

    close(listen_fd);
    (void)unlink(addr.sun_path);

    if (status == (int32_t)ET_WAVE4_COMPOSED_OK) {
      status = dist_err(et_dist5_comm_create(
          ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, 0u, world, peer_fds,
          (size_t)needed, out_comm));
    }
    for (uint32_t i = 0u; i < needed; ++i) {
      if (peer_fds[i] >= 0)
        close(peer_fds[i]);
    }
    return status;
  }

  /* Non-zero rank: retry connect until rank 0 binds and listens. */
  int fd = -1;
  for (;;) {
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0)
      return dist_err(ET_DIST5_ERR_IO);
    if (connect(fd, (const struct sockaddr *)&addr, sizeof(addr)) == 0)
      break;
    const int err = errno;
    close(fd);
    fd = -1;
    if (err != ENOENT && err != ECONNREFUSED && err != EAGAIN && err != EINTR)
      return dist_err(ET_DIST5_ERR_IO);
    if (remaining_ms(deadline) <= 0)
      return dist_err(ET_DIST5_ERR_TIMEOUT);
    const struct timespec pause = {.tv_sec = 0, .tv_nsec = 5000000L};
    nanosleep(&pause, NULL);
  }

  uint8_t hello[8];
  put_u32_le(hello, HANDSHAKE_MAGIC);
  put_u32_le(hello + 4u, rank);
  int32_t status = send_all(fd, hello, sizeof(hello), deadline);
  if (status == (int32_t)ET_WAVE4_COMPOSED_OK) {
    uint8_t ack[4];
    status = recv_all(fd, ack, sizeof(ack), deadline);
    if (status == (int32_t)ET_WAVE4_COMPOSED_OK &&
        get_u32_le(ack) != HANDSHAKE_MAGIC)
      status = dist_err(ET_DIST5_ERR_MISMATCH);
  }
  if (status == (int32_t)ET_WAVE4_COMPOSED_OK) {
    status = dist_err(et_dist5_comm_create(
        ET_DIST5_ABI_MAJOR, ET_DIST5_ABI_MINOR, rank, world, &fd, 1u,
        out_comm));
  }
  close(fd);
  return status;
}

static float reference_bf16(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  bits = (bits + UINT32_C(0x7fff) + ((bits >> 16u) & 1u)) &
         UINT32_C(0xffff0000);
  memcpy(&value, &bits, sizeof(bits));
  return value;
}

static int32_t synthesize_overflow_target(et_wave4_composed_aot_context *ctx,
                                          uint64_t step) {
  for (size_t i = 0u; i < WAVE4_X; ++i)
    ctx->overflow_target[i] = 0x1p20f;
  if (ctx->config.dtype == ET_WAVE4_COMPOSED_F16)
    return (int32_t)ET_WAVE4_COMPOSED_OK;

  /* BF16 has binary32 exponent range: probe the finite gradient amplitude on
   * BF16-rounded master weights and scale the target so grad * scale lands in
   * [0x1.ffp127f, FLT_MAX] (finite in f32, overflows BF16 RNE cast). */
  const size_t p = ctx->plan.ac4.parameter_floats;
  if (p > WAVE4_MAX_P || ctx->plan.ac4.activation_floats != WAVE4_X)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;

  float probe_weights[WAVE4_MAX_P];
  float probe_grads[WAVE4_MAX_P];
  float probe_output[WAVE4_X];
  float probe_d_input[WAVE4_X];
  for (size_t i = 0u; i < p; ++i)
    probe_weights[i] = reference_bf16(ctx->master[i]);

  et_ac4_config_v1 ac4 = ctx->config.ac4;
  ac4.step = step;
  const et_ac4_io_v1 io = {{ctx->input, WAVE4_X},
                           {ctx->overflow_target, WAVE4_X},
                           {probe_weights, p},
                           {probe_output, WAVE4_X},
                           {probe_d_input, WAVE4_X},
                           {probe_grads, p},
                           (float *)ctx->workspace.data,
                           ctx->plan.ac4.workspace_bytes};
  et_ac4_result_v1 probe_result;
  if (et_ac4_run_v1(&ac4, &io, &probe_result) != ET_AC4_OK)
    return (int32_t)ET_WAVE4_COMPOSED_NUMERIC_ERROR;

  float largest = 0.0f;
  for (size_t i = 0u; i < p; ++i)
    largest = fmaxf(largest, fabsf(probe_grads[i]));
  if (!(largest > 0.0f) || !isfinite(largest))
    return (int32_t)ET_WAVE4_COMPOSED_NUMERIC_ERROR;

  const double desired = 0x1.ff8p127 / (double)ctx->state.scaler.scale;
  const double amplitude =
      desired / (double)largest * (double)ctx->overflow_target[0];
  if (!isfinite(amplitude) || amplitude >= (double)FLT_MAX)
    return (int32_t)ET_WAVE4_COMPOSED_NUMERIC_ERROR;
  for (size_t i = 0u; i < WAVE4_X; ++i)
    ctx->overflow_target[i] = (float)amplitude;

  if (et_ac4_run_v1(&ac4, &io, &probe_result) != ET_AC4_OK)
    return (int32_t)ET_WAVE4_COMPOSED_NUMERIC_ERROR;
  float verify = 0.0f;
  for (size_t i = 0u; i < p; ++i)
    verify = fmaxf(verify, fabsf(probe_grads[i]));
  const float scaled = verify * ctx->state.scaler.scale;
  if (!isfinite(scaled) || scaled < 0x1.ffp127f)
    return (int32_t)ET_WAVE4_COMPOSED_NUMERIC_ERROR;
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_create(
    const char *sock_dir, int64_t rank, int64_t world, int64_t timeout_ms,
    int64_t layers, int64_t dtype, int64_t policy, int64_t checkpoint_every,
    double epsilon, double dropout_rate, int64_t seed, double learning_rate,
    double momentum, double init_scale, double growth_factor,
    double backoff_factor, double min_scale, double max_scale,
    int64_t growth_interval, et_wave4_composed_aot_context **out_ctx) {
  if (out_ctx == NULL)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (world < 1 || world > (int64_t)ET_DIST5_MAX_WORLD_SIZE || rank < 0 ||
      rank >= world || timeout_ms <= 0 || timeout_ms > (int64_t)INT_MAX)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (world > 1 && (sock_dir == NULL || sock_dir[0] == '\0'))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if ((layers != 4 && layers != 8) ||
      (dtype != (int64_t)ET_WAVE4_COMPOSED_F16 &&
       dtype != (int64_t)ET_WAVE4_COMPOSED_BF16) ||
      policy < (int64_t)ET_AC4_POLICY_NONE ||
      policy > (int64_t)ET_AC4_POLICY_EVERY_K || checkpoint_every < 0 ||
      checkpoint_every > (int64_t)UINT32_MAX || growth_interval < 1 ||
      growth_interval > (int64_t)UINT32_MAX)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (!is_finite_f64(epsilon) || !is_finite_f64(dropout_rate) ||
      !is_finite_f64(learning_rate) || !is_finite_f64(momentum) ||
      !is_finite_f64(init_scale) || !is_finite_f64(growth_factor) ||
      !is_finite_f64(backoff_factor) || !is_finite_f64(min_scale) ||
      !is_finite_f64(max_scale))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;

  const et_wave4_composed_config_v1 cfg = {
      .ac4 = {.shape = {.layers = (uint32_t)layers,
                        .batch = 1u,
                        .tokens = 2u,
                        .query_heads = 2u,
                        .kv_heads = 1u,
                        .head_dim = 2u},
              .mode = ET_AC4_FORWARD_BACKWARD,
              .policy = (et_ac4_policy)policy,
              .checkpoint_every = (uint32_t)checkpoint_every,
              .epsilon = f32_from_f64(epsilon),
              .dropout_rate = f32_from_f64(dropout_rate),
              .seed = (uint64_t)seed,
              .step = 0u,
              .activation_limit_bytes = 0u},
      .dtype = (et_wave4_composed_dtype)dtype,
      .learning_rate = f32_from_f64(learning_rate),
      .momentum = f32_from_f64(momentum)};

  et_wave4_composed_memory_v1 plan;
  const et_wave4_composed_status plan_status =
      et_wave4_composed_plan_v1(&cfg, &plan);
  if (plan_status != ET_WAVE4_COMPOSED_OK)
    return (int32_t)plan_status;

  const et_wave4_composed_scaler_config_v1 scfg = {
      .scale = f32_from_f64(init_scale),
      .growth_factor = f32_from_f64(growth_factor),
      .backoff_factor = f32_from_f64(backoff_factor),
      .min_scale = f32_from_f64(min_scale),
      .max_scale = f32_from_f64(max_scale),
      .growth_interval = (uint32_t)growth_interval};

  et_mp4_loss_scaler_v1 checked_scaler;
  if (et_mp4_loss_scaler_init(&checked_scaler, scfg.scale, scfg.growth_factor,
                              scfg.backoff_factor, scfg.growth_interval,
                              scfg.min_scale, scfg.max_scale) !=
          ET_MP4_LOSS_SCALER_STEP_ACCEPTED ||
      !invertible_scale(checked_scaler.min_scale))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;

  if (world > 1) {
    struct sockaddr_un probe;
    const size_t dir_len = strlen(sock_dir);
    if (dir_len + sizeof("/rank0.sock") > sizeof(probe.sun_path))
      return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  }

  et_wave4_composed_aot_context *ctx =
      (et_wave4_composed_aot_context *)calloc(1u, sizeof(*ctx));
  if (ctx == NULL)
    return ET_WAVE4_COMPOSED_AOT_OUT_OF_MEMORY;

  const size_t p = plan.ac4.parameter_floats;
  const size_t x = plan.ac4.activation_floats;
  ctx->config = cfg;
  ctx->scaler_cfg = scfg;
  ctx->plan = plan;
  ctx->rank = (uint32_t)rank;
  ctx->world = (uint32_t)world;
  ctx->timeout_ms = (int)timeout_ms;
  ctx->last_status = (int32_t)ET_WAVE4_COMPOSED_OK;

  ctx->master = (float *)calloc(p, sizeof(float));
  ctx->momentum = (float *)calloc(p, sizeof(float));
  ctx->input = (float *)calloc(x, sizeof(float));
  ctx->target = (float *)calloc(x, sizeof(float));
  ctx->overflow_target = (float *)calloc(x, sizeof(float));
  /* Allocate workspace with float alignment (calloc guarantees max_align_t). */
  ctx->workspace.data = calloc(1u, plan.workspace_bytes);
  ctx->workspace.bytes = plan.workspace_bytes;

  if (ctx->master == NULL || ctx->momentum == NULL || ctx->input == NULL ||
      ctx->target == NULL || ctx->overflow_target == NULL ||
      ctx->workspace.data == NULL) {
    et_wave4_composed_aot_destroy(ctx);
    return ET_WAVE4_COMPOSED_AOT_OUT_OF_MEMORY;
  }

  ctx->state.parameter_count = p;
  ctx->state.master = ctx->master;
  ctx->state.momentum = ctx->momentum;
  ctx->batch =
      (et_wave4_composed_batch_v1){{ctx->input, x}, {ctx->target, x}};

  const int32_t comm_status =
      connect_comm(sock_dir, ctx->rank, ctx->world, ctx->timeout_ms, &ctx->comm);
  if (comm_status != (int32_t)ET_WAVE4_COMPOSED_OK) {
    et_wave4_composed_aot_destroy(ctx);
    return comm_status;
  }

  *out_ctx = ctx;
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

void et_wave4_composed_aot_destroy(et_wave4_composed_aot_context *ctx) {
  if (ctx == NULL)
    return;
  if (ctx->comm != NULL)
    et_dist5_comm_destroy(ctx->comm);
  free(ctx->master);
  free(ctx->momentum);
  free(ctx->input);
  free(ctx->target);
  free(ctx->overflow_target);
  free(ctx->workspace.data);
  free(ctx);
}

int32_t et_wave4_composed_aot_set_master(et_wave4_composed_aot_context *ctx,
                                         int64_t index, double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.ac4.parameter_floats))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  ctx->master[index] = f32_from_f64(value);
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_set_input(et_wave4_composed_aot_context *ctx,
                                        int64_t index, double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.ac4.activation_floats))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  ctx->input[index] = f32_from_f64(value);
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_set_target(et_wave4_composed_aot_context *ctx,
                                         int64_t index, double value) {
  if (ctx == NULL || !valid_index(index, ctx->plan.ac4.activation_floats))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  ctx->target[index] = f32_from_f64(value);
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_set_overflow_inject(
    et_wave4_composed_aot_context *ctx, int64_t enabled) {
  if (ctx == NULL || (enabled != 0 && enabled != 1))
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  ctx->overflow_inject = (int)enabled;
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_set_activation_limit(
    et_wave4_composed_aot_context *ctx, int64_t limit_bytes) {
  if (ctx == NULL || limit_bytes < 0 ||
      (uint64_t)limit_bytes > (uint64_t)SIZE_MAX)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  ctx->config.ac4.activation_limit_bytes = (size_t)limit_bytes;
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int32_t et_wave4_composed_aot_bootstrap(et_wave4_composed_aot_context *ctx) {
  if (ctx == NULL)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  const et_wave4_composed_status status = et_wave4_composed_bootstrap_v1(
      &ctx->config, ctx->comm, ctx->timeout_ms, &ctx->scaler_cfg, &ctx->state,
      ctx->workspace);
  ctx->last_status = (int32_t)status;
  if (status == ET_WAVE4_COMPOSED_OK)
    ctx->bootstrapped = 1;
  return (int32_t)status;
}

int32_t et_wave4_composed_aot_step(et_wave4_composed_aot_context *ctx,
                                   int64_t step) {
  if (ctx == NULL || step < 0)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (!ctx->bootstrapped) {
    ctx->last_status = ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED;
    return ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED;
  }

  et_wave4_composed_batch_v1 batch = ctx->batch;
  if (ctx->overflow_inject) {
    const int32_t prep = synthesize_overflow_target(ctx, (uint64_t)step);
    if (prep != (int32_t)ET_WAVE4_COMPOSED_OK) {
      ctx->last_status = prep;
      return prep;
    }
    batch.target.data = ctx->overflow_target;
  }

  et_wave4_composed_config_v1 cfg = ctx->config;
  cfg.ac4.step = (uint64_t)step;
  et_wave4_composed_result_v1 result;
  const et_wave4_composed_status status = et_wave4_composed_step_v1(
      &cfg, ctx->comm, ctx->timeout_ms, &ctx->state, &batch, ctx->workspace,
      &result);
  ctx->last_status = (int32_t)status;
  if (status == ET_WAVE4_COMPOSED_OK ||
      status == ET_WAVE4_COMPOSED_OVERFLOW_BACKOFF) {
    ctx->config.ac4.step = cfg.ac4.step;
    ctx->last_result = result;
    ctx->has_step = 1;
  }
  return (int32_t)status;
}

int32_t et_wave4_composed_aot_evaluate(et_wave4_composed_aot_context *ctx,
                                       const char *eval_path) {
  if (ctx == NULL || eval_path == NULL || eval_path[0] == '\0')
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (!ctx->bootstrapped) {
    ctx->last_status = ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED;
    return ET_WAVE4_COMPOSED_AOT_NOT_BOOTSTRAPPED;
  }

  static const uint8_t train_doc[] = "red fox";
  static const uint8_t held_out_doc[] = "sky blue";
  static const uint8_t vocabulary[] =
      "four fixture classes: red green blue black";
  static const uint32_t target_ids[] = {0u, 2u};
  static const float eval_input[WAVE4_X] = {0.35f,  -0.55f, 0.41f,  0.28f,
                                            -0.21f, 0.49f,  -0.22f, 0.13f};
  static const float eval_target[WAVE4_X] = {0.0f};

  et_eval5_manifest identity = {
      .version_major = ET_EVAL5_VERSION_MAJOR,
      .version_minor = ET_EVAL5_VERSION_MINOR,
      .suite = "wave4-composed",
      .task = "fixture-logits",
      .split = "held-out",
      .max_samples = 1u,
      .contamination = {ET_EVAL5_NGRAM_BYTES, 2u, 1u, 0.5, 0.0}};

  sha256_compute(train_doc, sizeof(train_doc) - 1u,
                 identity.train_dataset_sha256);
  sha256_compute(held_out_doc, sizeof(held_out_doc) - 1u,
                 identity.eval_dataset_sha256);
  sha256_compute(vocabulary, sizeof(vocabulary) - 1u,
                 identity.tokenizer_sha256);
  sha256_compute(ctx->master, ctx->state.parameter_count * sizeof(float),
                 identity.checkpoint_sha256);

  const et_wave4_composed_eval_v1 eval = {
      {{eval_input, WAVE4_X}, {eval_target, WAVE4_X}},
      {train_doc, sizeof(train_doc) - 1u, NULL, 0u},
      {held_out_doc, sizeof(held_out_doc) - 1u, NULL, 0u},
      target_ids,
      sizeof(target_ids) / sizeof(target_ids[0]),
      eval_path,
      &identity};

  et_eval5_manifest verified;
  const et_wave4_composed_status status = et_wave4_composed_evaluate_v1(
      &ctx->config, &ctx->state, &eval, ctx->workspace, &verified);
  ctx->last_status = (int32_t)status;
  if (status == ET_WAVE4_COMPOSED_OK) {
    ctx->last_manifest = verified;
    ctx->has_eval = 1;
  }
  return (int32_t)status;
}

int32_t et_wave4_composed_aot_verify_manifest(
    const et_wave4_composed_aot_context *ctx, const char *eval_path) {
  if (ctx == NULL || !ctx->has_eval || eval_path == NULL ||
      eval_path[0] == '\0')
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;

  et_eval5_manifest reloaded;
  const et_eval5_status rc = et_eval5_manifest_read_file(eval_path, &reloaded);
  if (rc != ET_EVAL5_OK)
    return (int32_t)(ET_WAVE4_COMPOSED_EVAL5_ERROR_BASE - rc);

  if (memcmp(reloaded.manifest_sha256, ctx->last_manifest.manifest_sha256,
             ET_EVAL5_SHA256_BYTES) != 0 ||
      memcmp(reloaded.checkpoint_sha256, ctx->last_manifest.checkpoint_sha256,
             ET_EVAL5_SHA256_BYTES) != 0 ||
      reloaded.lm.token_count != ctx->last_manifest.lm.token_count ||
      reloaded.lm.byte_count != ctx->last_manifest.lm.byte_count ||
      reloaded.audit.contaminated_documents !=
          ctx->last_manifest.audit.contaminated_documents ||
      memcmp(&reloaded.lm.mean_nll, &ctx->last_manifest.lm.mean_nll,
             sizeof(double)) != 0 ||
      memcmp(&reloaded.lm.perplexity, &ctx->last_manifest.lm.perplexity,
             sizeof(double)) != 0)
    return (int32_t)(ET_WAVE4_COMPOSED_EVAL5_ERROR_BASE -
                     ET_EVAL5_ERR_CHECKSUM);

  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int64_t et_wave4_composed_aot_parameter_count(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.ac4.parameter_floats : -1;
}

int64_t et_wave4_composed_aot_activation_count(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.ac4.activation_floats : -1;
}

int64_t et_wave4_composed_aot_planned_activation_peak_bytes(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.ac4.activation_peak_bytes : -1;
}

int64_t et_wave4_composed_aot_planned_workspace_bytes(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.workspace_bytes : -1;
}

int64_t et_wave4_composed_aot_narrow_parameter_bytes(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? (int64_t)ctx->plan.narrow_parameter_bytes : -1;
}

int32_t et_wave4_composed_aot_last_status(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL ? ctx->last_status
                     : (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
}

double et_wave4_composed_aot_last_loss(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_step ? ctx->last_result.loss : (double)NAN;
}

int64_t et_wave4_composed_aot_last_local_overflow(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_step ? (int64_t)ctx->last_result.local_overflow
                                      : -1;
}

int64_t et_wave4_composed_aot_live_activation_peak_bytes(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_step
             ? (int64_t)ctx->last_result.memory.ac4.activation_peak_bytes
             : -1;
}

int64_t et_wave4_composed_aot_workspace_bytes(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_step
             ? (int64_t)ctx->last_result.memory.workspace_bytes
             : -1;
}

double et_wave4_composed_aot_scaler_scale(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->bootstrapped ? (double)ctx->state.scaler.scale
                                          : (double)NAN;
}

int64_t et_wave4_composed_aot_scaler_clean_steps(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->bootstrapped
             ? (int64_t)ctx->state.scaler.clean_steps
             : -1;
}

double et_wave4_composed_aot_get_master(
    const et_wave4_composed_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->plan.ac4.parameter_floats))
    return (double)NAN;
  return (double)ctx->master[index];
}

int64_t et_wave4_composed_aot_master_bits(
    const et_wave4_composed_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !valid_index(index, ctx->plan.ac4.parameter_floats))
    return -1;
  return (int64_t)f32_bits(ctx->master[index]);
}

int32_t et_wave4_composed_aot_param_digest_hex(
    const et_wave4_composed_aot_context *ctx, char *out_buf, int64_t capacity) {
  if (ctx == NULL || out_buf == NULL || capacity < 0)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (capacity < (int64_t)ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES)
    return ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL;
  uint8_t digest[ET_EVAL5_SHA256_BYTES];
  sha256_compute(ctx->master, ctx->state.parameter_count * sizeof(float),
                 digest);
  encode_hex(digest, out_buf);
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int64_t et_wave4_composed_aot_param_digest_byte(
    const et_wave4_composed_aot_context *ctx, int64_t index) {
  if (ctx == NULL || index < 0 || index >= (int64_t)ET_EVAL5_SHA256_BYTES)
    return -1;
  uint8_t digest[ET_EVAL5_SHA256_BYTES];
  sha256_compute(ctx->master, ctx->state.parameter_count * sizeof(float),
                 digest);
  return (int64_t)digest[index];
}

double et_wave4_composed_aot_eval_mean_nll(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_eval ? ctx->last_manifest.lm.mean_nll
                                      : (double)NAN;
}

double et_wave4_composed_aot_eval_perplexity(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_eval ? ctx->last_manifest.lm.perplexity
                                      : (double)NAN;
}

int64_t et_wave4_composed_aot_eval_contaminated_documents(
    const et_wave4_composed_aot_context *ctx) {
  return ctx != NULL && ctx->has_eval
             ? (int64_t)ctx->last_manifest.audit.contaminated_documents
             : -1;
}

int32_t et_wave4_composed_aot_manifest_digest_hex(
    const et_wave4_composed_aot_context *ctx, char *out_buf, int64_t capacity) {
  if (ctx == NULL || !ctx->has_eval || out_buf == NULL || capacity < 0)
    return (int32_t)ET_WAVE4_COMPOSED_INVALID_ARGUMENT;
  if (capacity < (int64_t)ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES)
    return ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL;
  encode_hex(ctx->last_manifest.manifest_sha256, out_buf);
  return (int32_t)ET_WAVE4_COMPOSED_OK;
}

int64_t et_wave4_composed_aot_manifest_digest_byte(
    const et_wave4_composed_aot_context *ctx, int64_t index) {
  if (ctx == NULL || !ctx->has_eval || index < 0 ||
      index >= (int64_t)ET_EVAL5_SHA256_BYTES)
    return -1;
  return (int64_t)ctx->last_manifest.manifest_sha256[index];
}

int64_t et_wave4_composed_aot_read_byte(const char *buffer, int64_t index) {
  if (buffer == NULL || index < 0)
    return -1;
  return (int64_t)(unsigned char)buffer[index];
}
