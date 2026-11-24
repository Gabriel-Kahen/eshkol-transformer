#include "dist5_eshkol_bridge.h"

#include <cstdint>
#include <cstring>
#include <type_traits>

static_assert(std::is_same_v<
              decltype(&et_dist5_aot_open),
              int32_t (*)(const char *, int64_t, int64_t, int64_t,
                          et_dist5_aot_handle_t **)>);
static_assert(std::is_same_v<decltype(&et_dist5_aot_close),
                             void (*)(et_dist5_aot_handle_t *)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_barrier),
              int32_t (*)(et_dist5_aot_handle_t *, int64_t, int64_t)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_broadcast_f32),
              int32_t (*)(et_dist5_aot_handle_t *, void *, int64_t, int64_t,
                          int64_t, int64_t)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_allreduce_f32),
              int32_t (*)(et_dist5_aot_handle_t *, const void *, void *,
                          int64_t, int32_t, int64_t, int64_t)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_allgather_f32),
              int32_t (*)(et_dist5_aot_handle_t *, const void *, int64_t,
                          void *, int64_t, int64_t)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_reduce_scatter_f32),
              int32_t (*)(et_dist5_aot_handle_t *, const void *, void *,
                          int64_t, int32_t, int64_t, int64_t)>);
static_assert(std::is_same_v<
              decltype(&et_dist5_aot_sync_gradients_f32),
              int32_t (*)(et_dist5_aot_handle_t *, const void *, void *,
                          int64_t, int64_t, int64_t)>);
static_assert(std::is_same_v<decltype(&et_dist5_aot_alloc_f32),
                             void *(*)(int64_t)>);
static_assert(std::is_same_v<decltype(&et_dist5_aot_free_f32), void (*)(void *)>);
static_assert(std::is_same_v<decltype(&et_dist5_aot_sha256_hex_f32),
                             int32_t (*)(const void *, int64_t, char *, int64_t)>);

int main() {
  et_dist5_aot_handle_t *handle = nullptr;
  if (et_dist5_aot_open(nullptr, 0, 1, 5000, &handle) != ET_DIST5_OK ||
      handle == nullptr)
    return 1;
  if (et_dist5_aot_rank(handle) != 0 || et_dist5_aot_world_size(handle) != 1) {
    et_dist5_aot_close(handle);
    return 2;
  }

  void *buf = et_dist5_aot_alloc_f32(4);
  void *out = et_dist5_aot_alloc_f32(4);
  if (buf == nullptr || out == nullptr) {
    et_dist5_aot_free_f32(buf);
    et_dist5_aot_free_f32(out);
    et_dist5_aot_close(handle);
    return 3;
  }

  for (int64_t i = 0; i < 4; ++i) {
    if (et_dist5_aot_store_f32(buf, 4, i, static_cast<double>(i + 1)) !=
        ET_DIST5_OK) {
      et_dist5_aot_free_f32(buf);
      et_dist5_aot_free_f32(out);
      et_dist5_aot_close(handle);
      return 4;
    }
  }

  char hex[65] = {};
  const bool ok =
      et_dist5_aot_barrier(handle, 1, 5000) == ET_DIST5_OK &&
      et_dist5_aot_broadcast_f32(handle, buf, 4, 0, 2, 5000) == ET_DIST5_OK &&
      et_dist5_aot_allreduce_f32(handle, buf, out, 4, ET_DIST5_SUM, 3, 5000) ==
          ET_DIST5_OK &&
      et_dist5_aot_allgather_f32(handle, buf, 4, out, 4, 5000) == ET_DIST5_OK &&
      et_dist5_aot_reduce_scatter_f32(handle, buf, out, 4, ET_DIST5_MEAN, 5,
                                      5000) == ET_DIST5_OK &&
      et_dist5_aot_sync_gradients_f32(handle, buf, out, 4, 6, 5000) ==
          ET_DIST5_OK &&
      et_dist5_aot_equal_bits_f32(buf, out, 4) == 1 &&
      et_dist5_aot_load_f32(out, 4, 3) == 4.0 &&
      et_dist5_aot_sha256_hex_f32(out, 4, hex, sizeof(hex)) == ET_DIST5_OK &&
      std::strlen(hex) == 64u;

  float x[128], gamma[16], y[128];
  float eps = 0.125f;
  for (int i = 0; i < 128; ++i)
    x[i] = 1.0f;
  for (int i = 0; i < 16; ++i)
    gamma[i] = 2.0f;
  const bool mod5_ok =
      et_dist5_aot_mod5_rms_forward(handle, x, gamma, &eps, y) == 0;

  et_dist5_aot_free_f32(buf);
  et_dist5_aot_free_f32(out);
  et_dist5_aot_close(handle);
  return (ok && mod5_ok) ? 0 : 5;
}
