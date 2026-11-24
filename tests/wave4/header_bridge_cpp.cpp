#include "wave4_composed_eshkol_bridge.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

int main() {
  et_wave4_composed_aot_context *bad = nullptr;
  if (et_wave4_composed_aot_create(nullptr, 0, 1, 5000, 5,
                                   ET_WAVE4_COMPOSED_F16, ET_AC4_POLICY_NONE, 0,
                                   0.125, 0.03125, 412775, 0.22, 0.5, 64.0, 2.0,
                                   0.5, 32.0, 128.0, 5, &bad) !=
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT ||
      bad != nullptr) {
    return 1;
  }

  if (et_wave4_composed_aot_bootstrap(nullptr) !=
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT ||
      et_wave4_composed_aot_step(nullptr, 0) !=
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT ||
      et_wave4_composed_aot_evaluate(nullptr, "/tmp/unused.eval5") !=
          ET_WAVE4_COMPOSED_INVALID_ARGUMENT) {
    return 2;
  }

  et_wave4_composed_aot_context *ctx = nullptr;
  const int32_t rc = et_wave4_composed_aot_create(
      nullptr, 0, 1, 5000, 4, ET_WAVE4_COMPOSED_F16,
      ET_AC4_POLICY_SQRT_SEGMENTS, 0, 0.125, 0.03125, 412775, 0.22, 0.5, 64.0,
      2.0, 0.5, 32.0, 128.0, 5, &ctx);
  if (rc != ET_WAVE4_COMPOSED_OK || ctx == nullptr)
    return 3;

  const int64_t p_count = et_wave4_composed_aot_parameter_count(ctx);
  const int64_t x_count = et_wave4_composed_aot_activation_count(ctx);
  if (p_count != 416 || x_count != 8) {
    et_wave4_composed_aot_destroy(ctx);
    return 4;
  }

  for (int64_t i = 0; i < p_count; ++i) {
    if (et_wave4_composed_aot_set_master(
            ctx, i, 0.05 * static_cast<double>((i % 9) - 4)) !=
        ET_WAVE4_COMPOSED_OK) {
      et_wave4_composed_aot_destroy(ctx);
      return 5;
    }
  }
  for (int64_t i = 0; i < x_count; ++i) {
    if (et_wave4_composed_aot_set_input(
            ctx, i, 0.1 * static_cast<double>((i % 5) - 2)) !=
            ET_WAVE4_COMPOSED_OK ||
        et_wave4_composed_aot_set_target(
            ctx, i, 0.05 * static_cast<double>((i % 3) - 1)) !=
            ET_WAVE4_COMPOSED_OK) {
      et_wave4_composed_aot_destroy(ctx);
      return 6;
    }
  }

  if (et_wave4_composed_aot_bootstrap(ctx) != ET_WAVE4_COMPOSED_OK ||
      et_wave4_composed_aot_step(ctx, 0) != ET_WAVE4_COMPOSED_OK) {
    et_wave4_composed_aot_destroy(ctx);
    return 7;
  }

  char hex[ET_WAVE4_COMPOSED_AOT_SHA256_HEX_BYTES];
  if (et_wave4_composed_aot_param_digest_hex(ctx, hex, 64) !=
          ET_WAVE4_COMPOSED_AOT_BUFFER_TOO_SMALL ||
      et_wave4_composed_aot_param_digest_hex(ctx, hex, sizeof(hex)) !=
          ET_WAVE4_COMPOSED_OK ||
      std::strlen(hex) != 64u ||
      !std::isfinite(et_wave4_composed_aot_last_loss(ctx)) ||
      et_wave4_composed_aot_live_activation_peak_bytes(ctx) != 576 ||
      et_wave4_composed_aot_scaler_scale(ctx) != 64.0) {
    et_wave4_composed_aot_destroy(ctx);
    return 8;
  }

  char dir[] = "/tmp/eshkol-w4-cpp.XXXXXX";
  if (mkdtemp(dir) == nullptr) {
    et_wave4_composed_aot_destroy(ctx);
    return 9;
  }
  char eval_path[128];
  std::snprintf(eval_path, sizeof(eval_path), "%s/cpp.eval5", dir);
  const int32_t eval_rc = et_wave4_composed_aot_evaluate(ctx, eval_path);
  const int32_t verify_rc =
      et_wave4_composed_aot_verify_manifest(ctx, eval_path);
  (void)unlink(eval_path);
  (void)rmdir(dir);

  const bool eval_ok =
      eval_rc == ET_WAVE4_COMPOSED_OK && verify_rc == ET_WAVE4_COMPOSED_OK &&
      std::isfinite(et_wave4_composed_aot_eval_mean_nll(ctx)) &&
      et_wave4_composed_aot_eval_perplexity(ctx) > 1.0 &&
      et_wave4_composed_aot_eval_contaminated_documents(ctx) == 0;
  et_wave4_composed_aot_destroy(ctx);
  return eval_ok ? 0 : 10;
}
