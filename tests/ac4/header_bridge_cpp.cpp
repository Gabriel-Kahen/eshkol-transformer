#include "ac4_eshkol_bridge.h"

#include <cmath>
#include <cstdint>

int main() {
  et_ac4_aot_context *bad = nullptr;
  if (et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD, 99, 0, 0.125,
                        0.0, 42, 0, &bad) != ET_AC4_INVALID_ARGUMENT ||
      bad != nullptr) {
    return 1;
  }

  if (et_ac4_aot_run(nullptr, 0) != ET_AC4_INVALID_ARGUMENT)
    return 2;

  et_ac4_aot_context *ctx = nullptr;
  const int32_t rc =
      et_ac4_aot_create(4, 2, 3, 4, 2, 2, ET_AC4_FORWARD_BACKWARD,
                        ET_AC4_POLICY_SQRT_SEGMENTS, 0, 0.125, 0.2, 12345, 0,
                        &ctx);
  if (rc != ET_AC4_OK || ctx == nullptr)
    return 3;

  const int64_t x_count = et_ac4_aot_activation_floats(ctx);
  const int64_t p_count = et_ac4_aot_parameter_floats(ctx);
  if (x_count != 48 || p_count != 1600) {
    et_ac4_aot_destroy(ctx);
    return 4;
  }

  for (int64_t i = 0; i < x_count; ++i) {
    if (et_ac4_aot_set_input(ctx, i, 0.1 * static_cast<double>((i % 7) - 3)) !=
            ET_AC4_OK ||
        et_ac4_aot_set_target(ctx, i, 0.05 * static_cast<double>((i % 5) - 2)) !=
            ET_AC4_OK) {
      et_ac4_aot_destroy(ctx);
      return 5;
    }
  }
  for (int64_t i = 0; i < p_count; ++i) {
    if (et_ac4_aot_set_parameter(
            ctx, i, 0.05 * static_cast<double>((i % 11) - 5)) != ET_AC4_OK) {
      et_ac4_aot_destroy(ctx);
      return 6;
    }
  }

  const int64_t planned_peak = et_ac4_aot_planned_peak_bytes(ctx);
  if (et_ac4_aot_set_activation_limit(ctx, planned_peak) != ET_AC4_OK) {
    et_ac4_aot_destroy(ctx);
    return 7;
  }
  if (et_ac4_aot_run(ctx, 0) != ET_AC4_OK) {
    et_ac4_aot_destroy(ctx);
    return 8;
  }
  const double loss = et_ac4_aot_get_loss(ctx);
  if (!std::isfinite(loss) || loss < 0.0 ||
      et_ac4_aot_peak_activation_bytes(ctx) != planned_peak ||
      et_ac4_aot_recomputation_count(ctx) != 6 ||
      et_ac4_aot_loss_bits_lo(ctx) < 0 || et_ac4_aot_loss_bits_hi(ctx) < 0 ||
      et_ac4_aot_d_parameter_bits(ctx, 0) < 0) {
    et_ac4_aot_destroy(ctx);
    return 9;
  }
  if (et_ac4_aot_sgd_step(ctx, 0.035) != ET_AC4_OK) {
    et_ac4_aot_destroy(ctx);
    return 10;
  }

  et_ac4_aot_destroy(ctx);
  return 0;
}
