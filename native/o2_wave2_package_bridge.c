/* O2 is a source-composed successor of I2. Including the reviewed I2 bridge
 * here preserves one P1/provider/owned-carrier identity universe. */
#ifndef ET_O2_NATIVE_HELPERS_ONLY
#include "e1b_error_consumer_bridge.h"
#include "i2_wave2_package_bridge.c"
#endif

#include "o2_optimizer_internal.h"

#include <stdint.h>

static _Thread_local et_o2_error_v1 et_o2_bridge_error;

int64_t et_o2_private_last_error_category_v1(void) {
  return (int64_t)et_o2_bridge_error.category;
}

void *et_o2_private_builder_create_v1(
    int64_t count, int64_t clip_kind, int64_t clip_bits,
    int64_t schedule_kind, int64_t warmup, int64_t total,
    int64_t minimum_ratio_bits) {
  et_o2_optimizer_builder *builder = NULL;
  if (count < 0 || clip_kind < 0 || clip_bits < 0 || schedule_kind < 0 ||
      warmup < 0 || total < 0 || minimum_ratio_bits < 0 ||
      (uint64_t)count > SIZE_MAX || (uint64_t)clip_kind > UINT32_MAX ||
      (uint64_t)clip_bits > UINT32_MAX ||
      (uint64_t)schedule_kind > UINT32_MAX ||
      (uint64_t)minimum_ratio_bits > UINT32_MAX) {
    return NULL;
  }
  if (et_o2_optimizer_builder_create_v1(
          (size_t)count, (uint32_t)clip_kind, (uint32_t)clip_bits,
          (uint32_t)schedule_kind, (uint64_t)warmup, (uint64_t)total,
          (uint32_t)minimum_ratio_bits, &builder, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return builder;
}

int64_t et_o2_private_builder_set_v1(
    void *builder, int64_t index, void *parameter, void *p1_handle,
    int64_t learning_rate_bits, int64_t beta1_bits, int64_t beta2_bits,
    int64_t epsilon_bits, int64_t weight_decay_bits) {
  if (index < 0 || learning_rate_bits < 0 || beta1_bits < 0 ||
      beta2_bits < 0 || epsilon_bits < 0 || weight_decay_bits < 0 ||
      (uint64_t)index > SIZE_MAX ||
      (uint64_t)learning_rate_bits > UINT32_MAX ||
      (uint64_t)beta1_bits > UINT32_MAX ||
      (uint64_t)beta2_bits > UINT32_MAX ||
      (uint64_t)epsilon_bits > UINT32_MAX ||
      (uint64_t)weight_decay_bits > UINT32_MAX) {
    return -1;
  }
  return (int64_t)et_o2_optimizer_builder_set_v1(
      (et_o2_optimizer_builder *)builder, (size_t)index,
      (et_f32_parameter *)parameter, p1_handle,
      (uint32_t)learning_rate_bits, (uint32_t)beta1_bits,
      (uint32_t)beta2_bits, (uint32_t)epsilon_bits,
      (uint32_t)weight_decay_bits, &et_o2_bridge_error);
}

void *et_o2_private_builder_finish_v1(void *opaque) {
  et_o2_optimizer_builder *builder = (et_o2_optimizer_builder *)opaque;
  et_o2_optimizer *optimizer = NULL;
  if (et_o2_optimizer_builder_finish_v1(
          &builder, &optimizer, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return optimizer;
}

int64_t et_o2_private_builder_abort_v1(void *opaque) {
  et_o2_optimizer_builder *builder = (et_o2_optimizer_builder *)opaque;
  return (int64_t)et_o2_optimizer_builder_abort_v1(
      &builder, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_step_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_step_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_zero_grad_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_zero_grad_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_require_absent_gradients_v1(void *optimizer) {
  return (int64_t)et_o2_optimizer_require_absent_gradients_v1(
      (et_o2_optimizer *)optimizer, &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_completed_updates_v1(void *optimizer) {
  uint64_t count = 0u;
  if (et_o2_optimizer_completed_updates_v1(
          (const et_o2_optimizer *)optimizer, &count,
          &et_o2_bridge_error) != 0 || count > INT64_MAX) {
    return -1;
  }
  return (int64_t)count;
}

int64_t et_o2_private_optimizer_schedule_factor_bits_v1(void *optimizer) {
  uint32_t bits = 0u;
  if (et_o2_optimizer_schedule_factor_bits_v1(
          (const et_o2_optimizer *)optimizer, &bits,
          &et_o2_bridge_error) != 0) {
    return -1;
  }
  return (int64_t)bits;
}

void *et_o2_private_optimizer_state_snapshot_v1(void *optimizer) {
  et_o2_optimizer_state *state = NULL;
  if (et_o2_optimizer_state_snapshot_v1(
          (et_o2_optimizer *)optimizer, &state, &et_o2_bridge_error) != 0) {
    return NULL;
  }
  return state;
}

int64_t et_o2_private_optimizer_load_state_v1(void *optimizer, void *state) {
  return (int64_t)et_o2_optimizer_load_state_v1(
      (et_o2_optimizer *)optimizer, (et_o2_optimizer_state *)state,
      &et_o2_bridge_error);
}

int64_t et_o2_private_optimizer_state_release_v1(void *state) {
  const int32_t status = et_o2_optimizer_state_release_v1(
      (et_o2_optimizer_state *)state, &et_o2_bridge_error);
  et_o2_optimizer_error saved_error;
  uint32_t lifecycle = 0u;
  if (status == 0) {
    return 0;
  }
  saved_error = et_o2_bridge_error;
  if (et_o2_optimizer_state_lifecycle_v1(
          (const et_o2_optimizer_state *)state, &lifecycle,
          &et_o2_bridge_error) != 0) {
    et_o2_bridge_error = saved_error;
    return -3;
  }
  et_o2_bridge_error = saved_error;
  if (lifecycle == ET_O2_OPTIMIZER_STATE_LIVE) {
    return -1;
  }
  if (lifecycle == ET_O2_OPTIMIZER_STATE_DEAD) {
    return -2;
  }
  return -3;
}

#ifndef ET_O2_NATIVE_HELPERS_ONLY
#define ET_O2_DECLARE_UNARY(name)                                           \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t value)
#define ET_O2_DECLARE_BINARY(name)                                          \
  extern eshkol_tagged_value_t name(eshkol_tagged_value_t left,             \
                                     eshkol_tagged_value_t right)

ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_create_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_step_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_zero_grad_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_cabi_v1);
ET_O2_DECLARE_BINARY(et_e1b_private_o2_optimizer_load_state_cabi_v1);
ET_O2_DECLARE_UNARY(et_e1b_private_o2_optimizer_state_release_cabi_v1);

#define ET_O2_PUBLIC_UNARY(name, target)                                    \
  void name(void *input, void *output) {                                    \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) = target(*et_e1b_box_value_v1(input));      \
  }

#define ET_O2_PUBLIC_BINARY(name, target)                                   \
  void name(void *left, void *right, void *output) {                        \
    et_e1b_ensure_private_initialized_v1();                                 \
    *et_e1b_box_value_v1(output) =                                          \
        target(*et_e1b_box_value_v1(left), *et_e1b_box_value_v1(right));     \
  }

ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_create_v1,
                    et_e1b_private_o2_optimizer_create_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_step_v1,
                   et_e1b_private_o2_optimizer_step_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_zero_grad_v1,
                   et_e1b_private_o2_optimizer_zero_grad_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_v1,
                   et_e1b_private_o2_optimizer_state_cabi_v1)
ET_O2_PUBLIC_BINARY(et_e1b_public_o2_optimizer_load_state_v1,
                    et_e1b_private_o2_optimizer_load_state_cabi_v1)
ET_O2_PUBLIC_UNARY(et_e1b_public_o2_optimizer_state_release_v1,
                   et_e1b_private_o2_optimizer_state_release_cabi_v1)
#endif
