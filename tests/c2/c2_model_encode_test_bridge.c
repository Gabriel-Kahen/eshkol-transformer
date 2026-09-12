#include "f32_parameter_internal.h"

#include <stdint.h>

void et_i2_test_live_builder_counts_v1(size_t *copy_count, size_t *reset_count,
                                       size_t *decode_count);
void et_i2_test_retired_builder_counts_v1(size_t *copy_count,
                                          size_t *reset_count,
                                          size_t *decode_count,
                                          size_t *retained_control_bytes);

static int64_t et_c2_model_encode_count(size_t value) {
  return value > (size_t)INT64_MAX ? -1 : (int64_t)value;
}

int64_t et_c2_model_encode_test_counter_v1(int64_t index) {
  et_f32_test_live_counts_v1 live = {.struct_size = sizeof(live)};
  et_f32_test_retired_counts_v1 retired = {.struct_size = sizeof(retired)};
  et_f32_test_borrow_event_counts_v1 events = {
      .struct_size = sizeof(events),
  };
  size_t live_copy = 0u;
  size_t live_reset = 0u;
  size_t live_decode = 0u;
  size_t retired_copy = 0u;
  size_t retired_reset = 0u;
  size_t retired_decode = 0u;
  size_t bridge_retained = 0u;

  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  et_f32_test_borrow_event_counts_snapshot_v1(&events);
  et_i2_test_live_builder_counts_v1(&live_copy, &live_reset, &live_decode);
  et_i2_test_retired_builder_counts_v1(&retired_copy, &retired_reset,
                                       &retired_decode, &bridge_retained);

  switch (index) {
  case 0:
    return et_c2_model_encode_count(live.tensors);
  case 1:
    return et_c2_model_encode_count(live.parameters);
  case 2:
    return et_c2_model_encode_count(live.borrows);
  case 3:
    return et_c2_model_encode_count(live.copy_plans);
  case 4:
    return et_c2_model_encode_count(live.gradient_plans);
  case 5:
    return et_c2_model_encode_count(live.reset_plans);
  case 6:
    return et_c2_model_encode_count(live.owned_clones);
  case 7:
    return et_c2_model_encode_count(retired.tensors);
  case 8:
    return et_c2_model_encode_count(retired.parameters);
  case 9:
    return et_c2_model_encode_count(retired.borrows);
  case 10:
    return et_c2_model_encode_count(retired.copy_plans);
  case 11:
    return et_c2_model_encode_count(retired.gradient_plans);
  case 12:
    return et_c2_model_encode_count(retired.reset_plans);
  case 13:
    return et_c2_model_encode_count(retired.retained_control_bytes);
  case 14:
    return et_c2_model_encode_count(events.begin_calls);
  case 15:
    return et_c2_model_encode_count(events.view_calls);
  case 16:
    return et_c2_model_encode_count(events.end_calls);
  case 17:
    return et_c2_model_encode_count(live_copy);
  case 18:
    return et_c2_model_encode_count(live_reset);
  case 19:
    return et_c2_model_encode_count(live_decode);
  case 20:
    return et_c2_model_encode_count(retired_copy);
  case 21:
    return et_c2_model_encode_count(retired_reset);
  case 22:
    return et_c2_model_encode_count(retired_decode);
  case 23:
    return et_c2_model_encode_count(bridge_retained);
  default:
    return -1;
  }
}
