#include "continuous_batch_eshkol_bridge.h"
#include "eshkol_transformer/continuous_batch_abi.h"

#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout_v<et_cb_config>);
static_assert(std::is_standard_layout_v<et_cb_request_spec>);
static_assert(std::is_standard_layout_v<et_cb_plan_entry>);
static_assert(std::is_standard_layout_v<et_cb_step_batch>);
static_assert(std::is_standard_layout_v<et_cb_commit_item>);
static_assert(std::is_standard_layout_v<et_cb_request_info>);
static_assert(std::is_standard_layout_v<et_cb_stats>);
static_assert(std::is_same_v<et_cb_status, int32_t>);

int main() {
  const et_cb_config cfg = {
      4u, 16u, 8u, 8u, 4u,
      static_cast<uint32_t>(ET_CB_POLICY_DECODE_FIRST_CHUNKED_PREFILL)};
  if (et_cb_validate_config(&cfg) != ET_CB_OK)
    return 1;

  static et_cb_scheduler sched;
  if (et_cb_init(&sched, &cfg) != ET_CB_OK)
    return 2;

  const et_cb_request_spec req = {10u, 6u, 2u, 1, 0u};
  if (et_cb_enqueue(&sched, &req) != ET_CB_OK)
    return 3;

  et_cb_step_batch plan = {};
  if (et_cb_step_plan(&sched, &plan) != ET_CB_OK)
    return 4;
  if (plan.entry_count != 1u || plan.entries[0].chunk_tokens != 6u ||
      plan.entries[0].produces_output_token != 1u ||
      plan.used_kv_blocks != 2u || plan.free_kv_blocks != 6u)
    return 5;

  const et_cb_commit_item item = {10u, 1u, 0u};
  if (et_cb_step_commit(&sched, plan.step_index, &item, 1u) != ET_CB_OK)
    return 6;

  if (et_cb_cancel(&sched, 10u) != ET_CB_OK)
    return 7;
  if (et_cb_cancel(&sched, 10u) != ET_CB_ERR_ALREADY_TERMINAL)
    return 8;

  et_cb_stats stats = {};
  if (et_cb_get_stats(&sched, &stats) != ET_CB_OK)
    return 9;
  if (stats.free_kv_blocks != 8u || stats.used_kv_blocks != 0u)
    return 10;
  if (et_cb_verify_invariants(&sched) != ET_CB_OK)
    return 11;

  et_cb_aot_ctx *ctx = nullptr;
  if (et_cb_aot_create(4, 16, 8, 8, 4, 0, &ctx) != ET_CB_OK || ctx == nullptr)
    return 12;
  if (et_cb_aot_enqueue(ctx, 1, 5, 2, 0) != ET_CB_OK)
    return 13;
  if (et_cb_aot_step_plan(ctx) != ET_CB_OK)
    return 14;
  if (et_cb_aot_commit_begin(ctx) != ET_CB_OK)
    return 15;
  if (et_cb_aot_step_commit(ctx, 0, 1) != ET_CB_OK)
    return 16;
  if (et_cb_aot_verify_invariants(ctx) != ET_CB_OK)
    return 17;
  et_cb_aot_destroy(ctx);
  return 0;
}
