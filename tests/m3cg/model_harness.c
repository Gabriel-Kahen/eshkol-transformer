/* This test TU replaces m3_model.c; never compile both. */
#include "../../src/eshkol_transformer/m3_model.c"
#include "../../src/eshkol_transformer/m3_call_pins.h"
static struct {
  owner *model;
  et_g3t_model_pins_internal pins;
  unsigned char saved[4736];
} m3cg_context;
int64_t et_m3cg_test_acquire_v1(void *candidate) {
  owner *o = admit(candidate, OWNER, 0);
  if (!o || o->r.state != 1 || o->active || m3cg_context.model) return 0;
  et_f32_tensor_error e;
  if (et_g3t_model_pins_begin_internal(o->p,
      (const void *const *)o->handles, &m3cg_context.pins, &e)) return 0;
  m3cg_context.model = o;
  o->active = &m3cg_context;
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    memcpy(m3cg_context.saved + offset, m3cg_context.pins.views[i].data,
           m3cg_context.pins.views[i].byte_length);
    offset += m3cg_context.pins.views[i].byte_length;
  }
  return offset == sizeof(m3cg_context.saved);
}
int64_t et_m3cg_test_check_v1(void) {
  owner *o = admit(m3cg_context.model, OWNER, 0);
  et_f32_tensor_error e;
  if (!o || o->active != &m3cg_context ||
      et_g3t_model_pins_check_internal(&m3cg_context.pins, &e)) return 0;
  size_t offset = 0;
  for (size_t i = 0; i < 14; ++i) {
    if (m3cg_context.pins.parameters[i] != o->p[i] ||
        m3cg_context.pins.identities[i] != o->handles[i] ||
        memcmp(m3cg_context.saved + offset, m3cg_context.pins.views[i].data,
               m3cg_context.pins.views[i].byte_length)) return 0;
    offset += m3cg_context.pins.views[i].byte_length;
  }
  return offset == sizeof(m3cg_context.saved);
}
static size_t m3cg_cycles;
static int m3cg_preliminary_cycle = 1;
static et_f32_test_live_counts_v1 m3cg_live;
static et_f32_test_retired_counts_v1 m3cg_retired;
static void m3cg_retention(void) {
  /* The separate different-model/manual-frame case precedes workspace teardown. */
  if (m3cg_preliminary_cycle) { m3cg_preliminary_cycle = 0; return; }
  et_f32_test_live_counts_v1 live = {0};
  et_f32_test_retired_counts_v1 retired = {0};
  live.struct_size = sizeof(live);
  retired.struct_size = sizeof(retired);
  et_f32_test_live_counts_snapshot_v1(&live);
  et_f32_test_retired_counts_snapshot_v1(&retired);
  if (++m3cg_cycles == 1) { m3cg_live = live; m3cg_retired = retired; }
  if (m3cg_cycles != 1024 && m3cg_cycles != 8192) return;
#define SAME(field) do { if (live.field != m3cg_live.field || retired.field != m3cg_retired.field) abort(); } while (0)
  SAME(tensors); SAME(parameters); SAME(borrows); SAME(copy_plans);
  SAME(gradient_plans); SAME(reset_plans);
#undef SAME
  if (live.owned_clones != m3cg_live.owned_clones ||
      retired.retained_control_bytes != m3cg_retired.retained_control_bytes) abort();
}
int64_t et_m3cg_test_end_v1(void) {
  if (!et_m3cg_test_check_v1()) abort();
  et_g3t_model_pins_end_internal(&m3cg_context.pins);
  if (m3cg_context.model->active != &m3cg_context ||
      m3cg_context.pins.self || m3cg_context.pins.held_mask) abort();
  m3cg_context.model->active = NULL;
  m3cg_context.model = NULL;
  m3cg_retention();
  return 1;
}
