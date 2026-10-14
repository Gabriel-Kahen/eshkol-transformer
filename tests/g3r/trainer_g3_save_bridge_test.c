/* Dependency-light causal cuts for the private roster-to-G3 hold bridge. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

extern void *et_g3r_private_trainer_g3_save_begin_v1(void *, void *);
extern int64_t et_g3r_private_trainer_g3_save_check_v1(void *);
extern int64_t et_g3r_private_trainer_g3_save_end_v1(void *);
extern int64_t et_g3r_private_trainer_g3_save_last_source_v1(void);

static int model, generator, hold, parameter[14], identity[14];
static int model_live = 1, generator_live = 1, generator_busy;
static int roster_calls, begin_calls, end_calls, native_live;

int64_t et_m3t_private_owner_save_roster_v1(
    void *candidate, void *parameters[14], const void *identities[14]) {
  ++roster_calls;
  if (candidate != &model || !model_live) return 2;
  for (int i = 0; i < 14; ++i) {
    parameters[i] = &parameter[i];
    identities[i] = &identity[i];
  }
  return 0;
}

void *et_g3r_private_g3t_save_begin_v1(
    void *candidate, void *const parameters[14],
    const void *const identities[14]) {
  ++begin_calls;
  if (candidate != &generator || !generator_live || generator_busy)
    return NULL;
  for (int i = 0; i < 14; ++i)
    if (parameters[i] != &parameter[i] || identities[i] != &identity[i])
      return NULL;
  native_live = 1;
  return &hold;
}

int64_t et_g3r_private_g3t_save_check_v1(void *candidate) {
  return candidate == &hold && native_live ? 0 : 2;
}

int64_t et_g3r_private_g3t_save_end_v1(void *candidate) {
  ++end_calls;
  if (candidate != &hold || !native_live) return 2;
  native_live = 0;
  return 0;
}

int main(void) {
  int foreign;
  void *published;
  published = et_g3r_private_trainer_g3_save_begin_v1(&generator, &foreign);
  assert(!published && !native_live && begin_calls == 0);
  assert(et_g3r_private_trainer_g3_save_last_source_v1() == 1);
  model_live = 0;
  published = et_g3r_private_trainer_g3_save_begin_v1(&generator, &model);
  assert(!published && !native_live && begin_calls == 0);
  model_live = 1;
  published = et_g3r_private_trainer_g3_save_begin_v1(&foreign, &model);
  assert(!published && !native_live && begin_calls == 1);
  assert(et_g3r_private_trainer_g3_save_last_source_v1() == 2);
  generator_live = 0;
  published = et_g3r_private_trainer_g3_save_begin_v1(&generator, &model);
  assert(!published && !native_live && begin_calls == 2);
  generator_live = 1;
  generator_busy = 1;
  published = et_g3r_private_trainer_g3_save_begin_v1(&generator, &model);
  assert(!published && !native_live && begin_calls == 3);
  generator_busy = 0;
  published = et_g3r_private_trainer_g3_save_begin_v1(&generator, &model);
  assert(published && native_live && begin_calls == 4);
  assert(et_g3r_private_trainer_g3_save_check_v1(published) == 0);
  assert(et_g3r_private_trainer_g3_save_end_v1(published) == 0);
  assert(!native_live && end_calls == 1);
  assert(et_g3r_private_trainer_g3_save_check_v1(published) == -1);
  assert(et_g3r_private_trainer_g3_save_end_v1(published) == -1);
  assert(end_calls == 1);
  assert(roster_calls == 6);
  puts("G3R-TRAINER-G3-BRIDGE-PASS");
  return 0;
}
