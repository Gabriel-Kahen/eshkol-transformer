#include "arena_memory.h"
#include <cstdlib>
static arena_t *target;
static size_t prior_used;
static bool prior_bounded;
extern "C" int64_t e3_p1_barrier_probe_begin(void) {
  const char *text = std::getenv("E3_P1_BARRIER_STAGE");
  const int stage = text ? std::atoi(text) : -1;
  const size_t budgets[] = {0, 48, 208, 240, 528, 816};
  if (stage < -1 || stage > 5) std::exit(2);
  if (stage < 0) return stage;
  target = region_current()->escape_base;
  prior_used = target->current_block->used;
  prior_bounded = target->bounded;
  if (target->current_block->size - prior_used < budgets[stage]) std::exit(2);
  target->bounded = true;
  target->current_block->used = target->current_block->size - budgets[stage];
  return stage;
}
extern "C" void e3_p1_barrier_probe_end(void) {
  if (!target) return;
  target->bounded = prior_bounded;
  target->current_block->used = prior_used;
  target = nullptr;
}
