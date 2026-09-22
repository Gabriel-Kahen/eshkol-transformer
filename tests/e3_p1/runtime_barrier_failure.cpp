// Defensive pinned-runtime diagnostic; no reclaimed memory is dereferenced.
#include "arena_memory.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" int eshkol_arena_poison_enabled(void);
using Value = eshkol_tagged_value_t;
static Value integer(int64_t n) { Value v{}; v.type = ESHKOL_VALUE_INT64; v.data.int_val = n; return v; }
static Value pointer(void *p) { Value v{}; v.type = ESHKOL_VALUE_HEAP_PTR; v.data.ptr_val = reinterpret_cast<uint64_t>(p); return v; }
static Value *slots(void *p) { return reinterpret_cast<Value *>(static_cast<char *>(p) + 8); }
static void *address(Value v) { return reinterpret_cast<void *>(v.data.ptr_val); }
static void *vector(arena_t *arena, size_t n) {
  void *p = arena_allocate_vector_with_header(arena, n);
  if (!p) std::exit(2);
  *static_cast<uint64_t *>(p) = n;
  for (size_t i = 0; i < n; ++i) slots(p)[i] = integer(0);
  return p;
}

int main(int argc, char **argv) {
  // -1: ordinary successful promotion; 0..5: exact root copy-prefix budgets.
  const int stage = argc == 2 ? std::atoi(argv[1]) : -1;
  const size_t budgets[] = {0, 48, 208, 240, 528, 816};
  if (stage < -1 || stage > 5) return 2;
  arena_t *root = get_global_arena_shared();
  if (!root) return 2;
  void *holder = vector(root, 1);
  slots(holder)[0] = integer(73);
  eshkol_region_t *region = region_create("e3-p1-barrier-probe", 4096);
  if (!region) return 2;
  region_push(region);
  // The actual P1 publication graph shape: cell -> nine-slot record -> inert
  // token and three fixed17 arrays. Model/frame fields here are inert values;
  // this is a runtime publication probe, not proof of E3 frame admission.
  void *token = vector(region->arena, 1);
  void *nodes = vector(region->arena, 17);
  void *saved = vector(region->arena, 17);
  void *heads = vector(region->arena, 17);
  void *record = vector(region->arena, 9);
  slots(record)[0] = pointer(token);
  slots(record)[5] = pointer(nodes);
  slots(record)[6] = pointer(saved);
  slots(record)[7] = pointer(heads);
  auto *cell = arena_allocate_cons_with_header(region->arena);
  if (!cell) return 2;
  cell->car = pointer(record);
  cell->cdr = integer(0);
  Value value = pointer(cell), output = integer(73);
  const bool bounded = root->bounded;
  const size_t used = root->current_block->used;
  if (stage >= 0) {
    if (root->current_block->size - used < budgets[stage]) return 2;
    root->bounded = true;
    root->current_block->used = root->current_block->size - budgets[stage];
  }
  // Same destination as vector-set!: vector payload, temporary barrier result,
  // then store. A NULL allocation is injected via existing bounded-arena policy.
  eshkol_region_write_barrier_into(&output, holder, &value);
  slots(holder)[0] = output;
  auto *canonical_cell = static_cast<arena_tagged_cons_cell_t *>(address(output));
  void *canonical_record = address(canonical_cell->car);
  size_t regional = canonical_cell == cell;
  regional += canonical_record == record;
  regional += address(slots(canonical_record)[0]) == token;
  regional += address(slots(canonical_record)[5]) == nodes;
  regional += address(slots(canonical_record)[6]) == saved;
  regional += address(slots(canonical_record)[7]) == heads;
  const bool unchanged = slots(holder)[0].type == ESHKOL_VALUE_INT64;
  root->bounded = bounded;
  if (stage >= 0) {
    // Retire the probe holder before destroying its region. No dangling read.
    slots(holder)[0] = integer(73);
    root->current_block->used = used;
  }
  region_pop();
  bool survived = false;
  if (stage < 0 && regional == 0) {
    // Only the proven distinct promoted graph is accessed after poisoned exit.
    auto *live_cell = static_cast<arena_tagged_cons_cell_t *>(address(slots(holder)[0]));
    void *live_record = address(live_cell->car);
    survived = *static_cast<uint64_t *>(live_record) == 9 &&
        *static_cast<uint64_t *>(address(slots(live_record)[0])) == 1 &&
        *static_cast<uint64_t *>(address(slots(live_record)[5])) == 17 &&
        *static_cast<uint64_t *>(address(slots(live_record)[6])) == 17 &&
        *static_cast<uint64_t *>(address(slots(live_record)[7])) == 17;
  }
  std::printf("stage=%d poison=%d barrier_returned=1 output_unchanged=%d regional_edges=%zu survived=%d\n",
              stage, eshkol_arena_poison_enabled(), unchanged, regional, survived);
  return stage < 0 ? (survived ? 0 : 2) : (regional ? 1 : 0);
}
