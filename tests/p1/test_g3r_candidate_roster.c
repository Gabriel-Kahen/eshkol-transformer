#define ET_P1_TRUSTED_BUILD 1
#define ET_P1_TEST_HOOKS 1
#define ET_G3R_CANDIDATE_RETIRE_PRIVATE 1
#include "../../native/p1_identity.c"

#include <assert.h>
#include <stdio.h>

void *__real_calloc(size_t, size_t);
static int allocation_limit = -1;
void *__wrap_calloc(size_t count, size_t size) {
  if (allocation_limit == 0) {
    allocation_limit = -1;
    return NULL;
  }
  if (allocation_limit > 0) --allocation_limit;
  return __real_calloc(count, size);
}

static void *result(void *context) {
  return et_p1_private_result_ptr_v1(context);
}

static int64_t value(void *context) {
  return et_p1_private_result_i64_v1(context);
}

static int64_t live(void *context) {
  assert(et_p1_private_live_entry_count_v1(context) == 0);
  return value(context);
}

static int64_t dead(void *context) {
  assert(et_p1_private_tombstone_count_v1(context) == 0);
  return value(context);
}

static int64_t roster_entries(void *context) {
  assert(et_p1_test_candidate_roster_entries_v1(context) == 0);
  return value(context);
}

static int64_t roster_bytes(void *context) {
  assert(et_p1_test_candidate_roster_bytes_v1(context) == 0);
  return value(context);
}

static void *begin_candidate(void *context) {
  assert(et_p1_private_candidate_construction_begin_v1(context) == 0);
  return result(context);
}

static void *add_token(void *context, void *scope, int handle) {
  assert((handle ? et_p1_private_construction_handle_create_v1(context, scope)
                 : et_p1_private_construction_module_create_v1(context, scope)) == 0);
  return result(context);
}

static void candidate_cycle(void *context, int prepared) {
  void *scope = begin_candidate(context);
  void *module = add_token(context, scope, 0);
  void *handle = add_token(context, scope, 1);
  int64_t baseline = live(context) - 2;
  assert(roster_entries(context) == 2);
  assert(roster_bytes(context) ==
         (int64_t)(ET_P1_CONSTRUCTION_CAPACITY * sizeof(et_p1_record *)));
  assert(et_p1_private_candidate_graph_preflight_v1(context, scope) ==
         ET_P1_STATUS_INVALID_STATE);
  if (prepared) {
    assert(et_p1_private_construction_prepare_v1(context, scope) == 0);
    assert(et_p1_private_candidate_graph_preflight_v1(context, scope) ==
           ET_P1_STATUS_INVALID_STATE);
    assert(et_p1_test_construction_commit_fail_next_v1() == 0);
    assert(et_p1_private_construction_commit_prepared_v1(context, scope) ==
           ET_P1_STATUS_INTERNAL);
    assert(roster_entries(context) == 2 && et_p1_public_token_live_v1(handle) == 1);
    assert(et_p1_private_construction_commit_prepared_v1(context, scope) == 0);
  } else {
    assert(et_p1_private_construction_seal_v1(context, scope) == 0);
  }
  assert(et_p1_private_candidate_graph_preflight_v1(context, scope) == 0);
  allocation_limit = 0;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) == 0);
  assert(allocation_limit == 0);
  allocation_limit = -1;
  assert(et_p1_public_token_live_v1(module) == 0);
  assert(et_p1_public_token_live_v1(handle) == 0);
  assert(live(context) == baseline);
  assert(roster_entries(context) == 0 && roster_bytes(context) == 0);
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) == 0);
  assert(et_p1_private_candidate_graph_preflight_v1(context, scope) ==
         ET_P1_STATUS_INVALID_STATE);
}

int main(void) {
  void *context = et_p1_private_context_create_v1();
  void *ordinary;
  void *ordinary_module;
  void *scope;
  void *other_scope;
  void *other_module;
  void *module;
  void *handle;
  et_p1_construction *ledger;
  et_p1_record *saved;
  et_p1_record **base;
  int64_t baseline;
  int64_t tombstones;
  assert(context != NULL);

  assert(et_p1_private_construction_begin_v1(context) == 0);
  ordinary = result(context);
  ordinary_module = add_token(context, ordinary, 0);
  assert(et_p1_private_construction_seal_v1(context, ordinary) == 0);
  assert(roster_bytes(context) == 0);
  assert(et_p1_private_candidate_graph_preflight_v1(context, ordinary) ==
         ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_candidate_graph_revoke_v1(context, ordinary) ==
         ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_public_token_live_v1(ordinary_module) == 1);
  scope = begin_candidate(context);
  module = add_token(context, scope, 0);
  assert(et_p1_private_construction_abort_v1(context, scope) == 0);
  assert(et_p1_public_token_live_v1(module) == 0);
  assert(roster_bytes(context) == 0);
  scope = begin_candidate(context);
  module = add_token(context, scope, 0);
  assert(et_p1_private_construction_prepare_v1(context, scope) == 0);
  assert(et_p1_private_construction_abort_prepared_v1(context, scope) == 0);
  assert(et_p1_public_token_live_v1(module) == 0);
  assert(roster_bytes(context) == 0);
  baseline = live(context);
  tombstones = dead(context);

  for (int cut = 0; cut < 2; ++cut) {
    allocation_limit = cut;
    assert(et_p1_private_candidate_construction_begin_v1(context) ==
           ET_P1_STATUS_INTERNAL);
    assert(result(context) == NULL && live(context) == baseline);
    assert(roster_bytes(context) == 0);
  }
  allocation_limit = -1;
  scope = begin_candidate(context);
  for (int cut = 0; cut < 2; ++cut) {
    allocation_limit = cut;
    assert(et_p1_private_construction_handle_create_v1(context, scope) ==
           ET_P1_STATUS_INTERNAL);
    assert(live(context) == baseline);
  }
  allocation_limit = -1;
  module = add_token(context, scope, 0);
  handle = add_token(context, scope, 1);
  assert(et_p1_private_construction_seal_v1(context, scope) == 0);
  ledger = (et_p1_construction *)scope;
  base = ledger->entries;
  saved = ledger->entries[1];
  assert(et_p1_private_candidate_graph_preflight_v1((void *)1, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_private_candidate_graph_preflight_v1(context, (char *)scope + 1) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_private_candidate_graph_preflight_v1(context, (void *)1) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  assert(et_p1_private_candidate_graph_preflight_v1(context, ordinary) ==
         ET_P1_STATUS_INVALID_STATE);

  ledger->entries = (et_p1_record **)module;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_STATE);
  ledger->entries = base;
  ledger->entries[1] = NULL;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ledger->entries[1] = (et_p1_record *)1;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ledger->entries[1] = ledger->entries[0];
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ledger->entries[1] = find_record(ordinary_module);
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ledger->entries[1] = saved;
  other_scope = begin_candidate(context);
  assert(et_p1_private_candidate_graph_preflight_v1(context, scope) ==
         ET_P1_STATUS_INVALID_STATE);
  other_module = add_token(context, other_scope, 0);
  assert(et_p1_private_construction_seal_v1(context, other_scope) == 0);
  ledger->entries[1] = find_record(other_module);
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ledger->entries[1] = saved;
  assert(et_p1_private_candidate_graph_revoke_v1(context, other_scope) == 0);
  ledger->count = 1;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_STATE);
  ledger->count = 2;
  saved->candidate_index = 0;
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  saved->candidate_index = 1;
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) ==
         ET_P1_STATUS_INVALID_ARGUMENT);
  ((uint64_t *)handle)[0] ^= UINT64_C(1);
  assert(et_p1_public_token_live_v1(module) == 1);
  assert(et_p1_public_token_live_v1(handle) == 1);
  assert(live(context) == baseline + 2 && dead(context) == tombstones + 1);
  assert(et_p1_private_candidate_graph_preflight_v1(context, scope) == 0);
  assert(et_p1_private_context_release_v1(context) == ET_P1_STATUS_INVALID_STATE);
  assert(et_p1_private_candidate_graph_revoke_v1(context, scope) == 0);
  assert(live(context) == baseline && dead(context) == tombstones + 3);
  assert(roster_entries(context) == 0 && roster_bytes(context) == 0);

  for (int run = 0; run < 8; ++run) {
    candidate_cycle(context, run & 1);
  }
  assert(live(context) == baseline);
  assert(dead(context) == tombstones + 19);
  assert(et_p1_public_token_live_v1(ordinary_module) == 1);
  puts("G3-R P1 candidate roster PASS: 19 retired tokens, exact native enrollment and retained-roster baseline");
  return 0;
}
