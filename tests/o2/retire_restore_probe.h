/* Test-only diagnostic shared by the genuine O2 and TR3 fixtures. */
#if !defined(ET_G3R_CANDIDATE_RETIRE_PRIVATE) || !defined(ET_CPU4_CANDIDATE_PRIVATE)
#error "retirement probe requires the real candidate-ON owner"
#endif

static void retire_probe_quiescent(et_o2_optimizer *optimizer) {
  CHECK(et_o2_builders == NULL && et_o2_reconstruct_builders == NULL);
  CHECK(optimizer->busy == 0u);
#ifdef ET_TR3_O2_PRIVATE_OPERATIONS
  CHECK(optimizer->active_operation == NULL);
#endif
  CHECK(et_o2_borrows == NULL);
  for (const et_o2_optimizer_state *state = et_o2_states; state;
       state = state->registry_next) {
    CHECK(state->lifecycle == ET_O2_OPTIMIZER_STATE_DEAD);
    CHECK(state->active_borrows == 0u);
  }
}

static void retire_probe_result(et_o2_optimizer *optimizer,
                                const et_o2_entry before[14], int changed,
                                const char *route) {
  const char *expected = getenv("ET_O2_RETIRE_RESTORE_EXPECT");
  CHECK(expected != NULL);
  if (expected == NULL) return;
  CHECK(strcmp(expected, "stale") == 0 || strcmp(expected, "synced") == 0);
  int stale = strcmp(expected, "stale") == 0;
  et_o2_retire_provenance *record = et_o2_retire_record_find(optimizer);
  CHECK(record != NULL);
  if (record == NULL) return;
  CHECK(record->owner == optimizer && record->entries == optimizer->entries);
  CHECK(record->live == 1u && record->count == 14u);
  retire_probe_quiescent(optimizer);
  for (size_t index = 0u; index < 14u; ++index) {
    const et_o2_entry *entry = &optimizer->entries[index];
    CHECK(entry->parameter == before[index].parameter);
    CHECK(entry->p1_handle == before[index].p1_handle);
    CHECK(entry->storage_owner == before[index].storage_owner);
    CHECK(entry->exp_avg == before[index].exp_avg);
    CHECK(entry->exp_avg_sq == before[index].exp_avg_sq);
    CHECK(entry->initialized == before[index].initialized);
    CHECK((memcmp(&entry->options, &before[index].options,
                  sizeof(entry->options)) != 0) == changed);
    CHECK(memcmp(&record->roster[index], stale ? &before[index] : entry,
                 sizeof(*entry)) == 0);
    printf("O2-PROVENANCE %s index=%zu old-lr=%08x current-lr=%08x recorded-lr=%08x\n",
           route, index, before[index].options.learning_rate_bits,
           entry->options.learning_rate_bits,
           record->roster[index].options.learning_rate_bits);
  }
  et_o2_error_v1 error;
  memset(&error, 0, sizeof(error));
  int32_t direct = et_o2_private_candidate_retire_preflight_v1(optimizer, NULL);
  int32_t detailed = et_o2_private_candidate_retire_preflight_v1(optimizer, &error);
  CHECK(direct == detailed);
  CHECK(direct == (stale && changed ? ET_O2_STATUS_INVALID_STATE : 0));
  if (stale && changed) {
    CHECK(error.code == ET_O2_CODE_INVALID_HANDLE);
    CHECK(strcmp(error.message,
                 "optimizer entry differs from its published roster") == 0);
  }
  printf("O2-RETIRE-DIAGNOSTIC route=%s changed=%d expected=%s status=%d "
         "category=%u code=%u operation=%s message=%s quiescent=1\n",
         route, changed, expected, direct, error.category, error.code,
         error.operation, error.message);
}

#ifdef ET_O2_RETIRE_RESTORE_REPAIR_CHECKS
/* Deliberate out-of-band corruption controls; never a legitimate restore. */
static void retire_probe_corrupt(et_o2_entry *entry, const et_o2_entry *peer,
                                 unsigned field) {
  switch (field) {
  case 0: entry->parameter = peer->parameter; break;
  case 1: entry->p1_handle = peer->p1_handle; break;
  case 2: entry->storage_owner = peer->storage_owner; break;
  case 3: entry->exp_avg = peer->exp_avg; break;
  case 4: entry->exp_avg_sq = peer->exp_avg_sq; break;
  case 5: entry->initialized = 0u; break;
  case 6: entry->options.learning_rate_bits ^= 1u; break;
  case 7: entry->options.beta1_bits ^= 1u; break;
  case 8: entry->options.beta2_bits ^= 1u; break;
  case 9: entry->options.epsilon_bits ^= 1u; break;
  case 10: entry->options.weight_decay_bits ^= 1u; break;
  default: CHECK(0); break;
  }
}
#endif
