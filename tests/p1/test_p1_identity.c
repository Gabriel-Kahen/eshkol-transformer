#define ET_P1_PRIVATE_API 1
#include "p1_identity_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define TOKEN_BYTES 264u
#define TOKEN_NONCE_OFFSET 0u
#define FORMER_BINDING_OFFSET 16u
#define FORMER_KIND_OFFSET 80u

static int checks;

static void check(int condition, const char *label) {
  if (!condition) {
    (void)fprintf(stderr, "P1 identity FAIL: %s\n", label);
    exit(1);
  }
  checks++;
}

static void *created(void *context, int64_t status, const char *label) {
  check(status == ET_P1_STATUS_OK, label);
  void *result = et_p1_private_result_ptr_v1(context);
  check(result != NULL, "create returned a token");
  return result;
}

static void create_callbacks(void *context, void *callbacks[7]) {
  size_t index;
  for (index = 0u; index < 7u; index++) {
    callbacks[index] = created(
        context, et_p1_private_callback_identity_create_v1(context),
        "callback identity creation succeeds");
  }
}

static int64_t seal_provider(void *context, void *provider,
                             void *callbacks[7]) {
  return et_p1_private_provider_seal_v1(
      context, provider, callbacks[0], callbacks[1], callbacks[2],
      callbacks[3], callbacks[4], callbacks[5], callbacks[6]);
}

static int64_t match_provider(void *context, void *provider,
                              void *callbacks[7]) {
  return et_p1_private_provider_snapshot_matches_v1(
      context, provider, callbacks[0], callbacks[1], callbacks[2],
      callbacks[3], callbacks[4], callbacks[5], callbacks[6]);
}

static int64_t live_count(void *context) {
  check(et_p1_private_live_entry_count_v1(context) == ET_P1_STATUS_OK,
        "live-count query succeeds");
  return et_p1_private_result_i64_v1(context);
}

static int64_t tombstone_count(void *context) {
  check(et_p1_private_tombstone_count_v1(context) == ET_P1_STATUS_OK,
        "tombstone-count query succeeds");
  return et_p1_private_result_i64_v1(context);
}

int main(void) {
  void *context;
  void *provider;
  void *provider_two;
  void *callbacks[7];
  void *callbacks_two[7];
  void *alias_callbacks[7];
  void *state;
  void *other_state;
  void *state_tensor;
  void *release_callback;
  void *swapped_release_callback;
  void *legacy_upgrade_release_callback;
  void *alias_provider;
  void *temporary_callback;
  void *duplicate_provider;
  void *duplicate_callback;
  unsigned char *copy;
  unsigned char saved_nonce_byte;
  int foreign = 0;
  int64_t before;
  pid_t child;
  int child_status;
  char provider_id[] = "fixture-v1";

  check(et_p1_public_identity_abi_major_v1() == 1,
        "ABI major is fixed");
  check(et_p1_public_identity_abi_minor_v1() == 1,
        "ABI minor is fixed");
  check(et_p1_public_token_kind_v1(NULL) == ET_P1_TOKEN_FOREIGN,
        "null token is foreign");
  check(et_p1_public_token_live_v1(&foreign) == 0,
        "arbitrary pointer is not live");

  context = et_p1_private_context_create_v1();
  check(context != NULL, "private capability context is created");
  check(et_p1_private_context_create_v1() == NULL,
        "second private capability is unavailable");

  check(et_p1_private_provider_create_v1(
            context, provider_id, (int64_t)(sizeof(provider_id) - 1u)) ==
            ET_P1_STATUS_OK,
        "provider identity creation succeeds");
  provider = et_p1_private_result_ptr_v1(context);
  provider_id[0] = 'X';
  check(et_p1_public_token_kind_v1(provider) == ET_P1_TOKEN_PROVIDER,
        "provider token has its exact kind");
  check(et_p1_public_token_live_v1(provider) == 1,
        "provider token is live");

  create_callbacks(context, callbacks);
  check(seal_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "provider snapshot seals");
  check(seal_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "same provider snapshot seal is idempotent");
  check(match_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "sealed provider snapshot matches");
  check(et_p1_private_provider_abort_v1(context, provider) ==
            ET_P1_STATUS_INVALID_STATE,
        "sealed provider cannot be aborted");

  temporary_callback = created(
      context, et_p1_private_callback_identity_create_v1(context),
      "temporary callback identity creation succeeds");
  {
    void *substituted[7];
    memcpy(substituted, callbacks, sizeof(substituted));
    substituted[6] = temporary_callback;
    check(seal_provider(context, provider, substituted) ==
              ET_P1_STATUS_INVALID_STATE,
          "callback-swapped reseal is rejected");
    check(et_p1_private_error_code_v1(context) ==
              ET_P1_CODE_ALREADY_SEALED,
          "callback-swapped reseal has exact status data");
    check(match_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
          "rejected reseal leaves the immutable snapshot intact");
  }
  check(et_p1_private_callback_identity_revoke_v1(context, callbacks[0]) ==
            ET_P1_STATUS_INVALID_STATE,
        "sealed callback identity cannot be revoked");
  check(et_p1_private_callback_identity_revoke_v1(context,
                                                   temporary_callback) ==
            ET_P1_STATUS_OK,
        "unsealed temporary callback identity is revocable");
  before = live_count(context);
  check(et_p1_private_callback_identity_revoke_v1(context,
                                                   temporary_callback) ==
            ET_P1_STATUS_INVALID_STATE,
        "repeat callback cleanup rejects a stale identity");
  check(live_count(context) == before,
        "repeat callback cleanup is failure-atomic");
  check(et_p1_private_callback_identity_revoke_v1(context, provider) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "callback cleanup rejects the wrong token kind");
  check(et_p1_public_token_live_v1(temporary_callback) == 0,
        "revoked callback identity is stale");

  before = live_count(context);
  duplicate_provider = created(
      context, et_p1_private_provider_create_v1(context, "duplicate", 9),
      "duplicate-role provider identity creation succeeds");
  duplicate_callback = created(
      context, et_p1_private_callback_identity_create_v1(context),
      "duplicate-role callback identity creation succeeds");
  check(et_p1_private_provider_seal_v1(
            context, duplicate_provider, duplicate_callback,
            duplicate_callback, duplicate_callback, duplicate_callback,
            duplicate_callback, duplicate_callback, duplicate_callback) ==
            ET_P1_STATUS_INVALID_STATE,
        "one callback identity cannot fill multiple provider roles");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_SNAPSHOT_MISMATCH,
        "duplicate callback roles have exact status data");
  check(live_count(context) == before + 2,
        "rejected duplicate roles do not change token liveness");
  check(et_p1_private_callback_identity_revoke_v1(context,
                                                   duplicate_callback) ==
            ET_P1_STATUS_OK,
        "duplicate-role callback remains unpublished and revocable");
  check(et_p1_private_provider_abort_v1(context, duplicate_provider) ==
            ET_P1_STATUS_OK,
        "duplicate-role provider remains unpublished and abortable");
  check(live_count(context) == before,
        "duplicate-role cleanup restores the live baseline");

  copy = (unsigned char *)malloc(TOKEN_BYTES);
  check(copy != NULL, "token-copy scratch allocation succeeds");
  memcpy(copy, provider, TOKEN_BYTES);
  check(et_p1_public_token_kind_v1(copy) == ET_P1_TOKEN_FOREIGN,
        "copied token bytes cannot forge identity");
  check(et_p1_public_token_live_v1(copy) == 0,
        "copied token bytes are not live");
  free(copy);

  before = live_count(context);
  check(et_p1_private_provider_create_v1(context, NULL, 0) ==
            ET_P1_STATUS_OK,
        "zero-length provider identity is represented exactly");
  {
    void *empty_provider = et_p1_private_result_ptr_v1(context);
    check(et_p1_private_provider_abort_v1(context, empty_provider) ==
              ET_P1_STATUS_OK,
          "unpublished provider identity aborts");
    check(live_count(context) == before,
          "unpublished provider abort restores the live baseline");
    check(et_p1_private_provider_abort_v1(context, empty_provider) ==
              ET_P1_STATUS_INVALID_STATE,
          "repeat unpublished-provider abort rejects stale identity");
    check(live_count(context) == before,
          "repeat unpublished-provider abort is failure-atomic");
  }

  saved_nonce_byte = ((unsigned char *)provider)[TOKEN_NONCE_OFFSET];
  ((unsigned char *)provider)[TOKEN_NONCE_OFFSET] ^= UINT8_C(1);
  check(et_p1_public_token_live_v1(provider) == 0,
        "mutated token integrity is rejected");
  check(match_provider(context, provider, callbacks) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "mutated provider cannot authorize snapshot lookup");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_TOKEN_INTEGRITY,
        "mutated provider reports token-integrity status");
  ((unsigned char *)provider)[TOKEN_NONCE_OFFSET] = saved_nonce_byte;
  check(match_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "restored registered identity remains valid");
  ((unsigned char *)provider)[FORMER_KIND_OFFSET] ^= UINT8_C(0xff);
  check(et_p1_public_token_kind_v1(provider) == ET_P1_TOKEN_PROVIDER,
        "caller storage cannot alter authoritative token kind");
  check(match_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "caller reserved-byte mutation cannot alter provider authority");

  state = created(context, et_p1_private_state_dict_create_v1(context),
                  "state identity creation succeeds");
  before = live_count(context);
  check(et_p1_private_state_bind_v1(context, state, provider) ==
            ET_P1_STATUS_OK,
        "state binds to admitted provider");
  check(et_p1_private_state_bind_v1(context, state, provider) ==
            ET_P1_STATUS_OK,
        "repeat binding is deterministic and idempotent");
  check(live_count(context) == before,
        "binding and repeated binding allocate no registry entry");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_OK,
        "bound provider lookup succeeds");
  check(et_p1_private_result_ptr_v1(context) == provider,
        "bound provider lookup preserves exact carrier identity");
  memset((unsigned char *)state + FORMER_BINDING_OFFSET, 0, sizeof(void *));
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_OK &&
            et_p1_private_result_ptr_v1(context) == provider,
        "caller storage cannot clear authoritative state binding");

  check(et_p1_private_provider_create_v1(context, "alternate-v1", 12) ==
            ET_P1_STATUS_OK,
        "second provider identity creation succeeds");
  provider_two = et_p1_private_result_ptr_v1(context);
  check(seal_provider(context, provider_two, callbacks) ==
            ET_P1_STATUS_INVALID_STATE,
        "callback identities cannot be transplanted between providers");
  create_callbacks(context, callbacks_two);
  release_callback = created(
      context, et_p1_private_callback_identity_create_v1(context),
      "release callback identity creation succeeds");
  check(et_p1_private_provider_seal_release_v1(
            context, provider_two, callbacks_two[0], callbacks_two[1],
            callbacks_two[2], callbacks_two[3], callbacks_two[4],
            callbacks_two[5], callbacks_two[6], release_callback) ==
            ET_P1_STATUS_OK,
        "release-capable provider snapshot seals");
  check(et_p1_private_provider_seal_release_v1(
            context, provider_two, callbacks_two[0], callbacks_two[1],
            callbacks_two[2], callbacks_two[3], callbacks_two[4],
            callbacks_two[5], callbacks_two[6], release_callback) ==
            ET_P1_STATUS_OK,
        "same release-capable provider snapshot seal is idempotent");
  check(seal_provider(context, provider_two, callbacks_two) ==
            ET_P1_STATUS_INVALID_STATE,
        "legacy seal cannot downgrade a release-capable provider");
  check(match_provider(context, provider_two, callbacks_two) ==
            ET_P1_STATUS_INVALID_STATE,
        "legacy match cannot accept a release-capable provider");
  check(et_p1_private_provider_snapshot_matches_release_v1(
            context, provider_two, callbacks_two[0], callbacks_two[1],
            callbacks_two[2], callbacks_two[3], callbacks_two[4],
            callbacks_two[5], callbacks_two[6], release_callback) ==
            ET_P1_STATUS_OK,
        "release-capable provider snapshot matches exactly");
  check(et_p1_private_callback_identity_revoke_v1(context,
                                                   release_callback) ==
            ET_P1_STATUS_INVALID_STATE,
        "sealed release callback identity cannot be revoked");

  swapped_release_callback = created(
      context, et_p1_private_callback_identity_create_v1(context),
      "swapped release callback identity creation succeeds");
  before = live_count(context);
  check(et_p1_private_provider_seal_release_v1(
            context, provider_two, callbacks_two[0], callbacks_two[1],
            callbacks_two[2], callbacks_two[3], callbacks_two[4],
            callbacks_two[5], callbacks_two[6], swapped_release_callback) ==
            ET_P1_STATUS_INVALID_STATE,
        "release callback substitution cannot reseal a provider");
  check(live_count(context) == before,
        "rejected release callback substitution is failure-atomic");
  check(et_p1_private_provider_snapshot_matches_release_v1(
            context, provider_two, callbacks_two[0], callbacks_two[1],
            callbacks_two[2], callbacks_two[3], callbacks_two[4],
            callbacks_two[5], callbacks_two[6], release_callback) ==
            ET_P1_STATUS_OK,
        "rejected release substitution preserves the exact snapshot");
  check(et_p1_private_callback_identity_revoke_v1(
            context, swapped_release_callback) == ET_P1_STATUS_OK,
        "rejected release substitute remains revocable");

  legacy_upgrade_release_callback = created(
      context, et_p1_private_callback_identity_create_v1(context),
      "legacy-upgrade release callback identity creation succeeds");
  before = live_count(context);
  check(et_p1_private_provider_seal_release_v1(
            context, provider, callbacks[0], callbacks[1], callbacks[2],
            callbacks[3], callbacks[4], callbacks[5], callbacks[6],
            legacy_upgrade_release_callback) == ET_P1_STATUS_INVALID_STATE,
        "release-aware seal cannot upgrade a legacy provider");
  check(et_p1_private_provider_snapshot_matches_release_v1(
            context, provider, callbacks[0], callbacks[1], callbacks[2],
            callbacks[3], callbacks[4], callbacks[5], callbacks[6],
            legacy_upgrade_release_callback) == ET_P1_STATUS_INVALID_STATE,
        "release-aware match cannot accept a legacy provider");
  check(live_count(context) == before,
        "rejected legacy upgrade is failure-atomic");
  check(match_provider(context, provider, callbacks) == ET_P1_STATUS_OK,
        "rejected release upgrade preserves the legacy snapshot");
  check(et_p1_private_callback_identity_revoke_v1(
            context, legacy_upgrade_release_callback) == ET_P1_STATUS_OK,
        "rejected legacy-upgrade release callback remains revocable");

  before = live_count(context);
  alias_provider = created(
      context, et_p1_private_provider_create_v1(context, "alias-v2", 8),
      "release-role-alias provider identity creation succeeds");
  create_callbacks(context, alias_callbacks);
  {
    size_t index;
    for (index = 0u; index < 7u; index++) {
      check(et_p1_private_provider_seal_release_v1(
                context, alias_provider, alias_callbacks[0],
                alias_callbacks[1], alias_callbacks[2], alias_callbacks[3],
                alias_callbacks[4], alias_callbacks[5], alias_callbacks[6],
                alias_callbacks[index]) == ET_P1_STATUS_INVALID_STATE,
            "release callback identity cannot alias a legacy callback role");
    }
  }
  check(live_count(context) == before + 8,
        "rejected release-role alias changes no token liveness");
  {
    size_t index;
    for (index = 0u; index < 7u; index++) {
      check(et_p1_private_callback_identity_revoke_v1(
                context, alias_callbacks[index]) == ET_P1_STATUS_OK,
            "release-role-alias callback remains revocable");
    }
  }
  check(et_p1_private_provider_abort_v1(context, alias_provider) ==
            ET_P1_STATUS_OK,
        "release-role-alias provider remains abortable");
  check(live_count(context) == before,
        "release-role-alias cleanup restores the live baseline");

  {
    int forged_state = 0;
    void *entry_state;
    void *sibling_state;
    void *first_entry;
    void *second_entry;
    void *sibling_entry;
    void *late_entry;
    int64_t live_baseline = live_count(context);
    int64_t tombstone_baseline = tombstone_count(context);

    check(et_p1_private_state_entry_create_for_state_v1(context, NULL) ==
              ET_P1_STATUS_INVALID_ARGUMENT,
          "state-bound entry creation rejects a null state");
    check(et_p1_private_error_code_v1(context) == ET_P1_CODE_NULL_ARGUMENT,
          "null state-bound entry creation has exact status data");
    check(et_p1_private_state_entry_create_for_state_v1(context,
                                                         &forged_state) ==
              ET_P1_STATUS_INVALID_ARGUMENT,
          "state-bound entry creation rejects a forged state");
    check(et_p1_private_error_code_v1(context) == ET_P1_CODE_FOREIGN_TOKEN,
          "forged state-bound entry creation has exact status data");
    check(et_p1_private_state_entry_create_for_state_v1(context, provider) ==
              ET_P1_STATUS_INVALID_ARGUMENT,
          "state-bound entry creation rejects a wrong-kind token");
    check(et_p1_private_error_code_v1(context) ==
              ET_P1_CODE_WRONG_TOKEN_KIND,
          "wrong-kind state-bound entry creation has exact status data");
    check(live_count(context) == live_baseline,
          "rejected state-bound entry creation is failure-atomic");

    entry_state = created(
        context, et_p1_private_state_dict_create_v1(context),
        "entry-owner state creation succeeds");
    sibling_state = created(
        context, et_p1_private_state_dict_create_v1(context),
        "sibling entry-owner state creation succeeds");
    first_entry = created(
        context,
        et_p1_private_state_entry_create_for_state_v1(context, entry_state),
        "first state-bound entry creation succeeds");
    second_entry = created(
        context,
        et_p1_private_state_entry_create_for_state_v1(context, entry_state),
        "second state-bound entry creation succeeds");
    sibling_entry = created(
        context,
        et_p1_private_state_entry_create_for_state_v1(context, sibling_state),
        "cross-state sibling entry creation succeeds");
    check(et_p1_public_token_kind_v1(first_entry) == ET_P1_TOKEN_STATE_ENTRY &&
              et_p1_public_token_live_v1(first_entry) == 1,
          "state-bound entry has its exact live kind");

    copy = (unsigned char *)malloc(TOKEN_BYTES);
    check(copy != NULL, "entry-owner state copy scratch allocation succeeds");
    memcpy(copy, entry_state, TOKEN_BYTES);
    before = live_count(context);
    check(et_p1_private_state_entry_create_for_state_v1(context, copy) ==
              ET_P1_STATUS_INVALID_ARGUMENT,
          "copied state bytes cannot authorize a bound entry");
    check(live_count(context) == before,
          "copied state entry rejection creates no token");
    free(copy);

    before = live_count(context);
    check(et_p1_private_state_bind_v1(context, entry_state, provider) ==
              ET_P1_STATUS_OK,
          "first state bind revokes its construction entries");
    check(et_p1_public_token_live_v1(first_entry) == 0 &&
              et_p1_public_token_live_v1(second_entry) == 0,
          "first bind revokes every entry bound to that state");
    check(et_p1_public_token_live_v1(sibling_entry) == 1,
          "first bind preserves another state's entry");
    check(live_count(context) == before - 2,
          "first bind removes the exact bound-entry count");
    check(tombstone_count(context) == tombstone_baseline + 2,
          "first bind creates exact bound-entry tombstones");
    before = live_count(context);
    check(et_p1_private_state_bind_v1(context, entry_state, provider) ==
              ET_P1_STATUS_OK,
          "repeated state bind remains idempotent");
    check(live_count(context) == before,
          "repeated state bind changes no entry liveness");

    check(et_p1_private_state_bind_v1(context, sibling_state, provider) ==
              ET_P1_STATUS_OK,
          "sibling state bind succeeds");
    check(et_p1_public_token_live_v1(sibling_entry) == 0,
          "sibling state bind revokes only its own entry");
    late_entry = created(
        context,
        et_p1_private_state_entry_create_for_state_v1(context, sibling_state),
        "bound state can mint a lifetime-bound entry handle");
    check(et_p1_private_state_release_begin_v1(context, sibling_state) ==
                ET_P1_STATUS_OK &&
              et_p1_private_result_i64_v1(context) == 1,
          "state release revokes a post-bind entry");
    check(et_p1_public_token_live_v1(late_entry) == 0,
          "state release invalidates its bound entry handle");
    check(et_p1_private_state_entry_create_for_state_v1(context,
                                                         sibling_state) ==
              ET_P1_STATUS_INVALID_STATE,
          "stale state cannot mint a bound entry");
    check(et_p1_private_state_release_begin_v1(context, entry_state) ==
                ET_P1_STATUS_OK &&
              et_p1_private_result_i64_v1(context) == 1,
          "entry-owner state release succeeds");
    check(live_count(context) == live_baseline,
          "bound-entry bind and release restore the live baseline");
    check(tombstone_count(context) == tombstone_baseline + 6,
          "bound-entry bind and release leave exact tombstones");
  }

  {
    void *revoke_state = created(
        context, et_p1_private_state_dict_create_v1(context),
        "revoke-path state creation succeeds");
    void *revoke_entry;
    void *revoke_tensor;
    int64_t live_baseline;
    int64_t tombstone_baseline;

    check(et_p1_private_state_bind_v1(context, revoke_state, provider) ==
              ET_P1_STATUS_OK,
          "revoke-path state bind succeeds");
    revoke_entry = created(
        context,
        et_p1_private_state_entry_create_for_state_v1(context, revoke_state),
        "revoke-path state entry creation succeeds");
    revoke_tensor = created(
        context, et_p1_private_state_tensor_create_v1(context, revoke_state),
        "revoke-path state tensor creation succeeds");
    live_baseline = live_count(context);
    tombstone_baseline = tombstone_count(context);
    check(et_p1_private_state_revoke_v1(context, revoke_state) ==
              ET_P1_STATUS_OK,
          "state revoke succeeds with bound dependents");
    check(et_p1_public_token_live_v1(revoke_state) == 0 &&
              et_p1_public_token_live_v1(revoke_entry) == 0 &&
              et_p1_public_token_live_v1(revoke_tensor) == 0,
          "state revoke invalidates entry and tensor dependents");
    check(live_count(context) == live_baseline - 3,
          "state revoke removes its exact dependent set");
    check(tombstone_count(context) == tombstone_baseline + 3,
          "state revoke creates exact dependent tombstones");
  }

  before = live_count(context);
  check(et_p1_private_state_bind_v1(context, state, provider_two) ==
            ET_P1_STATUS_INVALID_STATE,
        "conflicting provider rebinding is rejected");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_BINDING_CONFLICT,
        "conflicting provider reports binding-conflict status");
  check(live_count(context) == before,
        "rejected rebinding allocates no registry entry");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_OK &&
            et_p1_private_result_ptr_v1(context) == provider,
        "rejected rebinding leaves the original binding intact");

  check(et_p1_private_state_unbind_v1(context, state) == ET_P1_STATUS_OK,
        "trusted construction cleanup can unbind a state");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_UNSUPPORTED,
        "unbound state lookup fails explicitly");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_UNBOUND_STATE,
        "unbound state has exact status data");
  check(et_p1_private_state_bind_v1(context, state, provider) ==
            ET_P1_STATUS_OK,
        "explicit trusted rebind succeeds");

  child = fork();
  check(child >= 0, "fork probe starts");
  if (child == 0) {
    _exit(et_p1_public_token_live_v1(provider) == 0 &&
                  et_p1_public_token_kind_v1(provider) ==
                      ET_P1_TOKEN_FOREIGN
              ? 0
              : 1);
  }
  check(waitpid(child, &child_status, 0) == child,
        "fork probe is collected");
  check(WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0,
        "forked process cannot reuse parent authority");

  state_tensor = created(
      context, et_p1_private_state_tensor_create_v1(context, state),
      "state-backed tensor identity creation succeeds");
  check(et_p1_public_token_kind_v1(state_tensor) ==
            ET_P1_TOKEN_STATE_TENSOR,
        "state-backed tensor token has its exact kind");
  check(et_p1_private_state_tensor_validate_v1(context, state, state_tensor) ==
            ET_P1_STATUS_OK,
        "state-backed tensor validates against its owner");
  other_state = created(context, et_p1_private_state_dict_create_v1(context),
                        "second state identity creation succeeds");
  check(et_p1_private_state_bind_v1(context, other_state, provider) ==
            ET_P1_STATUS_OK,
        "second state binds to the admitted provider");
  check(et_p1_private_state_tensor_validate_v1(context, other_state,
                                                state_tensor) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "state-backed tensor rejects a cross-state borrow");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_CROSS_STATE,
        "cross-state borrow reports an exact argument status");
  check(et_p1_private_state_release_begin_v1(context, other_state) ==
                ET_P1_STATUS_OK &&
            et_p1_private_result_i64_v1(context) == 1,
        "independent state release succeeds");
  copy = (unsigned char *)malloc(TOKEN_BYTES);
  check(copy != NULL, "state-token copy scratch allocation succeeds");
  memcpy(copy, state, TOKEN_BYTES);
  check(et_p1_private_state_release_begin_v1(context, copy) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "copied state bytes cannot initiate release");
  free(copy);
  saved_nonce_byte = ((unsigned char *)state)[TOKEN_NONCE_OFFSET];
  ((unsigned char *)state)[TOKEN_NONCE_OFFSET] ^= UINT8_C(1);
  check(et_p1_private_state_release_begin_v1(context, state) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "mutated state cannot initiate release");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_TOKEN_INTEGRITY,
        "mutated state release reports token-integrity status");
  ((unsigned char *)state)[TOKEN_NONCE_OFFSET] = saved_nonce_byte;
  check(et_p1_private_state_tensor_validate_v1(context, state, state_tensor) ==
            ET_P1_STATUS_OK,
        "restored state remains live before release");
  before = live_count(context);
  check(et_p1_private_state_release_begin_v1(context, state) ==
            ET_P1_STATUS_OK,
        "state release begins exactly once");
  check(et_p1_private_result_i64_v1(context) == 1,
        "first state release reports a live transition");
  check(et_p1_public_token_live_v1(state) == 0,
        "revoked state is not live");
  check(et_p1_public_token_live_v1(state_tensor) == 0,
        "state release invalidates state-backed tensors");
  check(et_p1_private_state_tensor_validate_v1(context, state, state_tensor) ==
            ET_P1_STATUS_INVALID_STATE,
        "stale state-backed tensor cannot be validated");
  check(et_p1_public_token_kind_v1(state) == -ET_P1_TOKEN_STATE_DICT,
        "revoked state retains only a stale kind observation");
  check(live_count(context) == before - 2,
        "state disposal removes state and dependent tensor entries");
  check(et_p1_private_state_release_begin_v1(context, state) ==
            ET_P1_STATUS_OK &&
            et_p1_private_result_i64_v1(context) == 0,
        "repeat state release is idempotent");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_INVALID_STATE,
        "stale state lookup fails explicitly");
  saved_nonce_byte = ((unsigned char *)state)[TOKEN_NONCE_OFFSET];
  ((unsigned char *)state)[TOKEN_NONCE_OFFSET] ^= UINT8_C(1);
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "mutated dead state is invalid argument, not stale state");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_TOKEN_INTEGRITY,
        "mutated dead state reports token integrity before liveness");
  check(et_p1_public_token_kind_v1(state) == ET_P1_TOKEN_FOREIGN,
        "mutated dead state has no public stale-kind authority");
  ((unsigned char *)state)[TOKEN_NONCE_OFFSET] = saved_nonce_byte;
  check(et_p1_public_token_kind_v1(state) == -ET_P1_TOKEN_STATE_DICT,
        "restored dead state retains its exact stale kind");
  check(et_p1_private_state_release_begin_v1(context, state_tensor) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "dead state tensor used as state is wrong-kind argument");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_WRONG_TOKEN_KIND,
        "dead wrong-kind token reports kind before liveness");
  copy = (unsigned char *)malloc(TOKEN_BYTES);
  check(copy != NULL, "dead-token copy scratch allocation succeeds");
  memcpy(copy, state, TOKEN_BYTES);
  check(et_p1_private_state_release_begin_v1(context, copy) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "copied dead state remains foreign");
  memcpy(copy, state_tensor, TOKEN_BYTES);
  check(et_p1_private_state_release_begin_v1(context, copy) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "copied dead state tensor remains foreign");
  free(copy);
  saved_nonce_byte = ((unsigned char *)state_tensor)[TOKEN_NONCE_OFFSET];
  ((unsigned char *)state_tensor)[TOKEN_NONCE_OFFSET] ^= UINT8_C(1);
  check(et_p1_public_token_kind_v1(state_tensor) == ET_P1_TOKEN_FOREIGN,
        "mutated dead state tensor has no public stale-kind authority");
  check(et_p1_private_state_release_begin_v1(context, state_tensor) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "mutated dead state tensor is invalid argument");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_TOKEN_INTEGRITY,
        "mutated dead tensor reports integrity before kind and liveness");
  ((unsigned char *)state_tensor)[TOKEN_NONCE_OFFSET] = saved_nonce_byte;
  check(et_p1_private_context_release_v1(context) ==
            ET_P1_STATUS_INVALID_STATE,
        "context release refuses live identities");

  (void)printf("P1 identity PASS: %d checks\n", checks);
  return 0;
}
