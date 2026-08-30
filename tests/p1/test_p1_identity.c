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

int main(void) {
  void *context;
  void *provider;
  void *provider_two;
  void *callbacks[7];
  void *callbacks_two[7];
  void *state;
  void *temporary_callback;
  unsigned char *copy;
  unsigned char saved_nonce_byte;
  int foreign = 0;
  int64_t before;
  pid_t child;
  int child_status;
  char provider_id[] = "fixture-v1";

  check(et_p1_public_identity_abi_major_v1() == 1,
        "ABI major is fixed");
  check(et_p1_public_identity_abi_minor_v1() == 0,
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
  check(seal_provider(context, provider_two, callbacks_two) ==
            ET_P1_STATUS_OK,
        "second provider snapshot seals");
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

  before = live_count(context);
  check(et_p1_private_state_revoke_v1(context, state) == ET_P1_STATUS_OK,
        "temporary state identity is revocable");
  check(et_p1_public_token_live_v1(state) == 0,
        "revoked state is not live");
  check(et_p1_public_token_kind_v1(state) == -ET_P1_TOKEN_STATE_DICT,
        "revoked state retains only a stale kind observation");
  check(live_count(context) == before - 1,
        "state disposal removes one live registry entry");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_INVALID_STATE,
        "stale state lookup fails explicitly");
  check(et_p1_private_context_release_v1(context) ==
            ET_P1_STATUS_INVALID_STATE,
        "context release refuses live identities");

  (void)printf("P1 identity PASS: %d checks\n", checks);
  return 0;
}
