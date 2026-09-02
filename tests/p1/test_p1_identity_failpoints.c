#define ET_P1_PRIVATE_API 1
#define ET_P1_TEST_HOOKS 1
#include "p1_identity_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>

void *__real_calloc(size_t count, size_t size);
ssize_t __real_getrandom(void *buffer, size_t length, unsigned int flags);

static int fail_calloc_after = -1;
static int fail_entropy_after = -1;
static int checks;

void *__wrap_calloc(size_t count, size_t size) {
  if (fail_calloc_after == 0) {
    fail_calloc_after = -1;
    return NULL;
  }
  if (fail_calloc_after > 0) {
    fail_calloc_after--;
  }
  return __real_calloc(count, size);
}

ssize_t __wrap_getrandom(void *buffer, size_t length, unsigned int flags) {
  if (fail_entropy_after == 0) {
    fail_entropy_after = -1;
    errno = EIO;
    return -1;
  }
  if (fail_entropy_after > 0) {
    fail_entropy_after--;
  }
  return __real_getrandom(buffer, length, flags);
}

static void check(int condition, const char *label) {
  if (!condition) {
    (void)fprintf(stderr, "P1 failpoint FAIL: %s\n", label);
    exit(1);
  }
  checks++;
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
  void *context = et_p1_private_context_create_v1();
  void *provider;
  void *release_callback;
  void *state;
  void *state_entry;
  void *state_tensor;
  void *created_callbacks[7];
  int created = 0;
  int64_t baseline;
  int64_t tombstones;
  const unsigned char embedded_nul_id[3] = {'a', 0u, 'b'};
  unsigned char maximum_id[ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES];

  (void)memset(maximum_id, 'm', sizeof(maximum_id));

  check(context != NULL, "private context creation succeeds");
  baseline = live_count(context);
  tombstones = tombstone_count(context);

  fail_calloc_after = 0;
  check(et_p1_private_provider_create_v1(context, "alloc", 5) ==
            ET_P1_STATUS_INTERNAL,
        "provider allocation failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "provider allocation failure has exact status data");
  check(et_p1_private_result_ptr_v1(context) == NULL,
        "failed provider creation publishes no result");
  check(live_count(context) == baseline,
        "provider allocation failure leaves live count unchanged");
  check(tombstone_count(context) == tombstones,
        "provider allocation failure creates no tombstone");

  fail_calloc_after = 1;
  check(et_p1_private_provider_create_v1(context, "record", 6) ==
            ET_P1_STATUS_INTERNAL,
        "private registry-record allocation failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "registry-record allocation failure has exact status data");
  check(et_p1_private_result_ptr_v1(context) == NULL,
        "registry-record failure publishes no caller token");
  check(live_count(context) == baseline,
        "registry-record allocation failure leaves live count unchanged");
  check(tombstone_count(context) == tombstones,
        "registry-record allocation failure creates no tombstone");

  fail_entropy_after = 0;
  check(et_p1_private_provider_create_v1(context, "entropy", 7) ==
            ET_P1_STATUS_UNSUPPORTED,
        "provider entropy failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ENTROPY_UNAVAILABLE,
        "provider entropy failure has exact status data");
  check(et_p1_private_result_ptr_v1(context) == NULL,
        "entropy failure publishes no provider result");
  check(live_count(context) == baseline,
        "provider entropy failure leaves live count unchanged");
  check(tombstone_count(context) == tombstones,
        "provider entropy failure creates no tombstone");

  check(et_p1_private_provider_create_v1(context, NULL, 1) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "nonnull span requirement is explicit");
  check(et_p1_private_provider_create_v1(context, "negative", -1) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "negative provider identity length is rejected");
  check(et_p1_private_provider_create_v1(
            context, "oversized",
            (int64_t)ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES + 1) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "provider identity length never truncates");
  check(live_count(context) == baseline,
        "invalid provider byte spans publish no identity");

  check(et_p1_private_provider_create_v1(
            context, maximum_id,
            (int64_t)ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES) ==
            ET_P1_STATUS_OK,
        "maximum provider identity length is accepted exactly");
  provider = et_p1_private_result_ptr_v1(context);
  (void)memset(maximum_id, 'x', sizeof(maximum_id));
  check(et_p1_public_token_live_v1(provider) == 1,
        "caller mutation cannot revoke a copied maximum identity");
  check(et_p1_private_provider_abort_v1(context, provider) ==
            ET_P1_STATUS_OK,
        "maximum provider identity can be aborted before publication");
  check(live_count(context) == baseline,
        "maximum provider identity abort restores live baseline");
  tombstones++;
  check(tombstone_count(context) == tombstones,
        "maximum provider identity abort leaves one tombstone");

  check(et_p1_private_provider_create_v1(context, embedded_nul_id, 3) ==
            ET_P1_STATUS_OK,
        "provider identity is exact bytes, not a C string");
  provider = et_p1_private_result_ptr_v1(context);
  check(et_p1_private_provider_abort_v1(context, provider) ==
            ET_P1_STATUS_OK,
        "exact-byte provider can be aborted before publication");
  check(live_count(context) == baseline,
        "exact-byte provider abort restores live baseline");
  tombstones++;
  check(tombstone_count(context) == tombstones,
        "exact-byte provider abort leaves one permanent tombstone");

  check(et_p1_private_provider_create_v1(context, "partial", 7) ==
            ET_P1_STATUS_OK,
        "partial-admission provider token is created");
  provider = et_p1_private_result_ptr_v1(context);
  while (created < 3) {
    check(et_p1_private_callback_identity_create_v1(context) ==
              ET_P1_STATUS_OK,
          "partial callback identity creation succeeds");
    created_callbacks[created] = et_p1_private_result_ptr_v1(context);
    created++;
  }
  fail_calloc_after = 0;
  check(et_p1_private_callback_identity_create_v1(context) ==
            ET_P1_STATUS_INTERNAL,
        "injected partial callback allocation failure is explicit");
  check(live_count(context) == baseline + 4,
        "partial admission owns only provider plus completed callbacks");
  while (created > 0) {
    created--;
    check(et_p1_private_callback_identity_revoke_v1(
              context, created_callbacks[created]) == ET_P1_STATUS_OK,
          "partial callback cleanup succeeds");
  }
  check(et_p1_private_provider_abort_v1(context, provider) ==
            ET_P1_STATUS_OK,
        "partial provider admission abort succeeds");
  check(live_count(context) == baseline,
        "partial callback/provider cleanup restores exact live baseline");
  tombstones += 4;
  check(tombstone_count(context) == tombstones,
        "partial cleanup identities remain permanent tombstones");

  check(et_p1_private_provider_create_v1(context, "release-nonalloc", 16) ==
            ET_P1_STATUS_OK,
        "release-nonallocation provider identity is created");
  provider = et_p1_private_result_ptr_v1(context);
  for (created = 0; created < 7; created++) {
    check(et_p1_private_callback_identity_create_v1(context) ==
              ET_P1_STATUS_OK,
          "release-nonallocation callback identity is created");
    created_callbacks[created] = et_p1_private_result_ptr_v1(context);
  }
  check(et_p1_private_callback_identity_create_v1(context) ==
            ET_P1_STATUS_OK,
        "release-nonallocation release callback identity is created");
  release_callback = et_p1_private_result_ptr_v1(context);
  check(et_p1_private_provider_seal_release_v1(
            context, provider, created_callbacks[0], created_callbacks[1],
            created_callbacks[2], created_callbacks[3], created_callbacks[4],
            created_callbacks[5], created_callbacks[6], release_callback) ==
            ET_P1_STATUS_OK,
        "release-nonallocation provider snapshot seals");
  check(et_p1_private_state_dict_create_v1(context) == ET_P1_STATUS_OK,
        "release-nonallocation state identity is created");
  state = et_p1_private_result_ptr_v1(context);
  baseline = live_count(context);
  tombstones = tombstone_count(context);
  fail_calloc_after = 0;
  check(et_p1_private_state_entry_create_for_state_v1(context, state) ==
            ET_P1_STATUS_INTERNAL,
        "state-bound entry token allocation failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "state-bound entry token allocation failure has exact status data");
  check(live_count(context) == baseline,
        "state-bound entry token allocation failure preserves live count");
  check(tombstone_count(context) == tombstones,
        "state-bound entry token allocation failure creates no tombstone");
  fail_calloc_after = 1;
  check(et_p1_private_state_entry_create_for_state_v1(context, state) ==
            ET_P1_STATUS_INTERNAL,
        "state-bound entry record allocation failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "state-bound entry record allocation failure has exact status data");
  check(live_count(context) == baseline,
        "state-bound entry record allocation failure preserves live count");
  check(tombstone_count(context) == tombstones,
        "state-bound entry record allocation failure creates no tombstone");
  check(et_p1_private_state_entry_create_for_state_v1(context, state) ==
            ET_P1_STATUS_OK,
        "pre-bind state entry identity is created");
  state_entry = et_p1_private_result_ptr_v1(context);
  baseline = live_count(context);
  tombstones = tombstone_count(context);
  check(et_p1_test_state_bind_fail_next_v1() == ET_P1_STATUS_OK,
        "state-bind failure is armed");
  check(et_p1_private_state_bind_v1(context, state, provider) ==
            ET_P1_STATUS_INTERNAL,
        "injected state-bind failure is explicit");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "injected state-bind failure has exact status data");
  check(et_p1_public_token_live_v1(state_entry) == 1,
        "failed state bind preserves its bound construction entry");
  check(et_p1_private_state_provider_v1(context, state) ==
            ET_P1_STATUS_UNSUPPORTED,
        "failed state bind leaves the state unbound");
  check(live_count(context) == baseline,
        "failed state bind changes no dependent liveness");
  check(tombstone_count(context) == tombstones,
        "failed state bind creates no dependent tombstone");
  check(et_p1_private_state_bind_v1(context, state, provider) ==
            ET_P1_STATUS_OK,
        "release-nonallocation state bind retries");
  check(et_p1_public_token_live_v1(state_entry) == 0,
        "successful retry revokes the pre-bind state entry");
  check(live_count(context) == baseline - 1,
        "successful retry removes the exact pre-bind entry count");
  check(tombstone_count(context) == tombstones + 1,
        "successful retry creates one state-entry tombstone");
  check(et_p1_private_state_tensor_create_v1(context, state) ==
            ET_P1_STATUS_OK,
        "release-nonallocation state tensor identity is created");
  state_tensor = et_p1_private_result_ptr_v1(context);
  check(et_p1_public_token_live_v1(state_tensor) == 1,
        "release-nonallocation state tensor starts live");
  baseline = live_count(context);
  tombstones = tombstone_count(context);
  fail_calloc_after = 0;
  check(et_p1_private_state_release_begin_v1(context, state) ==
            ET_P1_STATUS_OK,
        "first state release succeeds with allocation failure armed");
  check(et_p1_private_result_i64_v1(context) == 1,
        "first state release reports its live transition");
  check(live_count(context) == baseline - 2,
        "first state release removes state and dependent handle");
  check(tombstone_count(context) == tombstones + 2,
        "first state release creates exact state and handle tombstones");
  check(et_p1_private_state_release_begin_v1(context, state) ==
            ET_P1_STATUS_OK,
        "repeated state release succeeds with allocation failure armed");
  check(et_p1_private_result_i64_v1(context) == 0,
        "repeated state release reports no transition");
  check(live_count(context) == baseline - 2,
        "repeated state release changes no token liveness");
  check(tombstone_count(context) == tombstones + 2,
        "repeated state release changes no tombstone count");
  check(et_p1_private_provider_create_v1(context, "consume-failpoint", 17) ==
            ET_P1_STATUS_INTERNAL,
        "state release leaves the allocation failpoint armed");
  check(et_p1_private_error_code_v1(context) ==
            ET_P1_CODE_ALLOCATION_FAILED,
        "post-release allocation failure has exact status data");
  check(et_p1_private_result_ptr_v1(context) == NULL,
        "post-release allocation failure publishes no identity");
  check(live_count(context) == baseline - 2,
        "post-release allocation failure preserves the live count");
  check(tombstone_count(context) == tombstones + 2,
        "post-release allocation failure preserves tombstones");

  (void)printf("P1 failpoint PASS: %d checks\n", checks);
  return 0;
}
