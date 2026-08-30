#define ET_P1_PRIVATE_API 1
#include "p1_identity_internal.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
  void *created_callbacks[7];
  int created = 0;
  int64_t baseline;
  int64_t tombstones;
  const unsigned char embedded_nul_id[3] = {'a', 0u, 'b'};

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
  check(et_p1_private_provider_create_v1(
            context, "oversized",
            (int64_t)ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES + 1) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "provider identity length never truncates");
  check(live_count(context) == baseline,
        "invalid provider byte spans publish no identity");

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

  (void)printf("P1 failpoint PASS: %d checks\n", checks);
  return 0;
}
