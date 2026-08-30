#if defined(ET_P1_TRUSTED_BUILD)
#define ET_P1_PRIVATE_API 1
#endif
#include "p1_identity_internal.h"

#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__GNUC__) || defined(__clang__)
#define ET_P1_PUBLIC __attribute__((visibility("default")))
#define ET_P1_PRIVATE __attribute__((visibility("hidden")))
#else
#define ET_P1_PUBLIC
#define ET_P1_PRIVATE
#endif

typedef struct et_p1_context et_p1_context;

typedef struct et_p1_token {
  uint64_t nonce_lo;
  uint64_t nonce_hi;
  uint8_t reserved[248];
} et_p1_token;

typedef struct et_p1_record {
  struct et_p1_record *next;
  et_p1_token *token;
  et_p1_context *owner;
  struct et_p1_record *binding;
  struct et_p1_record *callbacks[ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT];
  int64_t kind;
  uint64_t expected_nonce_lo;
  uint64_t expected_nonce_hi;
  int64_t origin_pid;
  uint32_t provider_id_bytes;
  uint8_t live;
  uint8_t sealed;
  uint8_t reserved[2];
  uint8_t provider_id[ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES];
} et_p1_record;

#if defined(ET_P1_TRUSTED_BUILD)
struct et_p1_context {
  struct et_p1_context *next;
  et_p1_identity_error_v1 error;
  void *result_ptr;
  int64_t result_i64;
  uint64_t nonce_lo;
  uint64_t nonce_hi;
  uint64_t expected_nonce_lo;
  uint64_t expected_nonce_hi;
  int64_t origin_pid;
  uint8_t live;
  uint8_t reserved[7];
};
#endif

_Static_assert(sizeof(et_p1_token) == 264u, "P1 token layout changed");
_Static_assert(offsetof(et_p1_token, nonce_lo) == 0u &&
                   offsetof(et_p1_token, nonce_hi) == 8u,
               "P1 caller token nonce layout changed");
_Static_assert(sizeof(et_p1_record) == 256u,
               "P1 private registry record layout changed");
_Static_assert(offsetof(et_p1_record, callbacks) == 32u &&
                   offsetof(et_p1_record, kind) == 88u &&
                   offsetof(et_p1_record, provider_id) == 128u,
               "P1 private registry record offsets changed");
#if defined(ET_P1_TRUSTED_BUILD)
_Static_assert(sizeof(et_p1_context) == 344u, "P1 context layout changed");
#endif

static et_p1_record *records;
#if defined(ET_P1_TRUSTED_BUILD)
static et_p1_context *contexts;
static uint8_t private_context_claimed;
#if defined(ET_P1_TEST_HOOKS)
static int64_t test_callback_successes_before_failure = INT64_C(-1);
#endif

static et_p1_record *find_record(const void *candidate) {
  et_p1_record *cursor = records;
  while (cursor != NULL) {
    if ((const void *)cursor->token == candidate) {
      return cursor;
    }
    cursor = cursor->next;
  }
  return NULL;
}

static int fill_entropy(void *destination, size_t size) {
  unsigned char *cursor = (unsigned char *)destination;
  while (size > 0u) {
    const ssize_t count = getrandom(cursor, size, 0u);
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      return 0;
    }
    if (count == 0) {
      return 0;
    }
    cursor += (size_t)count;
    size -= (size_t)count;
  }
  return 1;
}

static int context_integrity(const et_p1_context *context) {
  return context != NULL && context->live != 0u &&
         context->origin_pid == (int64_t)getpid() &&
         context->nonce_lo == context->expected_nonce_lo &&
         context->nonce_hi == context->expected_nonce_hi;
}
#endif

static int token_integrity(const et_p1_record *record) {
  return record != NULL && record->token != NULL && record->live != 0u &&
         record->origin_pid == (int64_t)getpid() &&
         record->token->nonce_lo == record->expected_nonce_lo &&
         record->token->nonce_hi == record->expected_nonce_hi;
}

#if !defined(ET_P1_TRUSTED_BUILD)
static et_p1_record *find_record(const void *candidate) {
  et_p1_record *cursor = records;
  while (cursor != NULL) {
    if ((const void *)cursor->token == candidate) {
      return cursor;
    }
    cursor = cursor->next;
  }
  return NULL;
}
#endif

#if defined(ET_P1_TRUSTED_BUILD)
static int token_nonce_exists(uint64_t lo, uint64_t hi) {
  const et_p1_record *cursor = records;
  while (cursor != NULL) {
    if (cursor->expected_nonce_lo == lo && cursor->expected_nonce_hi == hi) {
      return 1;
    }
    cursor = cursor->next;
  }
  return 0;
}

static int create_unique_nonce(uint64_t *lo, uint64_t *hi) {
  uint64_t values[2];
  do {
    if (!fill_entropy(values, sizeof(values))) {
      return 0;
    }
  } while ((values[0] == 0u && values[1] == 0u) ||
           token_nonce_exists(values[0], values[1]));
  *lo = values[0];
  *hi = values[1];
  return 1;
}

static void clear_error(et_p1_context *context) {
  if (context != NULL) {
    memset(&context->error, 0, sizeof(context->error));
    context->result_ptr = NULL;
    context->result_i64 = 0;
  }
}

static int64_t set_error(et_p1_context *context, int64_t category,
                         int64_t code, const char *operation,
                         const char *message) {
  if (context != NULL) {
    clear_error(context);
    context->error.category = category;
    context->error.code = code;
    (void)snprintf(context->error.operation,
                   sizeof(context->error.operation), "%s", operation);
    (void)snprintf(context->error.message, sizeof(context->error.message),
                   "%s", message);
  }
  return category;
}

static et_p1_context *find_context(const void *candidate) {
  et_p1_context *cursor = contexts;
  while (cursor != NULL) {
    if ((const void *)cursor == candidate) {
      return cursor;
    }
    cursor = cursor->next;
  }
  return NULL;
}

static et_p1_context *require_context(void *candidate) {
  et_p1_context *context = find_context(candidate);
  if (!context_integrity(context)) {
    return NULL;
  }
  clear_error(context);
  return context;
}

static et_p1_record *require_token(et_p1_context *context,
                                   const void *candidate, int64_t kind,
                                   const char *operation) {
  et_p1_record *record;
  if (candidate == NULL) {
    (void)set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                    ET_P1_CODE_NULL_ARGUMENT, operation,
                    "token is null");
    return NULL;
  }
  record = find_record(candidate);
  if (record == NULL) {
    (void)set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                    ET_P1_CODE_FOREIGN_TOKEN, operation,
                    "token identity is foreign");
    return NULL;
  }
  if (record->live == 0u) {
    (void)set_error(context, ET_P1_STATUS_INVALID_STATE,
                    ET_P1_CODE_STALE_TOKEN, operation,
                    "token identity is stale or revoked");
    return NULL;
  }
  if (record->origin_pid != (int64_t)getpid()) {
    (void)set_error(context, ET_P1_STATUS_UNSUPPORTED,
                    ET_P1_CODE_FORKED_PROCESS, operation,
                    "token identity belongs to a different process");
    return NULL;
  }
  if (!token_integrity(record)) {
    (void)set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                    ET_P1_CODE_TOKEN_INTEGRITY, operation,
                    "token identity is foreign or invalid");
    return NULL;
  }
  if (record->owner != context) {
    (void)set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                    ET_P1_CODE_CROSS_CONTEXT, operation,
                    "token belongs to a different private context");
    return NULL;
  }
  if (record->kind != kind) {
    (void)set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                    ET_P1_CODE_WRONG_TOKEN_KIND, operation,
                    "token has the wrong identity kind");
    return NULL;
  }
  return record;
}

static int64_t create_token(void *candidate, int64_t kind,
                            const char *operation, const void *provider_id,
                            size_t provider_id_bytes) {
  et_p1_context *context = require_context(candidate);
  et_p1_token *token;
  et_p1_record *record;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  {
    size_t live_count = 0u;
    const et_p1_record *cursor = records;
    while (cursor != NULL) {
      if (cursor->owner == context && cursor->live != 0u) {
        live_count++;
      }
      cursor = cursor->next;
    }
    if (live_count >= ET_P1_IDENTITY_MAX_LIVE_TOKENS) {
      return set_error(context, ET_P1_STATUS_UNSUPPORTED,
                       ET_P1_CODE_CAPACITY_EXCEEDED, operation,
                       "opaque identity token capacity is exhausted");
    }
  }
  token = (et_p1_token *)calloc(1u, sizeof(*token));
  if (token == NULL) {
    return set_error(context, ET_P1_STATUS_INTERNAL,
                     ET_P1_CODE_ALLOCATION_FAILED, operation,
                     "cannot allocate opaque identity token");
  }
  record = (et_p1_record *)calloc(1u, sizeof(*record));
  if (record == NULL) {
    free(token);
    return set_error(context, ET_P1_STATUS_INTERNAL,
                     ET_P1_CODE_ALLOCATION_FAILED, operation,
                     "cannot allocate private identity registry record");
  }
  record->token = token;
  record->owner = context;
  record->kind = kind;
  record->origin_pid = (int64_t)getpid();
  if (!create_unique_nonce(&record->expected_nonce_lo,
                           &record->expected_nonce_hi)) {
    free(record);
    free(token);
    return set_error(context, ET_P1_STATUS_UNSUPPORTED,
                     ET_P1_CODE_ENTROPY_UNAVAILABLE, operation,
                     "cannot obtain process-local identity entropy");
  }
  token->nonce_lo = record->expected_nonce_lo;
  token->nonce_hi = record->expected_nonce_hi;
  record->live = 1u;
  if (provider_id_bytes != 0u) {
    memcpy(record->provider_id, provider_id, provider_id_bytes);
  }
  record->provider_id_bytes = (uint32_t)provider_id_bytes;
  record->next = records;
  records = record;
  context->result_ptr = token;
  return ET_P1_STATUS_OK;
}
#endif

ET_P1_PUBLIC int64_t et_p1_public_identity_abi_major_v1(void) {
  return (int64_t)ET_P1_IDENTITY_ABI_MAJOR;
}

ET_P1_PUBLIC int64_t et_p1_public_identity_abi_minor_v1(void) {
  return (int64_t)ET_P1_IDENTITY_ABI_MINOR;
}

ET_P1_PUBLIC int64_t et_p1_public_token_kind_v1(const void *token) {
  const et_p1_record *entry = find_record(token);
  if (entry == NULL) {
    return ET_P1_TOKEN_FOREIGN;
  }
  if (entry->live == 0u) {
    return -entry->kind;
  }
  return token_integrity(entry) ? entry->kind : ET_P1_TOKEN_FOREIGN;
}

ET_P1_PUBLIC int64_t et_p1_public_token_live_v1(const void *token) {
  const et_p1_record *entry = find_record(token);
  return token_integrity(entry) ? INT64_C(1) : INT64_C(0);
}

#if defined(ET_P1_TRUSTED_BUILD)
ET_P1_PRIVATE void *et_p1_private_context_create_v1(void) {
  uint64_t nonce[2];
  et_p1_context *context = (et_p1_context *)calloc(1u, sizeof(*context));
  if (private_context_claimed != 0u || context == NULL) {
    free(context);
    return NULL;
  }
  if (!fill_entropy(nonce, sizeof(nonce)) ||
      (nonce[0] == 0u && nonce[1] == 0u)) {
    free(context);
    return NULL;
  }
  private_context_claimed = 1u;
  context->nonce_lo = nonce[0];
  context->nonce_hi = nonce[1];
  context->expected_nonce_lo = nonce[0];
  context->expected_nonce_hi = nonce[1];
  context->origin_pid = (int64_t)getpid();
  context->live = 1u;
  context->next = contexts;
  contexts = context;
  return context;
}

ET_P1_PRIVATE int64_t et_p1_private_context_release_v1(void *candidate) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *record;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  record = records;
  while (record != NULL) {
    if (record->owner == context && record->live != 0u) {
      return set_error(context, ET_P1_STATUS_INVALID_STATE,
                       ET_P1_CODE_LIVE_ENTRIES,
                       "p1-private-context-release",
                       "private context still owns live tokens");
    }
    record = record->next;
  }
  context->live = 0u;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t
et_p1_private_error_category_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? ET_P1_STATUS_INVALID_ARGUMENT
                         : context->error.category;
}

ET_P1_PRIVATE int64_t et_p1_private_error_code_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? ET_P1_CODE_FOREIGN_CONTEXT : context->error.code;
}

ET_P1_PRIVATE const char *
et_p1_private_error_operation_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? "p1-private-context" : context->error.operation;
}

ET_P1_PRIVATE const char *
et_p1_private_error_message_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? "foreign P1 private context"
                         : context->error.message;
}

ET_P1_PRIVATE void *et_p1_private_result_ptr_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? NULL : context->result_ptr;
}

ET_P1_PRIVATE int64_t et_p1_private_result_i64_v1(const void *candidate) {
  const et_p1_context *context = find_context(candidate);
  return context == NULL ? INT64_C(0) : context->result_i64;
}

#define ET_P1_DEFINE_CREATE(function_name, token_kind, operation_name)          \
  ET_P1_PRIVATE int64_t function_name(void *context) {                         \
    return create_token(context, token_kind, operation_name, NULL, 0u);        \
  }

ET_P1_DEFINE_CREATE(et_p1_private_module_create_v1, ET_P1_TOKEN_MODULE,
                    "p1-module-create")
ET_P1_DEFINE_CREATE(et_p1_private_parameter_handle_create_v1,
                    ET_P1_TOKEN_PARAMETER_HANDLE, "p1-parameter-handle-create")
ET_P1_DEFINE_CREATE(et_p1_private_parameter_tree_create_v1,
                    ET_P1_TOKEN_PARAMETER_TREE, "p1-parameter-tree-create")
ET_P1_DEFINE_CREATE(et_p1_private_state_dict_create_v1,
                    ET_P1_TOKEN_STATE_DICT, "p1-state-dict-create")
ET_P1_DEFINE_CREATE(et_p1_private_state_entry_create_v1,
                    ET_P1_TOKEN_STATE_ENTRY, "p1-state-entry-create")

#undef ET_P1_DEFINE_CREATE

ET_P1_PRIVATE int64_t
et_p1_private_callback_identity_create_v1(void *candidate) {
#if defined(ET_P1_TEST_HOOKS)
  et_p1_context *context = require_context(candidate);
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  if (test_callback_successes_before_failure == 0) {
    test_callback_successes_before_failure = INT64_C(-1);
    return set_error(context, ET_P1_STATUS_INTERNAL,
                     ET_P1_CODE_ALLOCATION_FAILED,
                     "p1-callback-identity-create",
                     "injected callback identity allocation failure");
  }
  if (test_callback_successes_before_failure > 0) {
    test_callback_successes_before_failure--;
  }
#endif
  return create_token(candidate, ET_P1_TOKEN_CALLBACK_IDENTITY,
                      "p1-callback-identity-create", NULL, 0u);
}

#if defined(ET_P1_TEST_HOOKS)
ET_P1_PRIVATE int64_t
et_p1_test_callback_fail_after_v1(int64_t successful_creations) {
  test_callback_successes_before_failure = successful_creations;
  return ET_P1_STATUS_OK;
}

static int64_t test_count_records(int live) {
  const et_p1_record *record = records;
  int64_t count = 0;
  while (record != NULL) {
    if ((record->live != 0u) == (live != 0)) {
      count++;
    }
    record = record->next;
  }
  return count;
}

ET_P1_PRIVATE int64_t et_p1_test_live_entry_count_v1(void) {
  return test_count_records(1);
}

ET_P1_PRIVATE int64_t et_p1_test_tombstone_count_v1(void) {
  return test_count_records(0);
}
#endif

ET_P1_PRIVATE int64_t et_p1_private_provider_create_v1(
    void *candidate, const void *provider_id, int64_t provider_id_bytes) {
  et_p1_context *context = require_context(candidate);
  int64_t status;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  if (provider_id_bytes < 0 ||
      provider_id_bytes > (int64_t)ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES ||
      (provider_id_bytes != 0 && provider_id == NULL)) {
    return set_error(context, ET_P1_STATUS_INVALID_ARGUMENT,
                     ET_P1_CODE_INVALID_TEXT, "p1-provider-create",
                     "provider identity byte span is invalid or exceeds the bound");
  }
  status = create_token(context, ET_P1_TOKEN_PROVIDER, "p1-provider-create",
                        provider_id, (size_t)provider_id_bytes);
  if (status != ET_P1_STATUS_OK) {
    return status;
  }
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_provider_abort_v1(
    void *candidate, void *provider) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  token = require_token(context, provider, ET_P1_TOKEN_PROVIDER,
                        "p1-provider-abort");
  if (token == NULL) {
    return context->error.category;
  }
  if (token->sealed != 0u) {
    return set_error(context, ET_P1_STATUS_INVALID_STATE,
                     ET_P1_CODE_ALREADY_SEALED, "p1-provider-abort",
                     "sealed or published provider identity cannot be aborted");
  }
  token->live = 0u;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_callback_identity_revoke_v1(
    void *candidate, void *identity) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  token = require_token(context, identity, ET_P1_TOKEN_CALLBACK_IDENTITY,
                        "p1-callback-identity-revoke");
  if (token == NULL) {
    return context->error.category;
  }
  if (token->binding != NULL) {
    return set_error(context, ET_P1_STATUS_INVALID_STATE,
                     ET_P1_CODE_ALREADY_SEALED,
                     "p1-callback-identity-revoke",
                     "sealed provider callback identity cannot be revoked");
  }
  token->live = 0u;
  return ET_P1_STATUS_OK;
}

static int64_t require_callback_set(et_p1_context *context,
                                    void *const callbacks[],
                                    const char *operation) {
  size_t index;
  size_t previous;
  for (index = 0u; index < ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT; index++) {
    if (require_token(context, callbacks[index],
                      ET_P1_TOKEN_CALLBACK_IDENTITY, operation) == NULL) {
      return context->error.category;
    }
    for (previous = 0u; previous < index; previous++) {
      if (callbacks[previous] == callbacks[index]) {
        return set_error(context, ET_P1_STATUS_INVALID_STATE,
                         ET_P1_CODE_SNAPSHOT_MISMATCH, operation,
                         "provider callback identities must be distinct");
      }
    }
  }
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_provider_seal_v1(
    void *candidate, void *provider, void *describe, void *clone,
    void *storage_identical, void *value_equal, void *device_equal,
    void *prepare, void *commit) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *provider_token;
  void *callbacks[ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT] = {
      describe, clone, storage_identical, value_equal, device_equal, prepare,
      commit};
  size_t index;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  provider_token = require_token(context, provider, ET_P1_TOKEN_PROVIDER,
                                 "p1-provider-seal");
  if (provider_token == NULL) {
    return context->error.category;
  }
  if (require_callback_set(context, callbacks, "p1-provider-seal") != 0) {
    return context->error.category;
  }
  if (provider_token->sealed != 0u) {
    for (index = 0u; index < ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT; index++) {
      if (provider_token->callbacks[index] != find_record(callbacks[index])) {
        return set_error(context, ET_P1_STATUS_INVALID_STATE,
                         ET_P1_CODE_ALREADY_SEALED, "p1-provider-seal",
                         "provider has a conflicting immutable snapshot");
      }
    }
    return ET_P1_STATUS_OK;
  }
  for (index = 0u; index < ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT; index++) {
    et_p1_record *callback = find_record(callbacks[index]);
    if (callback == NULL || callback->binding != NULL) {
      return set_error(context, ET_P1_STATUS_INVALID_STATE,
                       ET_P1_CODE_SNAPSHOT_MISMATCH, "p1-provider-seal",
                       "callback identity is already sealed into a provider");
    }
  }
  for (index = 0u; index < ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT; index++) {
    et_p1_record *callback = find_record(callbacks[index]);
    provider_token->callbacks[index] = callback;
    callback->binding = provider_token;
  }
  provider_token->sealed = 1u;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_provider_snapshot_matches_v1(
    void *candidate, const void *provider, const void *describe,
    const void *clone, const void *storage_identical, const void *value_equal,
    const void *device_equal, const void *prepare, const void *commit) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *provider_token;
  const void *callbacks[ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT] = {
      describe, clone, storage_identical, value_equal, device_equal, prepare,
      commit};
  size_t index;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  provider_token = require_token(context, provider, ET_P1_TOKEN_PROVIDER,
                                 "p1-provider-snapshot-matches");
  if (provider_token == NULL) {
    return context->error.category;
  }
  if (provider_token->sealed == 0u) {
    return set_error(context, ET_P1_STATUS_INVALID_STATE,
                     ET_P1_CODE_SNAPSHOT_MISMATCH,
                     "p1-provider-snapshot-matches",
                     "provider snapshot is not sealed");
  }
  for (index = 0u; index < ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT; index++) {
    const et_p1_record *callback = find_record(callbacks[index]);
    if (provider_token->callbacks[index] != callback ||
        !token_integrity(callback) ||
        callback->kind != ET_P1_TOKEN_CALLBACK_IDENTITY ||
        callback->owner != context || callback->binding != provider_token) {
      return set_error(context, ET_P1_STATUS_INVALID_STATE,
                       ET_P1_CODE_SNAPSHOT_MISMATCH,
                       "p1-provider-snapshot-matches",
                       "provider callback identity snapshot differs");
    }
  }
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_state_bind_v1(
    void *candidate, void *state, const void *provider) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *state_token;
  et_p1_record *provider_token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  state_token = require_token(context, state, ET_P1_TOKEN_STATE_DICT,
                              "p1-state-bind");
  if (state_token == NULL) {
    return context->error.category;
  }
  provider_token = require_token(context, provider, ET_P1_TOKEN_PROVIDER,
                                 "p1-state-bind");
  if (provider_token == NULL) {
    return context->error.category;
  }
  if (provider_token->sealed == 0u) {
    return set_error(context, ET_P1_STATUS_UNSUPPORTED,
                     ET_P1_CODE_SNAPSHOT_MISMATCH, "p1-state-bind",
                     "provider identity is not admitted and sealed");
  }
  if (state_token->binding == NULL) {
    state_token->binding = provider_token;
    return ET_P1_STATUS_OK;
  }
  if (state_token->binding == provider_token) {
    return ET_P1_STATUS_OK;
  }
  return set_error(context, ET_P1_STATUS_INVALID_STATE,
                   ET_P1_CODE_BINDING_CONFLICT, "p1-state-bind",
                   "state is already bound to a different provider identity");
}

ET_P1_PRIVATE int64_t et_p1_private_state_provider_v1(
    void *candidate, const void *state) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *state_token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  state_token = require_token(context, state, ET_P1_TOKEN_STATE_DICT,
                              "p1-state-provider");
  if (state_token == NULL) {
    return context->error.category;
  }
  if (state_token->binding == NULL) {
    return set_error(context, ET_P1_STATUS_UNSUPPORTED,
                     ET_P1_CODE_UNBOUND_STATE, "p1-state-provider",
                     "state has no admitted provider binding");
  }
  context->result_ptr = state_token->binding->token;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_state_unbind_v1(void *candidate,
                                                     void *state) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *state_token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  state_token = require_token(context, state, ET_P1_TOKEN_STATE_DICT,
                              "p1-state-unbind");
  if (state_token == NULL) {
    return context->error.category;
  }
  state_token->binding = NULL;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_state_revoke_v1(void *candidate,
                                                     void *state) {
  et_p1_context *context = require_context(candidate);
  et_p1_record *state_token;
  if (context == NULL) {
    return ET_P1_STATUS_INVALID_ARGUMENT;
  }
  state_token = require_token(context, state, ET_P1_TOKEN_STATE_DICT,
                              "p1-state-revoke");
  if (state_token == NULL) {
    return context->error.category;
  }
  state_token->binding = NULL;
  state_token->live = 0u;
  return ET_P1_STATUS_OK;
}

static int64_t count_entries(et_p1_context *context, int live) {
  const et_p1_record *token = records;
  int64_t count = 0;
  while (token != NULL) {
    if (token->owner == context && ((token->live != 0u) == (live != 0))) {
      count++;
    }
    token = token->next;
  }
  context->result_i64 = count;
  return ET_P1_STATUS_OK;
}

ET_P1_PRIVATE int64_t et_p1_private_live_entry_count_v1(void *candidate) {
  et_p1_context *context = require_context(candidate);
  return context == NULL ? ET_P1_STATUS_INVALID_ARGUMENT
                         : count_entries(context, 1);
}

ET_P1_PRIVATE int64_t et_p1_private_tombstone_count_v1(void *candidate) {
  et_p1_context *context = require_context(candidate);
  return context == NULL ? ET_P1_STATUS_INVALID_ARGUMENT
                         : count_entries(context, 0);
}
#endif
