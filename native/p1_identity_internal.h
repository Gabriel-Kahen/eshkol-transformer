#ifndef ESHKOL_TRANSFORMER_P1_IDENTITY_INTERNAL_H
#define ESHKOL_TRANSFORMER_P1_IDENTITY_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_P1_IDENTITY_ABI_MAJOR UINT32_C(1)
#define ET_P1_IDENTITY_ABI_MINOR UINT32_C(0)
#define ET_P1_IDENTITY_ERROR_OPERATION_CAPACITY 64u
#define ET_P1_IDENTITY_ERROR_MESSAGE_CAPACITY 192u
#define ET_P1_IDENTITY_PROVIDER_CALLBACK_COUNT 7u
#define ET_P1_IDENTITY_MAX_PROVIDER_ID_BYTES 127u
#define ET_P1_IDENTITY_MAX_LIVE_TOKENS 65536u

enum {
  ET_P1_TOKEN_FOREIGN = 0,
  ET_P1_TOKEN_PROVIDER = 1,
  ET_P1_TOKEN_MODULE = 2,
  ET_P1_TOKEN_PARAMETER_HANDLE = 3,
  ET_P1_TOKEN_PARAMETER_TREE = 4,
  ET_P1_TOKEN_STATE_DICT = 5,
  ET_P1_TOKEN_STATE_ENTRY = 6,
  ET_P1_TOKEN_CALLBACK_IDENTITY = 7
};

enum {
  ET_P1_STATUS_OK = 0,
  ET_P1_STATUS_INVALID_ARGUMENT = 1,
  ET_P1_STATUS_UNSUPPORTED = 2,
  ET_P1_STATUS_INVALID_STATE = 3,
  ET_P1_STATUS_INTERNAL = 4
};

enum {
  ET_P1_CODE_NONE = 0,
  ET_P1_CODE_NULL_ARGUMENT = 1,
  ET_P1_CODE_FOREIGN_CONTEXT = 2,
  ET_P1_CODE_FOREIGN_TOKEN = 3,
  ET_P1_CODE_STALE_TOKEN = 4,
  ET_P1_CODE_WRONG_TOKEN_KIND = 5,
  ET_P1_CODE_CROSS_CONTEXT = 6,
  ET_P1_CODE_ALLOCATION_FAILED = 7,
  ET_P1_CODE_ALREADY_SEALED = 8,
  ET_P1_CODE_SNAPSHOT_MISMATCH = 9,
  ET_P1_CODE_BINDING_CONFLICT = 10,
  ET_P1_CODE_UNBOUND_STATE = 11,
  ET_P1_CODE_LIVE_ENTRIES = 12,
  ET_P1_CODE_ENTROPY_UNAVAILABLE = 13,
  ET_P1_CODE_CAPACITY_EXCEEDED = 14,
  ET_P1_CODE_FORKED_PROCESS = 15,
  ET_P1_CODE_TOKEN_INTEGRITY = 16,
  ET_P1_CODE_INVALID_TEXT = 17
};

typedef struct et_p1_identity_error_v1 {
  int64_t category;
  int64_t code;
  char operation[ET_P1_IDENTITY_ERROR_OPERATION_CAPACITY];
  char message[ET_P1_IDENTITY_ERROR_MESSAGE_CAPACITY];
} et_p1_identity_error_v1;

#if defined(__cplusplus)
#define ET_P1_STATIC_ASSERT(condition, message) static_assert(condition, message)
#else
#define ET_P1_STATIC_ASSERT(condition, message) _Static_assert(condition, message)
#endif
ET_P1_STATIC_ASSERT(sizeof(void *) == 8u && sizeof(size_t) == 8u,
                    "P1 identity ABI v1 requires 64-bit pointers and size_t");
ET_P1_STATIC_ASSERT(sizeof(et_p1_identity_error_v1) == 272u,
                    "P1 identity error layout changed");
ET_P1_STATIC_ASSERT(offsetof(et_p1_identity_error_v1, operation) == 16u &&
                        offsetof(et_p1_identity_error_v1, message) == 80u,
                    "P1 identity error offsets changed");
#undef ET_P1_STATIC_ASSERT

int64_t et_p1_public_identity_abi_major_v1(void);
int64_t et_p1_public_identity_abi_minor_v1(void);
int64_t et_p1_public_token_kind_v1(const void *token);
int64_t et_p1_public_token_live_v1(const void *token);

#if defined(ET_P1_PRIVATE_API)
void *et_p1_private_context_create_v1(void);
int64_t et_p1_private_context_release_v1(void *context);
int64_t et_p1_private_error_category_v1(const void *context);
int64_t et_p1_private_error_code_v1(const void *context);
const char *et_p1_private_error_operation_v1(const void *context);
const char *et_p1_private_error_message_v1(const void *context);
void *et_p1_private_result_ptr_v1(const void *context);
int64_t et_p1_private_result_i64_v1(const void *context);

int64_t et_p1_private_provider_create_v1(void *context,
                                         const void *provider_id,
                                         int64_t provider_id_bytes);
int64_t et_p1_private_provider_abort_v1(void *context, void *provider);
int64_t et_p1_private_module_create_v1(void *context);
int64_t et_p1_private_parameter_handle_create_v1(void *context);
int64_t et_p1_private_parameter_tree_create_v1(void *context);
int64_t et_p1_private_state_dict_create_v1(void *context);
int64_t et_p1_private_state_entry_create_v1(void *context);
int64_t et_p1_private_callback_identity_create_v1(void *context);
int64_t et_p1_private_callback_identity_revoke_v1(void *context,
                                                   void *identity);

int64_t et_p1_private_provider_seal_v1(
    void *context, void *provider, void *describe, void *clone,
    void *storage_identical, void *value_equal, void *device_equal,
    void *prepare, void *commit);
int64_t et_p1_private_provider_snapshot_matches_v1(
    void *context, const void *provider, const void *describe,
    const void *clone, const void *storage_identical, const void *value_equal,
    const void *device_equal, const void *prepare, const void *commit);

int64_t et_p1_private_state_bind_v1(void *context, void *state,
                                    const void *provider);
int64_t et_p1_private_state_provider_v1(void *context, const void *state);
int64_t et_p1_private_state_unbind_v1(void *context, void *state);
int64_t et_p1_private_state_revoke_v1(void *context, void *state);
int64_t et_p1_private_live_entry_count_v1(void *context);
int64_t et_p1_private_tombstone_count_v1(void *context);
#endif

#ifdef __cplusplus
}
#endif

#endif
