#if defined(ET_P1_CPP_TRUSTED_PROBE)
#define ET_P1_PRIVATE_API 1
#endif
#include "p1_identity_internal.h"

#include <type_traits>

#define CHECK_SIGNATURE(function_name, ...)                                   \
  static_assert(std::is_same_v<decltype(&function_name), __VA_ARGS__>)

CHECK_SIGNATURE(et_p1_public_identity_abi_major_v1, int64_t (*)(void));
CHECK_SIGNATURE(et_p1_public_identity_abi_minor_v1, int64_t (*)(void));
CHECK_SIGNATURE(et_p1_public_token_kind_v1, int64_t (*)(const void *));
CHECK_SIGNATURE(et_p1_public_token_live_v1, int64_t (*)(const void *));

#if defined(ET_P1_CPP_TRUSTED_PROBE)
CHECK_SIGNATURE(et_p1_private_context_create_v1, void *(*)(void));
CHECK_SIGNATURE(et_p1_private_context_release_v1, int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_error_category_v1,
                int64_t (*)(const void *));
CHECK_SIGNATURE(et_p1_private_error_code_v1, int64_t (*)(const void *));
CHECK_SIGNATURE(et_p1_private_error_operation_v1,
                const char *(*)(const void *));
CHECK_SIGNATURE(et_p1_private_error_message_v1,
                const char *(*)(const void *));
CHECK_SIGNATURE(et_p1_private_result_ptr_v1, void *(*)(const void *));
CHECK_SIGNATURE(et_p1_private_result_i64_v1, int64_t (*)(const void *));

CHECK_SIGNATURE(et_p1_private_provider_create_v1,
                int64_t (*)(void *, const void *, int64_t));
CHECK_SIGNATURE(et_p1_private_provider_abort_v1,
                int64_t (*)(void *, void *));
CHECK_SIGNATURE(et_p1_private_module_create_v1, int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_parameter_handle_create_v1,
                int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_parameter_tree_create_v1,
                int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_state_dict_create_v1, int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_state_entry_create_v1, int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_callback_identity_create_v1,
                int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_callback_identity_revoke_v1,
                int64_t (*)(void *, void *));

CHECK_SIGNATURE(et_p1_private_provider_seal_v1,
                int64_t (*)(void *, void *, void *, void *, void *, void *,
                            void *, void *, void *));
CHECK_SIGNATURE(
    et_p1_private_provider_snapshot_matches_v1,
    int64_t (*)(void *, const void *, const void *, const void *, const void *,
                const void *, const void *, const void *, const void *));
CHECK_SIGNATURE(et_p1_private_state_bind_v1,
                int64_t (*)(void *, void *, const void *));
CHECK_SIGNATURE(et_p1_private_state_provider_v1,
                int64_t (*)(void *, const void *));
CHECK_SIGNATURE(et_p1_private_state_unbind_v1,
                int64_t (*)(void *, void *));
CHECK_SIGNATURE(et_p1_private_state_revoke_v1,
                int64_t (*)(void *, void *));
CHECK_SIGNATURE(et_p1_private_live_entry_count_v1, int64_t (*)(void *));
CHECK_SIGNATURE(et_p1_private_tombstone_count_v1, int64_t (*)(void *));
#endif

#undef CHECK_SIGNATURE

int main() {
  if (et_p1_public_identity_abi_major_v1() !=
          static_cast<int64_t>(ET_P1_IDENTITY_ABI_MAJOR) ||
      et_p1_public_identity_abi_minor_v1() !=
          static_cast<int64_t>(ET_P1_IDENTITY_ABI_MINOR) ||
      et_p1_public_token_kind_v1(nullptr) != ET_P1_TOKEN_FOREIGN ||
      et_p1_public_token_live_v1(nullptr) != 0) {
    return 1;
  }
#if defined(ET_P1_CPP_TRUSTED_PROBE)
  void *context = et_p1_private_context_create_v1();
  if (context == nullptr ||
      et_p1_private_context_release_v1(context) != ET_P1_STATUS_OK) {
    return 2;
  }
#endif
  return 0;
}
