#include <stddef.h>

typedef void (*et_p1_guessed_function)(void);

#define ET_P1_GUESS(name) extern void name(void)
ET_P1_GUESS(et_p1_private_callback_identity_create_v1);
ET_P1_GUESS(et_p1_private_callback_identity_revoke_v1);
ET_P1_GUESS(et_p1_private_context_create_v1);
ET_P1_GUESS(et_p1_private_context_release_v1);
ET_P1_GUESS(et_p1_private_error_category_v1);
ET_P1_GUESS(et_p1_private_error_code_v1);
ET_P1_GUESS(et_p1_private_error_message_v1);
ET_P1_GUESS(et_p1_private_error_operation_v1);
ET_P1_GUESS(et_p1_private_live_entry_count_v1);
ET_P1_GUESS(et_p1_private_module_create_v1);
ET_P1_GUESS(et_p1_private_parameter_handle_create_v1);
ET_P1_GUESS(et_p1_private_parameter_tree_create_v1);
ET_P1_GUESS(et_p1_private_provider_abort_v1);
ET_P1_GUESS(et_p1_private_provider_create_v1);
ET_P1_GUESS(et_p1_private_provider_seal_v1);
ET_P1_GUESS(et_p1_private_provider_snapshot_matches_v1);
ET_P1_GUESS(et_p1_private_result_i64_v1);
ET_P1_GUESS(et_p1_private_result_ptr_v1);
ET_P1_GUESS(et_p1_private_state_bind_v1);
ET_P1_GUESS(et_p1_private_state_dict_create_v1);
ET_P1_GUESS(et_p1_private_state_entry_create_v1);
ET_P1_GUESS(et_p1_private_state_provider_v1);
ET_P1_GUESS(et_p1_private_state_revoke_v1);
ET_P1_GUESS(et_p1_private_state_unbind_v1);
ET_P1_GUESS(et_p1_private_tombstone_count_v1);
#undef ET_P1_GUESS

et_p1_guessed_function const et_p1_all_private_guesses[] = {
    et_p1_private_callback_identity_create_v1,
    et_p1_private_callback_identity_revoke_v1,
    et_p1_private_context_create_v1,
    et_p1_private_context_release_v1,
    et_p1_private_error_category_v1,
    et_p1_private_error_code_v1,
    et_p1_private_error_message_v1,
    et_p1_private_error_operation_v1,
    et_p1_private_live_entry_count_v1,
    et_p1_private_module_create_v1,
    et_p1_private_parameter_handle_create_v1,
    et_p1_private_parameter_tree_create_v1,
    et_p1_private_provider_abort_v1,
    et_p1_private_provider_create_v1,
    et_p1_private_provider_seal_v1,
    et_p1_private_provider_snapshot_matches_v1,
    et_p1_private_result_i64_v1,
    et_p1_private_result_ptr_v1,
    et_p1_private_state_bind_v1,
    et_p1_private_state_dict_create_v1,
    et_p1_private_state_entry_create_v1,
    et_p1_private_state_provider_v1,
    et_p1_private_state_revoke_v1,
    et_p1_private_state_unbind_v1,
    et_p1_private_tombstone_count_v1,
};

size_t et_p1_all_private_guess_count(void) {
  return sizeof(et_p1_all_private_guesses) /
         sizeof(et_p1_all_private_guesses[0]);
}
