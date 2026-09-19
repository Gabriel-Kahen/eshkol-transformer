#ifndef ET_M3T_TRANSPORT_H
#define ET_M3T_TRANSPORT_H
#include <stdint.h>
#include "eshkol_transformer/f32_tensor.h"
/* Private source ABI; not installed. All calls require serialization. */
enum { ET_M3T_OK=0, ET_M3T_INVALID_ARGUMENT=1, ET_M3T_INVALID_STATE=2,
       ET_M3T_SHAPE_MISMATCH=3, ET_M3T_UNSUPPORTED=4, ET_M3T_INTERNAL=5 };
enum { ET_M3T_CODE_OK=0, ET_M3T_CODE_IDENTITY=1, ET_M3T_CODE_LIFECYCLE=2,
       ET_M3T_CODE_SELECTOR=3, ET_M3T_CODE_SHAPE=4, ET_M3T_CODE_ALLOCATION=5,
       ET_M3T_CODE_STALE=6, ET_M3T_CODE_COUNTER=7, ET_M3T_CODE_TOPOLOGY=8,
       ET_M3T_CODE_REENTRANCY=9, ET_M3T_CODE_INTERNAL=10 };
void *et_m3t_private_initializer_create_v1(int64_t seed);
int64_t et_m3t_private_initializer_word_v1(void *initializer, int64_t index);
void *et_m3t_private_owner_create_v1(void *initializer);
void *et_m3t_private_owner_parameter_v1(void *owner, int64_t index);
int64_t et_m3t_private_owner_bind_v1(void *owner, int64_t index, void *handle);
int64_t et_m3t_private_owner_initialize_v1(void *owner, int64_t index);
void *et_m3t_private_owner_successor_v1(void *owner);
void *et_m3t_private_input_create_v1(void);
int64_t et_m3t_private_input_copy_v1(void *input, int64_t first, int64_t second);
int64_t et_m3t_private_input_copy_tokenizer_v1(void *input, void *tokenizer_owner);
void *et_m3t_private_logits_create_v1(void);
void *et_m3t_private_workspace_create_v1(void *owner);
int64_t et_m3t_private_workspace_begin_v1(void *workspace, void *input, int64_t mode);
int64_t et_m3t_private_workspace_check_primals_v1(void *workspace, int64_t mode);
void *et_m3t_private_workspace_gradient_v1(void *workspace, int64_t index);
int64_t et_m3t_private_sum_v1(void *workspace, int64_t role);
int64_t et_m3t_private_initializer_seal_v1(void *initializer);
int64_t et_m3t_private_initializer_abort_v1(void *initializer);
int64_t et_m3t_private_owner_seal_v1(void *owner);
int64_t et_m3t_private_owner_abort_v1(void *owner);
int64_t et_m3t_private_input_release_v1(void *input);
int64_t et_m3t_private_logits_release_v1(void *logits);
int64_t et_m3t_private_workspace_release_v1(void *workspace);
int64_t et_m3t_private_workspace_reset_v1(void *workspace);
int64_t et_m3t_private_logits_copy_from_v1(void *logits, void *bytevector_header, int64_t count);
int64_t et_m3t_private_logits_copy_to_v1(void *logits, void *bytevector_header, int64_t count);
int64_t et_m3t_private_workspace_vjp_begin_v1(void *workspace, void *logits);
int64_t et_m3t_private_workspace_copy_logits_v1(void *workspace, void *logits);
int64_t et_m3t_private_embedding_v1(void *workspace, int64_t role, int64_t reverse);
int64_t et_m3t_private_linear_v1(void *workspace, int64_t role, int64_t reverse);
int64_t et_m3t_private_layer_norm_v1(void *workspace, int64_t role, int64_t reverse);
int64_t et_m3t_private_residual_v1(void *workspace, int64_t role, int64_t reverse);
int64_t et_m3t_private_heads_split_v1(void *workspace, int64_t role, int64_t reverse);
int64_t et_m3t_private_gelu_v1(void *workspace, int64_t reverse);
int64_t et_m3t_private_heads_merge_v1(void *workspace, int64_t reverse);
int64_t et_m3t_private_attention_v1(void *workspace, int64_t reverse);
int64_t et_m3t_private_last_error_domain_v1(void);
int64_t et_m3t_private_last_error_category_v1(void);
int64_t et_m3t_private_last_error_code_v1(void);
/* Approved I2 construction observer; never installed or application-callable. */
int32_t et_m3t_construction_parameter_preflight_internal(void *parameter,
    const void *exact_handle, et_f32_tensor_error *error);
#endif
