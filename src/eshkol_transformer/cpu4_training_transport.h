#ifndef ET_CPU4_TRAINING_TRANSPORT_H
#define ET_CPU4_TRAINING_TRANSPORT_H
#include <stdint.h>
#ifdef ET_CPU4_TRAINING_PRIVATE

void *et_cpu4_private_input_create_v1(void *lease, int64_t a, int64_t b, int64_t c, int64_t d);
int64_t et_cpu4_private_input_release_v1(void *input);
void *et_cpu4_private_logits_create_v1(void *lease);
int64_t et_cpu4_private_logits_copy_from_v1(void *logits, void *bytevector_header, int64_t byte_count);
int64_t et_cpu4_private_logits_copy_to_v1(void *logits, void *bytevector_header, int64_t byte_count);
int64_t et_cpu4_private_logits_release_v1(void *logits);
void *et_cpu4_private_workspace_create_v1(void *lease, void *input, int64_t mode);
int64_t et_cpu4_private_workspace_role_v1(void *workspace, int64_t family, int64_t index, int64_t reverse);
int64_t et_cpu4_private_workspace_copy_logits_v1(void *workspace, void *logits);
int64_t et_cpu4_private_workspace_vjp_begin_v1(void *workspace, void *upstream);
int64_t et_cpu4_private_workspace_check_primals_v1(void *workspace, int64_t mode);
int64_t et_cpu4_private_workspace_contribute_v1(void *workspace, int64_t mode, int64_t weight_bits, int64_t ordinal);
int64_t et_cpu4_private_workspace_release_v1(void *workspace);
int64_t et_cpu4_private_last_error_domain_v1(void);
int64_t et_cpu4_private_last_error_category_v1(void);
int64_t et_cpu4_private_last_error_code_v1(void);

#ifdef ET_CPU4_OBJECTIVE_PRIVATE
struct et_kernel_tensor_view_v1;
int64_t et_cpu4_private_batch_copy_v1(void *lease, const struct et_kernel_tensor_view_v1 *view,
                                     void *bytes, int64_t kind);
int64_t et_cpu4_private_objective_prepare_v1(void *logits, void *seed, void *targets,
                                            void *mask, void *observation);
#endif
#endif
#endif
