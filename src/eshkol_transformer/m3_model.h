#ifndef ET_M3_MODEL_H
#define ET_M3_MODEL_H
#include <stdint.h>
#include "eshkol_transformer/i64_tensor.h"
/* Accepted M3 source-private seams. No installed ABI or generic graph resolver. */
void *et_m3_private_graph_capture_v1(void *workspace);
int64_t et_m3_private_graph_restore_v1(void *workspace, void *graph);
int64_t et_m3_private_graph_check_primals_v1(void *graph, int64_t mode);
int64_t et_m3_private_graph_release_v1(void *graph);
void *et_m3_private_model_logits_create_v1(void *graph);
int64_t et_m3_private_model_logits_copy_to_v1(void *logits, void *header, int64_t count);
int64_t et_m3_private_model_logits_release_v1(void *logits);
int64_t et_m3_private_workspace_contribute_v1(void *workspace, void *graph,
    int64_t mode, int64_t weight_bits, int64_t ordinal);
int64_t et_m3_private_workspace_reset_v1(void *workspace);
int32_t et_m3_private_i64_unborrowed_v1(const et_i64_tensor *tensor,
    et_i64_tensor_error *error);
#endif
