#ifndef ESHKOL_TRANSFORMER_C2_CHECKPOINT_INSPECT_BRIDGE_H
#define ESHKOL_TRANSFORMER_C2_CHECKPOINT_INSPECT_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ET_C2_INSPECT_RESULT_BYTES INT64_C(320)
#define ET_C2_INSPECT_CODE_PROJECTION 100u
#define ET_C2_INSPECT_CODE_RELEASE 101u

/* Private pinned-Eshkol bytevector bridge. */
int64_t et_c2_private_checkpoint_inspect_bridge_v1(
    void *path_bytevector, int64_t maximum_file_bytes,
    int64_t maximum_metadata_bytes, int64_t maximum_tensor_bytes,
    int64_t maximum_tensors, int64_t enforce_operational_profile,
    void *result_bytevector);

#ifdef ET_C2_CHECKPOINT_INSPECT_TESTING
void et_c2_checkpoint_inspect_test_reset_v1(void);
void et_c2_checkpoint_inspect_test_fail_projection_v1(int64_t enabled);
void et_c2_checkpoint_inspect_test_fail_release_v1(int64_t enabled);
#endif

#ifdef __cplusplus
}
#endif

#endif
