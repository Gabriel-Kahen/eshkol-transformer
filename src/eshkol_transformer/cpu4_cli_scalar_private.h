#ifndef ET_CPU4_CLI_SCALAR_PRIVATE_H
#define ET_CPU4_CLI_SCALAR_PRIVATE_H
#include "eshkol_transformer/f32_tensor.h"
/* Source-private serialized CLI custody. Never an installed facade export. */
et_f32_tensor *et_cpu4_private_cli_scalar_create_v1(void);
int64_t et_cpu4_private_cli_scalar_read_bits_v1(const et_f32_tensor *);
int64_t et_cpu4_private_cli_scalar_claim_v1(et_f32_tensor *, int64_t);
int64_t et_cpu4_private_cli_scalar_release_v1(et_f32_tensor *);
int64_t et_cpu4_private_cli_scalar_last_error_ref_v1(int64_t);
#endif
