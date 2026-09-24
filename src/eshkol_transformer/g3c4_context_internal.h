#ifndef ET_G3C4_CONTEXT_INTERNAL_H
#define ET_G3C4_CONTEXT_INTERNAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Source-private ABI. All context calls require external serialization. */
void *et_g3c4_private_context_create_v1(void *c4_owner);
int64_t et_g3c4_private_context_close_v1(void *context);

#ifdef __cplusplus
}
#endif

#endif
