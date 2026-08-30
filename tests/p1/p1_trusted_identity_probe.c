#define ET_P1_PRIVATE_API 1
#include "p1_identity_internal.h"

void *p1_trusted_module_identity(void *context) {
  if (et_p1_private_module_create_v1(context) != ET_P1_STATUS_OK) {
    return NULL;
  }
  return et_p1_private_result_ptr_v1(context);
}
