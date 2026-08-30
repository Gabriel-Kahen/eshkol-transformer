#include "p1_identity_internal.h"

#include <stdint.h>

int p1_public_identity_probe(const void *token, int64_t expected_kind) {
  return et_p1_public_token_live_v1(token) == INT64_C(1) &&
         et_p1_public_token_kind_v1(token) == expected_kind;
}
