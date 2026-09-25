#define ET_P1_PRIVATE_API 1
#define ET_P1_TEST_HOOKS 1
#include "p1_identity_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INDEX_CASES 512u

static int checks;

static void check(int condition, const char *label) {
  if (!condition) {
    (void)fprintf(stderr, "P1 native index FAIL: %s\n", label);
    exit(1);
  }
  checks++;
}

int main(void) {
  void *context = et_p1_private_context_create_v1();
  void *identities[INDEX_CASES];
  unsigned int index;
  int forged = 0;
  void *later;

  check(context != NULL, "private context starts");
  for (index = 0u; index < INDEX_CASES; index++) {
    check(et_p1_private_callback_identity_create_v1(context) ==
              ET_P1_STATUS_OK,
          "token creation publishes a record");
    identities[index] = et_p1_private_result_ptr_v1(context);
    check(et_p1_public_token_kind_v1(identities[index]) ==
              ET_P1_TOKEN_CALLBACK_IDENTITY,
          "indexed exact token has its kind");
  }
  for (index = 0u; index < INDEX_CASES; index += 2u) {
    check(et_p1_private_callback_identity_revoke_v1(context,
                                                     identities[index]) ==
              ET_P1_STATUS_OK,
          "revoke preserves terminal record");
    check(et_p1_public_token_kind_v1(identities[index]) ==
              -ET_P1_TOKEN_CALLBACK_IDENTITY,
          "indexed tombstone keeps its exact kind");
    check(et_p1_public_token_live_v1(identities[index]) == 0,
          "indexed tombstone stays stale");
  }
  check(et_p1_public_token_kind_v1(&forged) == ET_P1_TOKEN_FOREIGN,
        "foreign pointer is never dereferenced");
  check(et_p1_public_token_live_v1(&forged) == 0,
        "foreign pointer is not live");
  check(et_p1_private_callback_identity_revoke_v1(context, &forged) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "foreign private token is rejected");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_FOREIGN_TOKEN,
        "foreign private token keeps its category");
  check(et_p1_private_state_release_begin_v1(context, identities[1]) ==
            ET_P1_STATUS_INVALID_ARGUMENT,
        "wrong-kind indexed identity is rejected");
  check(et_p1_private_error_code_v1(context) == ET_P1_CODE_WRONG_TOKEN_KIND,
        "wrong-kind category is unchanged");

  check(et_p1_test_record_index_invalidate_v1() == ET_P1_STATUS_OK,
        "test invalidates advisory index");
  for (index = 0u; index < INDEX_CASES; index++) {
    int64_t expected_kind = (index % 2u == 0u)
                                ? -ET_P1_TOKEN_CALLBACK_IDENTITY
                                : ET_P1_TOKEN_CALLBACK_IDENTITY;
    check(et_p1_public_token_kind_v1(identities[index]) == expected_kind,
          "invalid index falls back to exact terminal registry identity");
    check(et_p1_public_token_live_v1(identities[index]) ==
              (index % 2u == 0u ? 0 : 1),
          "invalid index retains exact liveness");
  }
  check(et_p1_private_callback_identity_create_v1(context) ==
            ET_P1_STATUS_OK,
        "publication while index invalid still succeeds");
  later = et_p1_private_result_ptr_v1(context);
  check(et_p1_public_token_kind_v1(later) ==
            ET_P1_TOKEN_CALLBACK_IDENTITY,
        "new identity remains visible through registry fallback");
  check(et_p1_private_callback_identity_revoke_v1(context, later) ==
            ET_P1_STATUS_OK,
        "new identity remains revocable through fallback");
  check(et_p1_public_token_live_v1(later) == 0,
        "new tombstone remains visible through fallback");

  (void)printf("P1 native index PASS: %d checks\n", checks);
  return 0;
}
