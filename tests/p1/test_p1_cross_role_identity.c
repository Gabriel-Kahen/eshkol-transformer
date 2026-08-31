#define ET_P1_PRIVATE_API 1
#include "p1_identity_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int p1_public_identity_probe(const void *token, int64_t expected_kind);
void *p1_trusted_module_identity(void *context);

static int checks;

static void check(int condition, const char *label) {
  if (!condition) {
    (void)fprintf(stderr, "P1 cross-role FAIL: %s\n", label);
    exit(1);
  }
  checks++;
}

int main(void) {
  void *context = et_p1_private_context_create_v1();
  void *first;
  void *second;
  unsigned char copy[264];
  int foreign = 0;

  check(context != NULL, "trusted role obtains its private capability");
  first = p1_trusted_module_identity(context);
  second = p1_trusted_module_identity(context);
  check(first != NULL && second != NULL, "trusted role mints identities");
  check(first != second, "separate trusted identities stay distinct");
  check(p1_public_identity_probe(first, ET_P1_TOKEN_MODULE),
        "public-only translation unit recognizes trusted identity");
  check(p1_public_identity_probe(second, ET_P1_TOKEN_MODULE),
        "public observation is deterministic across identities");
  check(!p1_public_identity_probe(first, ET_P1_TOKEN_STATE_DICT),
        "public observation enforces exact kind");
  check(!p1_public_identity_probe(&foreign, ET_P1_TOKEN_MODULE),
        "public observation rejects arbitrary pointers");
  memcpy(copy, first, sizeof(copy));
  check(!p1_public_identity_probe(copy, ET_P1_TOKEN_MODULE),
        "public observation rejects copied token bytes");

  (void)printf("P1 cross-role PASS: %d checks\n", checks);
  return 0;
}
