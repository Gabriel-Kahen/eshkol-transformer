#!/usr/bin/env python3
"""Check the private P1 token index stays advisory and allocation-free."""

from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / "native/p1_identity.c").read_text()
header = (root / "native/p1_identity_internal.h").read_text()


def require(condition, message):
    if not condition:
        raise SystemExit(f"P1 NATIVE INDEX STRUCTURE FAIL: {message}")


def between(start, end):
    require(source.count(start) == 1, f"missing or repeated start: {start}")
    return source.split(start, 1)[1].split(end, 1)[0]


helpers = between("static uint8_t record_index_height(",
                  "static et_p1_record *find_record(")
lookup = between("static et_p1_record *find_record(",
                 "static int token_nonce_exists(")
creation = between("static int64_t create_token(",
                   "ET_P1_PUBLIC int64_t et_p1_public_identity_abi_major_v1(")

for forbidden in ("calloc(", "malloc(", "realloc(", "free("):
    require(forbidden not in helpers,
            f"index mutation allocates or frees: {forbidden}")
require("uintptr_t key = (uintptr_t)record->token" in helpers and
        "record_index_rotate_left" in helpers and
        "record_index_rotate_right" in helpers,
        "exact address-order AVL insertion changed")
require("uintptr_t key = (uintptr_t)candidate" in lookup and
        "(const void *)indexed->token == candidate" in lookup and
        "cursor = records" in lookup and
        "(const void *)cursor->token == candidate" in lookup,
        "exact indexed lookup or authoritative full-list fallback changed")
require("candidate->" not in lookup,
        "lookup dereferences a caller token")
ordered = ("record->next = records;", "records = record;",
           "record_index_insert(record_index_root, record,",
           "record_index_valid = 0u;", "context->result_ptr = token;")
positions = [creation.index(item) for item in ordered]
require(positions == sorted(positions),
        "index publication precedes authoritative record or result order changed")
require("_Static_assert(sizeof(et_p1_token) == 264u" in source and
        "_Static_assert(sizeof(et_p1_record) == 280u" in source and
        "offsetof(et_p1_record, provider_id) == 128u" in source,
        "token ABI or private record layout changed unexpectedly")
require("int64_t et_p1_test_record_index_invalidate_v1(void);" in
        header.split("#if defined(ET_P1_TEST_HOOKS)", 1)[1].split("#endif", 1)[0],
        "index invalidation is not confined to test-only private API")

print("P1 NATIVE INDEX STRUCTURE PASS: authoritative list, exact pointer AVL, "
      "fallback, no index allocation, token ABI unchanged")
