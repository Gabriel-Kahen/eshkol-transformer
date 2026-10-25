# P2/G2 original bytevector allocation extent witness

Status: bounded test-only source candidate. The focused compiled fixture is
**NOT RUN** while the shared heavy gate is occupied. This leaf cannot commit a
terminal output and does not admit public G2 or checkpoint continuation.

The exact pinned Eshkol `make-bytevector` lowering requests `8+len` bytes for
the in-payload length header and byte data
(`lib/backend/llvm_codegen.cpp:16286-16318`).
`arena_allocate_with_header` prepends a separate eight-byte object header,
rounds the total to eight, records the requested payload size and returns the
payload pointer (`lib/core/runtime_object_alloc.cpp:20-61`). An extern `ptr`
argument receives that tagged payload address unchanged
(`lib/backend/llvm_codegen.cpp:18772-18792`). For original lengths 1, 2, 8,
16, the payload extents are respectively 9, 10, 16, 24 bytes and the rounded
backing allocations are 24, 24, 24, 32 bytes.

`tests/g3c4/call_entry_allocation_shim.cpp` now has a test-only ledger. While
armed around the exact protected terminal preparation, linker wrappers retain
the original eight bytevector allocation results, requested payload sizes,
actual aligned allocation sizes, raw addresses and returned payload addresses.
The ledger is native static storage: the Eshkol working record and anchor
cannot rewrite it. `tests/g3c4/p2_g2_terminal_extent_witness_test.esk`
checks each of the eight original carriers through the compiled extern `ptr`
path. It also checks wrong declared sizes, swapped originals and a paired
equal-length substitution of both mutable terminal records. The latter may
pass the existing record comparison, but must fail the original-allocation
ledger. No ledger predicate reads a header or data from a supplied pointer.

Run the focused normal, repeat and ASan+UBSan fixture with
`scripts/test-g3c4-p2g2-terminal-extent-witness.sh` when `heavy.lock` is free.
The test asserts byte-identical normal/repeat/sanitizer output. A passing run
would validate this exact compiler/runtime/fixture construction path. It would
not make the test-only ledger a production authentication API. Production
terminal close still needs an inaccessible creation-time original binding,
first/second T1 and EOS integrity, post-native-zero fail-stop tail, detached
snapshot/clone, and E3's trained-G2 sanitizer classification. Direct compiled
`make-bytevector` also stores through an unchecked allocator return; classify
allocation-failure behavior separately before claiming recoverable cuts.
