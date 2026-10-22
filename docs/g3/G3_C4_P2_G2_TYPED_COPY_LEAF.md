# Private C4 P2/G2 typed copy accessor

Status: source-only native successor to the reviewed typed result head
`2d607e2a526d784e0674f7456e4ba0b8cc9bdbf3`. No Eshkol wrapper or public
G2 admission is implied.

The existing `typed_element_v1` writes through `int64_t *`. An Eshkol
bytevector passed as `ptr` addresses its eight-byte length header, so it
cannot safely serve as that element destination. The feature-private
`et_g3c4_private_p2g2_typed_copy_v1(owner, expected_length,
destination_header, destination_bytes)` copies the entire typed tensor into
the payload of a bytevector whose header stays unchanged. There is no raw
I1 pointer or new public dispatch.

The owner must be a registered live typed kind 10/11/12. `expected_length`
must equal the native rank-one extent: 1 or 2 for IDs, 1 for generated/cache
length. The destination is aligned, has a readable complete extent of
`8 + 8 * expected_length` bytes (16 or 24), declares exactly
`8 * expected_length` payload bytes in its signed-i64 header, and does not
overlap any C4 control or owned tensor/model/cache storage. The Eshkol caller
must prove the actual original bytevector allocation and identity; a raw C
pointer cannot establish that fact. Source admission verifies dense CPU-i64
shape, values and balanced borrow. The implementation reauthenticates owner,
source values, destination alias and header before one payload write. A
rejected call leaves header and payload byte-identical. Values are encoded as
little-endian i64 words. A held I1 lease rejects and allows retry.

The native host test covers EOS `[1]`, two-ID `[2]`, both scalars, detached
source lifetime, wrong/dead owner, wrong length, null and malformed
destination, extent/overflow, control alias, held lease, malformed source
view, exact bytes and unchanged header. Feature-off compilation and symbol
delta are checked by `scripts/test-g3c4-p2g2-typed-result.sh`. The supported
one-archive and same-registry Eshkol gates remain separate acceptance steps.
