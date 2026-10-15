# Private C4 P2/G2 Eshkol typed result wrapper

Status: source-only candidate on the reviewed private native typed-copy
successor `43756db161b39be8ebc2f664946435397efe169f`. This is no public G2
route, package export or CLI admission.

The same-registry `g3c4-p2g2-typed-create-private` consumes an authenticated
live kind-9 logical ID clone shell and one of `typed-ids`, `typed-lengths`, or
`typed-cache-lengths`. Each call creates an independent kind-10/11/12 native
CPU-i64 rank-one owner and returns `(list shell)`. IDs have shape `[1]` or
`[2]` and exact logical values; scalar lengths have shape `[1]` and values
`G=1/2`, `P+G=3/4`. The wrapper checks native length and bytewise copied
little-endian values before publishing. The native typed read checks dense
CPU-i64 rank, shape, layout, stride and value range.

The Eshkol wrapper allocates `8*count` payload bytes itself, captures that
original bytevector and its length in a rooted closure, and verifies its
identity and length before and after the header-aware native copy. It passes
the complete `8+8*count` extent, so the native entry writes only the payload
and leaves the length header intact. `g3c4-p2g2-typed-copy-private` allocates
a fresh carrier and repeats native copy/readback checks. Once published, the
typed result no longer depends on the clone or terminal output being live.

On failed creation, the exact created native owner is released before the
pending Eshkol entry is tombstoned; the original registry and anchor heads
are restored only when both still point at this attempt's nodes. An unexpected
head is fail-stop. The first Eshkol error is rethrown. A
native held-I1 lease rejects release without Eshkol mutation, allowing retry.
On successful native release, the wrapper scrubs its retained copy and
tombstones the entry through a fail-stop tail. Exact repeated release requires
the original rooted closure's recorded native success. Forged dead state,
changed shell/entry/carrier/native identity, wrong kind and stale ownership
cannot claim cleanup.

The source gate checks closure/load boundaries, balanced syntax and mutations
that remove allocation, extent, identity, checked native copy and cleanup
steps. A host Guile harness loads the actual wrapper with controlled native
stubs: 64 failures each at creation, length, and copy leave the original
registry/anchor roots and lookup unchanged (969 checks). It detects removal
of registry rollback or either native-copy status check. These stubs do not
prove the C FFI or Eshkol compiler behavior. A compiled same-registry Eshkol gate, release fault injection,
normal/repeat/sanitizers and the supported one-archive gate remain required
before this candidate is admitted. The native accessor cannot prove actual
Eshkol allocation; this wrapper's allocation-time root is that proof.
