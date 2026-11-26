# G3-R private companion file-read prerequisite

This source-only leaf composes the accepted in-memory C2/SHA pair decoder with
the existing C1 exact-file reader. It does not create a checkpoint or companion,
load a model, construct a generator, or admit public continuation.

The private `g3r-c2-pair-file-decode-internal!(receipt, path, destination)`
takes an existing live same-image C2/SHA receipt, a bounded nonempty path
without NUL, and the eight-slot destination already required by
`g3r-c2-pair-decode-internal!`. It allocates a fresh 288-byte staging
bytevector. The inherited `et_checkpoint_io_read_exact_v1` binding is called
with expected and maximum length both 288; that C1 entry rejects a symlink or
nonregular final component, short/long file, read failure and close failure
before copying bytes into staging. A nonzero C1 status raises the existing
private C1 `io` error with `checkpoint-load` operation. Only a successful
read enters the accepted pair decoder, which authenticates the receipt's
authoritative C2-image SHA and atomically copies eight detached fields after
closing its borrow. A failed read never borrows the receipt; any read or decode
failure leaves the caller destination and receipt authority unchanged.

`native/g3r_c2_pair_file_read_root.esk` loads the accepted SHA-receipt root
before the new extension. The separate source-closure manifest retains that
entire predecessor and adds no installed symbol. The deterministic Python
fixture writer generates canonical, wrong-image, torn, short, long, symlink
and FIFO files in a fresh directory. A prepared same-registry Eshkol fixture
also injects C1 read/source-close/directory-close cuts and tests forged, busy
and dead receipts. Host-only source/mutation and fixture-layout tests run now;
the genuine compiled Eshkol normal/repeat/sanitizer file/receipt test remains
a separate acceptance gate. C1's descriptor checks detect size/identity
changes, but they do not establish immutability against a concurrent
same-size in-place writer; the continuation contract requires immutable
published pair files.

This leaf reads a companion against an already loaded C2 receipt only. It
does not resolve the current-history same-owner SAVE transaction, atomic
two-file publication, restored model/tokenizer provenance, no-draw cache
replay, or resumed generation.
