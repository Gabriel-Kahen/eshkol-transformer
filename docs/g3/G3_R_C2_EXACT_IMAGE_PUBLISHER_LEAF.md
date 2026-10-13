# G3-R private exact-C2-image pair publisher candidate

This feature-local source leaf uses C2's existing synchronous
`c2-save-with-owner-borrow` callback. The callback receives the one fully
measured, encoded and C2-validated image in its parent scratch region, after
the C2 owner borrow ends. The publisher hashes **those exact bytes** with
ordinary SHA-256, encodes and decodes a 288-byte G3RCV1 companion against
that digest, then uses the C1 no-overwrite atomic writer for the C2 image
first and companion second. The regional image never escapes the callback.
No ordinary SAVE/path reread supplies the digest.

`g3r-c2-exact-image-pair-publish-internal!(state, c2-path,
companion-path, policy, fields)` accepts an existing detached C2 owner and
seven detached G3 fields: policy[6], RNG[4], ordered raw history, prompt,
generated and manual counts, and raw V256 T1 fingerprint[96]. It validates
and copies those fields and both distinct paths before C2 borrowing or file
I/O. Both path snapshots are made before path validation or existence checks;
the callback captures only those snapshots, so mutation of either original
string cannot redirect its write. An existing path rejects early; the native
no-overwrite writer remains
the final authority if the directory changes. Encoding, record round trip,
or C2 measurement failure leaves both target paths untouched. A first-write
failure never attempts the companion. A companion failure can leave a complete
unreachable C2 image. A post-rename sync/close failure can leave a complete
file visible despite a raised error. The function returns `#t` only after
both writes report success; it does not delete possible orphans.

The caller's seven G3 fields are **not same-owner save authority**. This leaf
does not prove that they came from the C2 model or authenticate the C2
tokenizer field against the supplied fingerprint. It therefore cannot be
used as a user-visible SAVE or claimed as G3-R continuation. The precise
next integration edge is the candidate
`g3r-save-c2-capture-internal(generator, model, resolved, tokenizer,
dataset, optimizer, capsule)` on divergent `codex/g3r-save-fixed-caller-20261011`:
it returns a detached C2 owner and native-owned 18-word G3 snapshot after a
joint original-owner hold. A reviewed adapter must map that authenticated
image to the seven codec fields, attest the raw T1 fingerprint against the
captured C2 controls, and own/release the C2 state across this publisher's
success and every failure. That capture's compiled post-adoption and
partial-claim rollback gates are still pending; it has not been composed
with this leaf or with PR #203.

The focused source test checks exact-image hash, encode/decode-before-write,
no-overwrite order, both failure stops, preflight order and caller-field
copying. A development-only Scheme harness executes the actual publisher
source with controlled C2/record/I/O callbacks: it mutates each original path
after preflight and proves both writes retain the copied names, then checks
encode/decode and first/second-write failure cuts. Its two path-copy mutants
causally fail. This is source execution with stubs, not compiled Eshkol or
native filesystem evidence;
compiled C2/G3 aggregate, filesystem failpoints, repeat/sanitizer runs,
same-owner capture and fresh-process LOAD remain required acceptance gates.
