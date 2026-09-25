# TR3-C private C trainer entry: contract blocker

Status: source-backed C-entry blocker. A separate private lease-unenroll
operation now ends registry retention; no C constructor, C handle or public
trainer facade is accepted by this note.

The accepted 46-source root in `native/tr3_c_private_package_root.esk`
contains `tr3-lease-create-internal` with five operands: resolved configuration,
tokenizer, dataset, model and optimizer. The fixed compiler emits both its
internal Eshkol implementation and a five-tagged-value C ABI thunk. Successful
source calls return a fresh `#(trainer)` shell and enroll a 22-slot record in
`tr3-trainers`; categorized failures raise under operation `trainer-create`.
The acquisition handler clears the staging root and guard on failure. It
does not take ownership of the five source operands on failure. The sole
successful head store is at `native/tr3_lease_core_extension.esk:479`.

The linked package deliberately exports only
`et_tr3_c_private_initialize_v1` and its version marker. Its initializer
owns the same-runtime arena setup and E1B-style exception scope, but it does
not produce any of the five constructor operands. The only complete producer
sequence is the source test fixture in
`tests/tr3_snapshot/runtime_smoke.esk:69-110`: X1 resolve, T2 byte tokenizer,
D2 open, M3T model creation, P1 handles, and O2 optimizer creation. Exporting
the constructor thunk alone would permit a C call only with five already
valid tagged values from **this same compiled identity universe**. It would
not provide a constructible package API. Raw addresses or objects made by a
separately compiled Eshkol package are not substitutes for those identities.

`tr3-trainers` is a rooted registry in
`native/tr3_lease_core_extension.esk:9`; acquisition prepends its record to
slot 0. It retains the shell, all five receivers and captured handles after
the caller drops its shell. The public ownership rule says the exclusive
lease lasts until the trainer is unreachable. The
[private lease-unenroll seam](TR3_C_PRIVATE_LEASE_UNENROLL.md) now removes an
authenticated idle record so the registry no longer retains it; release is
explicit and does not infer collector reachability. D2 has an idempotent
dataset close; `docs/PUBLIC_API_CONTRACT.md` explicitly says O2 v1 has no
optimizer-destroy operation and keeps live optimizer moments process-local
until exit. The accepted M3T model path likewise has no trainer-owned close.
Snapshot-state release is a different operation and cannot close the trainer.

The remaining upstream decision is how a same-package C producer obtains the
five operands and which C handle owns them. Unenrollment returns the five
retained receivers to their existing callers without destroying them. A
package-owned C factory additionally needs an accepted data-only input schema
and a same-package producer for all five operands, including failure cleanup
for each producer. Neither schema nor child teardown may be inferred from the
test fixture.

Once those contracts exist, a testable private C ABI sequence is: `dlopen`;
call the accepted initializer; construct the five operands inside this exact
package; call the genuine lease constructor and publish one opaque handle only
after enrollment; reject malformed and overlapping requests without a handle;
reject busy close; close one idle exact handle and verify registry unlink and
specified child ownership; verify repeat close's specified result. A real
exception boundary must map E1 categories/statuses without exposing a partial
handle and must keep the runtime-owned arena and library loaded. The pinned
f31/LLVM21 linked-package gate already proves initializer invocation,
localization and hostile-link negatives, but proves none of these trainer
lifecycle operations. No trainer C entry or resume claim should be added until
the producer and close contracts are frozen and exercised in that gate.
