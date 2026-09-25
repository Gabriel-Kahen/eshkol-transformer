# TR3-C private C trainer entry: contract blocker

Status: historical source-backed C-entry blocker. A separate private
lease-unenroll operation ends registry retention. The subsequent
[bounded private factory candidate](TR3_C_PRIVATE_FACTORY_CONTRACT.md) now
defines a data-only C request and proves a narrow linked create/close path;
its full failure matrix, production exports, public trainer and resume remain
unaccepted. The analysis below records why the earlier initializer-only
package could not construct a trainer.

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

## Producer and C ownership decision at `ccf1ab4`

The genuine source producers can be called together only inside the compiled
identity universe. Their accepted inputs are separate operations, not one
accepted C request:

| Operand | Existing source operation | Input and lifetime constraint |
| --- | --- | --- |
| Resolved X1 configuration | `x1-private-config-parse` then `x1-private-config-resolve` | UTF-8 configuration text and an explicit override list; immutable caller-owned result. |
| T2 tokenizer | `t2-wave2-tokenizer-byte` | Resolved configuration; selects only the byte tokenizer path. The public contract also permits artifact loading, but this fixture does not implement that choice. |
| D2 dataset | `d2-token-dataset-open` | Exact ten-key options, finite D1 directory and tokenizer; newly owned receiver with `d2-token-dataset-close!`. |
| M3T model | `m3t-public-initializer-state` then `m3t-public-model-create` | Seed and resolved configuration; this is the fixed diagnostic profile, with process-local model lifetime and no destructor. |
| O2 optimizer | `module-parameters` then `o2-public-optimizer-create` | O2's distinct versioned data-only configuration must name the model's canonical parameter paths; live moment tensors are process-local, with no optimizer destroy. |

The concrete `tests/tr3_snapshot/runtime_smoke.esk:69-110` producer chooses a
fixed X1 JSON string, the byte tokenizer, D2 bounds and packing, a corpus
directory from its test command line, a model seed, and a one-group constant
AdamW O2 configuration. Those choices are fixture policy. In particular, X1
does not carry the D2 directory/options or O2 group configuration. The accepted
public `trainer-create` contract takes five existing receivers; it does not
transfer their ownership to the trainer or specify a data-only factory request.
`docs/CLI3_PROPOSAL.md` deliberately reserves `pretrain` without flags or an
accepted trainer producer. A constructor that hardcodes the fixture's choices
would silently turn test policy into a package API.

The failure cuts prevent a safe opaque-handle factory from being inferred.
After D2 succeeds, a later M3T or O2 failure can close the D2 receiver, but
M3T has no model destructor; after O2 succeeds, neither model nor optimizer
has a destroy operation. `tr3-lease-create-internal` leaves all five operands
with their caller on failure. Successful `tr3-lease-unenroll-internal!` removes
only the trainer registry edge, leaving the five operands live. A C factory
that owns them therefore needs an explicit process-lifetime retention policy
or reviewed child teardown; neither is accepted. The private initializer's
`READY/BUSY/RUNTIME/ARENA/HANDLER/EXCEPTION` statuses cover initialization
only, not E1 trainer categories or a C handle's invalid/repeat/busy states.

**Decision:** do not add a producer or export a C trainer constructor yet.
The smallest required contract is a versioned data-only request selecting the
X1 source, T2 byte/artifact mode, exact D2 options and corpus identity, M3T
initializer seed/profile, and O2 logical configuration or an explicit rule
for deriving its paths. It must define validation order and limits; which
party owns every published child receiver; cleanup/retention at every failure
cut and after unenrollment; an opaque handle's identity, repeat/forgery/busy
semantics; E1-to-C status mapping and result lifetime; and the library/runtime
lifetime. These are contract questions, not an implementation proposal or
accepted public API. The current linked symbol manifest exposes only the
versioned initializer; a C caller cannot construct or close a trainer through
it. The pinned linked gate's hostile-link and `dlsym` negatives enforce that
boundary, while source-only fixture construction does not prove C ownership.

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

## Same-package primitive ingress finding

The linked object already contains localized C ABI thunks for X1, O2 and the
TR3 lease constructor; external `dlsym` does not expose them. The pinned
Eshkol runtime can
copy a NUL-terminated C string into its arena and construct tagged exact
integers. A bridge must first bound and validate the byte input, including
embedded NUL, because the string copier uses `strlen`. It can then call a new
same-package Eshkol wrapper with only primitive tagged arguments. That wrapper
would build the exact D2 options and O2 configuration as Eshkol data and call
the existing five-operand producer sequence in one identity universe. Passing
a raw C-built Scheme list or a tagged value from another package is not an
accepted ingress. The initializer pops its E1 handler before returning, so
each factory call needs its own exception scope and categorized error capture.

A bounded candidate for the next private proof is one construction attempt per
initialized process, byte tokenizer only, fixed M3T profile, caller-supplied
X1 JSON, D2 options, initializer seed and O2 hyperparameters. The package
would retain created children until process exit; idle close would unenroll
only the trainer lease. This is a proposed policy, not an accepted public
factory or resource-destruction contract. It avoids claiming repeated use
while M3T and O2 have no destroy operation. The next gate must compile and
invoke the primitive wrapper through the actual linked package, exercise
malformed inputs and every producer failure cut, prove no partial C handle,
and check exact idle unenrollment, busy/forged/repeat close, and child
retention. Until then the exported manifest remains initializer-only.
