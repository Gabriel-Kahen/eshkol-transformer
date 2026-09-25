# Versioned trainer facade package boundary

The canonical `lib/transformer/trainer.esk` now provides the A0
`trainer-create resolved tokenizer dataset model optimizer` constructor and
`trainer-release! trainer` operation, alongside the unchanged C2
`trainer-state-release! state`. `trainer-release!` ends the exclusive lease
while leaving all five caller-owned receivers live. Dropping the trainer
reference alone does not release its lease. Forged or repeated release is
`invalid-argument`; a busy trainer is `invalid-state`. The public E1 operation
is `trainer-release!`, and the private unenroll operation is not exposed as
an error cause. These semantics are implemented by the reviewed private
same-aggregate constructor and unenroll path.

This canonical facade **requires** the single
`libeshkol_transformer_tr3_public_installed.a` aggregate built by
`scripts/test-tr3-public-installed-linked.sh`. Its bridge adds the C2
state-release entry to the reviewed TR3 release candidate in the same
registry-owning object. The linked caller imports `transformer.trainer`,
constructs authentic operands through test-only same-aggregate entries,
checks create/release/re-lease and negative identity/lease behavior, and
checks the retained C2 state-release name. The two copied facade files and
the archive form one versioned package tuple. The prior 48-source candidate and
the production initializer-only TR3 package remain unchanged.

The accepted C2 archive is a distinct versioned tuple. `scripts/build-c2.sh`
copies the byte-pinned pre-TR3 `transformer/trainer.esk` into its artifact's
`facades` directory; the source fixture is
`tests/c2/legacy_facades/transformer/trainer.esk` (SHA-256
`b2f3818cc7645a9accf397479e3e4f859d73b635e39752310e76cd067921b9a8`).
To use `libeshkol_transformer_wave2.a`, put its sibling `facades` directory
before `lib` in the Eshkol include path. `scripts/test-c2-public.sh` checks
the copied bytes and depfile selection, then runs its original exact C2
archive, public API, runtime, and deterministic rebuild assertions. The C2
object, archive, symbol manifests, and original trainer facade bytes are not
changed. A client migrating to the canonical trainer facade must also switch
to the single TR3 aggregate; mixing that facade with the C2 archive is an
unsupported package tuple.

This is an intermediate trainer facet. Ordinary installed producers for all
five operands, `trainer-step!`, training/evaluation, state/load, and resume
remain separate work. The linked test-only producers do not make those APIs
installed or accepted.

The exact `57ca36b` source/tree passed both versioned package gates. The C2
gate retained 81 global definitions, 75 public exports, 81 public-name
strings, 32 Eshkol sources and 53 native sources, and passed two byte-identical
rebuilds and its linked public checks. Its archive is SHA-256
`09b1c87bf6d1ebfb2a5c95888252fcf7e0364b71532266adaa65d1396837fab1`;
the complete evidence is sealed at
`/home/gabe/.codex/evidence/eshkol-transformer/tr3-public-installed-c2-57ca36b-20260925/`
(`SHA256SUMS` SHA-256 `ed69bd39204af1af53c36a1e7f0284d1c1bfe461d29252d9eb6a48a519f067a2`).
The separate pinned TR3 linked gate passed the public caller with 49 Eshkol
sources, 30 native objects, 69 native source/header paths, 22 localized public
exports, and 173 undefined runtime symbols. Its evidence is sealed at
`/home/gabe/.codex/evidence/eshkol-transformer/tr3-public-installed-linked-57ca36b-20260925/`
(`SHA256SUMS` SHA-256 `9932eec33411c078f23bf9d147a86295be40357bb80638706e34d9b8fcbdf393`).
Both seals passed `sha256sum -c`.

## X1 operand facade successor candidate

The next isolated candidate copies the unchanged six-operation
`lib/transformer/config.esk` into the same TR3 archive's facade bundle.
Parse/resolve were already exported; the four accepted X1
validate/canonical/fingerprint/ref operations now use their existing
source-private definitions, the exact X1 package rename signatures, and
boxed wrappers in the same native bridge. The linked caller imports
`transformer.config`, exercises all six public operations and typed failures,
derives the M3T initializer seed from the genuine resolved configuration,
and passes that same receiver to the installed trainer constructor. T2, D2,
M3T, P1 and O2 creation remain test-local direct entries. The C2 versioned
tuple and the prior sealed `57ca36b` package evidence remain unchanged.
This successor is pending its full linked gate and exact manifest measurement.
