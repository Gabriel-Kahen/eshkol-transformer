# M3T diagnostic transport implementation

M3T implements the [accepted transport contract](M3T_TRANSPORT_PROPOSAL.md)
for `eshkol-diagnostic-byte-decoder-v1`. Implementation acceptance, supported
exact-head CI and integration merge remain pending. M3T is in review in the
[roadmap](ROADMAP.md); M3 owns the production model schedule and full-model
forward/gradient parity.

## Public and private contracts

The installed `transformer.diagnostic_transport` facade has exactly 38 operations.
It supplies typed initializer identities, construction of a P1 module, reusable
input/logits/workspace owners, and fixed-role primitive/VJP dispatch. The accepted
proposal lists every arity, selector, tensor shape and readiness dependency.

The only profile is CPU f32 with N=1, T=2, V=256, D=4, Hq=Hkv=2, Dh=2, L=1
and F=8, learned positions, three LayerNorms, tied embedding/head weights and no
dropout. Input ingress owns i64[1,2]; IDs must be exact real signed-i64 values in
[0,256). Logits ingress/egress copies exactly 2048 bytes for f32[1,2,256].
Construction publishes 15 logical paths over 14 canonical parameter identities.
Initializer state is authenticated by registered identity; inspecting or copying
its words grants no authority. An initial counter of zero advances to 290 for
one model, and the cached successor can seed another model.

Workspaces snapshot inputs, parameter bytes and mode at begin, and upstream
logits at VJP begin. Readiness admits each fixed destination only once per frame.
Exact primal checks reject mutation and accept exact restoration. One model has
at most one active frame. Reset reuses storage; release permanently closes an
owner and is idempotent. Public errors carry the invoked operation, bounded
data-only diagnostic fields, and `cause #f`.

The private construction protocol preserves P1 slots 0..58 and all ten C2 seams,
appending five entries at 59..63: 46 private / 64 total Eshkol entries and,
independently, 36 private native identity symbols. A preallocated owner anchor
authenticates all 14 native parameters even before P1 registration. Abort checks
the entire owner and each enrolled canonical binding before revocation, then
restores the exact saved I2 registry/count without allocation. Seal and abort
clear temporary anchor/baseline references. Ordinary I2/C2 construction rejects
as unsupported before allocation. The standalone f32 public API is unchanged;
scoped guards exist only in the reviewed private M3T source-including tuple.

## Packaging and execution

Build the canonical I2 prerequisite with `scripts/build-i2.sh`, then build M3T
with `scripts/build-m3t.sh`. Both use the repository's verified pinned toolchain.
Link only `libeshkol_transformer_m3t.a` and import its installed facades. The
archive has one registry-owning object, exactly 85 globals and 79 package exports.
Its six facades are config, module, byte tokenizer, error public, error consumer
and diagnostic transport. Checked-in manifests bind source/native dependency
closures, globals, exports, undefined symbols and archive membership.

Exactly one registry-owning aggregate may be linked into a process. M3T does not
compose with D2/O2/K2/C2, export private carriers, accept arbitrary providers or
shapes, or execute a complete production model schedule. Training composition,
rank-two checkpoint support and end-to-end training remain downstream work.

## Verification and evidence boundaries

`scripts/test-m3t.sh` runs native and construction gates, two fresh aggregate
builds (including hostile ambient include settings), installed-facade AOT,
binary determinism, exact manifests, private-symbol and wrong-arity negatives,
duplicate/cross-aggregate rejection and poisoned-region retention comparisons.
CI adds one diagnostic-transport suite to the accepted planner, retaining all
existing commands: 17 suites and 24 top-level commands.

The native gate passed normal and ASan/UBSan/LSan runs on the explicit local
CachyOS/Clang 22 compatibility host. It checks independent literal provider
wiring for all 21 forward and 25 reverse roles, without using production role
descriptors as its oracle. It exercises 431 K1 discovery, 182 owner I2, 318
workspace I2 and eight workspace I1 allocation failpoints, plus M3T allocation
failpoints, stale restoration and failure-byte preservation. Existing accepted
providers retain their own numerical-gradient evidence.

The native reuse witness measured 110 live I2 tensors, 1520 M3T control bytes and
5824 retained I2 control bytes at 10, 100 and 1000 frames, with stable addresses
and zero growth. The scoped guard witness passed 30,000 allocation-disabled
cycles without retained-shell growth. These are fixed-fixture transport reuse
measurements, not full training-memory or performance claims.

The construction wrapper gate passed 1123 checks. Both zero and partial
enrollment attempt all 14 positions and seven hold modes. Five holds are
reachable while unbound; gradient-reset and gradient-contribution plans require
a canonical identity and instead verify explicit invalid-state with unchanged
counts. All seven holds are tested on bound partial-enrollment positions. The
native owner gate separately tests all seven holds on each of 14 bound
parameters. Rejected aborts preserve all carriers and permit release/retry.

The canonical package and two independent fresh packages passed exact manifests,
installed public execution, reverse import order, private/arity/cross-registry
negatives and real compiler dependency checks. Each public runtime/reverse/arena
depfile must contain exactly its caller and all six installed facades; missing,
empty, incomplete, duplicate, foreign and aliased dependencies reject. Eight
package tests, 98 CI tests, five Python-isolation tests and the 17-suite /
24-command topology check passed.

Package objects and archives match byte-for-byte across separate build roots and
against the canonical artifact. Caller objects, executables and stdout match
after each fresh package is cleanly installed at the same path. This is fixed-path
caller reproducibility: the two tested installation paths differed in one embedded
facade diagnostic filename byte and, in the executable, the derived 20-byte build ID.
No relocation-invariance claim is made.

Local evidence is composed: the full driver exercised native/construction gates
and both independent builds, then the final amended caller/remainder commands
were replayed against those retained artifacts. This is not an untouched final
script PASS. The amendments require real compile-only depfiles (executable mode
does not emit them) and a stable caller installation path, preserving every raw
comparison. Earlier public assertions were corrected for E1's sorted detail keys
and the compiler's two precise arity diagnostics. All final gate commands have
local evidence; supported Ubuntu 22.04 / LLVM-Clang 21.1.8 CI remains required.

The final public reuse witness retained exactly 4,325,376 arena bytes at both
1024 and 8192 frames. Supplementary peak RSS was 94,460 and 94,376 KiB,
respectively; it does not substitute for the exact retained-allocation counter.
Owner creation, fresh allocating accessor snapshots, errors and arbitrary
application allocations are outside the success-only repeated-frame retention
claim. Reusable begin/VJP snapshots are included in the repeated frames.
Failed constructions retain small identity tombstones while reclaiming native
parameter storage and active construction capacity. General model destruction
and arbitrary carrier revocation are outside this contract; public release
operations cover input, logits and workspace owners.

P1 native sanitizers, runtime (419 checks), registry atomicity (169), construction
(23), public/private negatives and publication proof passed. Its long driver
crossed a source correction and rejected mixed-source objects; a fresh frozen-root
object subsequently matched byte-for-byte. The affected C2 model-encode and
training-state-owner gates passed before the final constructor-only begin
refinement. Supported CI must run the full final P1/I2/E1B/C2/T1 regressions.
Detailed local logs, provenance, failed attempts and review hashes are retained
under `build/m3t-evidence/` outside Git.
