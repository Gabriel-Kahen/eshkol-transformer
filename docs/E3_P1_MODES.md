# E3-P1 mode restoration evidence ledger

Issue [#108](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/108)
implements the accepted [exact private contract](e3/E3_P1_MODES_CONTRACT.md).
This ledger starts at merged contract commit
`ba0e37d06076d0a16c473ff742ab95723cb2cb89`, tree
`221e00c9c258841995195a051b9f44df515679b9`. Pending rows are requirements,
not passing implementation or acceptance evidence.

Implementation freeze is currently held at root publication failure atomicity.
Independent direct-runtime and generated Eshkol `vector-set!` probes show the
pinned runtime write barrier returning/publishing regional graph edges after
promotion allocation returns null. The root independently confirmed this and filed
[upstream issue713](https://github.com/tsotchke/eshkol/issues/713); the
[executed AOT follow-up](https://github.com/tsotchke/eshkol/issues/713#issuecomment-5772455008)
records the six graph-copy-prefix failures. This is deterministic allocation-failure
injection, not natural host OOM, supported-lane behavior, or execution of the complete
P1 bind helper. Ordinary semantic
testing can continue, but the short-region promotion and setup-atomicity requirements
remain unresolved. No manifest/CI freeze or contract weakening follows from this
probe.

## Scope and stable boundary

Canonical P1 gains lexical slots 64–68, for a total of 69. The first 64
closures, 18 installed operations, 46 named internal operations, existing native
identity ABI, and public source remain unchanged. The five E3-only wrappers in
`native/e3_p1_modes_extension.esk` dispatch bind/prepare/enter/restore/unbind
with arities 2/1/1/1/1. They are source-private Eshkol calls and supply no
native export, generic authority, alternate P1 module, or shared guard.

The real topology has 17 distinct module nodes, including distinct tied head and
token-embedding nodes. P1 authenticates its own model/provider/topology. A fixture's
private frame key proves only P1 helper semantics. Genuine E3 frame/model admission,
the shared guard and fixed14 pins, cursor restoration, six-output publication,
and the outward E3 error table remain downstream obligations.

The canonical source is generated from the trusted section of
`templates/p1/module_roots.esk.tmpl`; both copies must agree. The unchanged
`tools/p1/module_surface.tsv` describes named provides, not the 69-element lexical
vector. Its 18 public plus 46 internal names must not be expanded to add these
source-private helpers. `scripts/check-e3-p1-contract.py` pins the inherited
closures and top-level forms to the merged base, checks the exact five wrappers,
and checks source/template agreement. It is a development structural check;
compiled lifetime and nonallocation evidence is separately required.

## Inherited package inventory

Counts below are the checked-in merged-base manifests. Every listed artifact
inherits the same canonical P1 source and consequently requires candidate evidence.
Global/export counts count the six E1 accessors separately from package exports.

| Inheritor | Manifest prefix | Globals / package exports | Eshkol sources | Native sources |
|---|---|---:|---:|---:|
| P1 | `native/p1_package` | 24 / 18 | 4 | No separate closure manifest |
| C1 | `native/c1_checkpoint` | Internal checkpoint source gate | 8 | Native I/O tested separately |
| T1 | `native/t1_wave1` | 47 / 41 | 10 | No separate closure manifest |
| T2 | `native/t2_wave2` | 47 / 41 | 12 | No separate closure manifest |
| I2 | `native/i2_wave2` | 47 / 41 | 13 | No separate closure manifest |
| O2 | `native/o2_wave2` | 53 / 47 | 15 | 29 |
| D2 | `native/d2_wave2` | 58 / 52 | 18 | Native carrier gate separately |
| K2 | `native/k2_wave2` | 59 / 53 | 15 | 29 |
| C2 | `native/c2_wave2` | 81 / 75 | 32 | 53 |
| M3T | `native/m3t_package` | 85 / 79 | 15 | 36 |
| M3 | `native/m3_package` | 93 / 87 | 18 | 40 |

Additional explicit P1-containing closures are
`t2_wave2_d1_test_source_closure.txt`,
`c2_checkpoint_load_source_closure.txt`,
`c2_checkpoint_save_source_closure.txt`,
`c2_d2_cursor_pair_source_closure.txt`,
`c2_model_encode_source_closure.txt`,
`c2_o2_encode_source_closure.txt`, and
`c2_training_state_source_closure.txt`, all under `native/`.
These require the same source and isolation review; they are not independent
registry owners to link together.

The lexical append adds no dependency path to these closures and must not make
them load the E3 wrapper file. Existing defined-symbol, public-export,
public-string, rename, archive-member, facade, and native-source manifests should
remain unchanged. A compiler-generated private-symbol or undefined-runtime delta
must be measured and reviewed individually before any relevant exact allowlist
is edited. The shared builder localizes every nonpublic definition; private symbol
renumbering alone is not justification to add exports or public strings.

The existing package policies authenticate exact repository input tuples and
paths; their source-closure lists do not hash the P1 source contents. Final
source/wrapper digests and approved builder registration must therefore be
recorded explicitly. The baseline canonical P1 source SHA-256 is
`4e8ecb142b2103f16f3883237db5596172779a29b96c39d376fdfbe64cb030d6`.
No blanket allowlist regeneration is authorized.

## Build and retention obligations

The shared E1B builder compiles strict, no-stdlib shared-library IR, then compiles
that IR with the verified native compiler. Its current command does not explicitly
set an Eshkol optimization level. Several direct predecessor tests intentionally
use `--optimize 0`; optimized lifetime probes explicitly use `-O 2`. Changing
either mode changes the evidence and requires separate approval and measurement.
Preserving source-level exports does not establish byte-identical artifacts,
generated private symbols, fixed arena costs, or compile-time resources.

The new root holder, lexical closures, and captured environments may add fixed
retention even in predecessors that never invoke E3 helpers. Measure initialized
baseline and candidate root arena bytes, native live/retired controls and bytes,
binary/object sizes, compile time, and peak RSS using identical profiles. Record
the exact compiler/source digests and actual flags for each measurement.

An active binding owns a token, record, list cell and three fixed vectors of 17
entries. Unbind clears record slots 2–7 and leaves an inert identity tombstone.
Successful and failed setup retention is cumulative and must be measured apart
from repeated evaluation. The repeated-call claim is limited to reuse of one
already-bound token/arrays. It does not establish bounded total frame creation.

Existing D2/K2/C2/M3T optimized 1,024/8,192 checks require exact root-arena
equality. Existing M3 retention evidence explicitly reports cumulative costs and
must retain its semantics and native accounting. None of those results may be
inferred from the new helper-only resource witness.

## Candidate evidence

| Proof | Required recorded evidence | Current disposition |
|---|---|---|
| Source preservation | First64, unchanged public/named surface, template agreement, wrapper slots/arities, source digests | Static draft check passed; exact final candidate pending |
| Topology/modes | Real17-node tree; exact mixed train/eval restoration; all identity/topology/mode corruptions rejected before writes | Pending |
| Lifecycle/statuses | Exact 0–4 statuses, every phase transition, copied/foreign/dead tokens and comparator reentry | Pending |
| Root lifetime | Bind in short region; read back canonical token; poisoned exit and successful reuse | Pending |
| Setup atomicity | Every allocation and barrier failpoint; untouched setup output/modes; dead published records clear frame/model roots | Pending |
| Nonallocating tails | Allocation-disabled prepare/enter/prepared-restore/entered-restore/unbind; exact counters | Pending |
| Reuse | Separate 1,024/8,192 token-reuse arena/native/cumulative allocation counters and peak RSS | Pending |
| Inherited packages | Each row above: exact public/local/undefined/source/native/archive inventories, public-caller/hostile-link negatives, determinism | Pending |
| Fixed setup cost | Per-inheritor baseline/candidate root bytes and artifact/build resources; success/dead/failed bind separately | Pending |
| Supported acceptance | Exact candidate commit/tree, independent review, supported Ubuntu22/LLVM21 run and complete original suite union | Pending |

Local CachyOS/LLVM22 measurements, if performed, are compatibility evidence only.
They do not establish supported Ubuntu22/LLVM21 behavior, sanitizer/leak success,
or accepted resource bounds. No runtime result, PR acceptance, downstream E3
composition, or roadmap completion is claimed by this ledger.

## Recorded blocked-work checkpoint

The independently reviewed diagnostic is commit
`c388b42d6f9444a07450fbd0760a3b8f572cd5dc`, tree
`9452b5615cc4e0f9e3aa744c1b239e550090a8d8`, based on the contract commit above.
It contains only the diagnostic script and three small fixtures. It is deliberately
absent from default green gates. No PR or full CI was launched for this blocked
checkpoint.

The unaccepted helper/source-static checkpoint is commit
`f8015ef8eb84443c22e649e473b6840551ada563`, tree
`432f6fff1838a25270cea640451da1d96a045960`. Canonical module SHA-256 is
`805388596d234c9d652074983d4a23f11b5dc740a50bc99abf6c0c55ee8920ed`;
the wrapper SHA-256 is
`9e6580896b98a46156763ad70b5466c98bd98476d063f68dd0d31c672cdfa646`.
This checkpoint is unsuitable for integration until the publication prerequisite
and all pending runtime/package gates are satisfied.

Exact compatibility invocation:

```sh
ESHKOL_SOURCE_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-src \
ESHKOL_BUILD_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build \
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=llvm-config \
/usr/bin/bash scripts/test-e3-p1-runtime-barrier.sh
```

Result: **exit1, required promotion guarantee violated**. The normal C++ and
Eshkol AOT cases report zero regional edges and survive poisoned region exit.
For each execution route, injected stages0..5 leave respectively6/5/4/3/2/1
regional graph edges. Every failure case retires its holder before region exit;
no reclaimed pointer is dereferenced. The test changes the target arena's bounded
capacity/used controls temporarily and resets them only for diagnostic teardown;
this is not a production rollback mechanism or a natural-memory-exhaustion claim.

The script records exact compile/link/run commands, provenance and artifact hashes
under `build/e3-p1-runtime-barrier-diagnostic/`. Compiler SHA-256 is
`3e0b923e2e272a89474dff6739b6a2023d71ae483863a64b685ec8ea71ca1cc0`;
runtime archive SHA-256 is
`580946c291d88e2008de02202b8e07942e7b85c0ac9c6b7f861397c1d115e240`.
No shared dependency was modified or rebuilt.

Source-static checks pass with `python3 scripts/check-e3-p1-contract.py`,
`/usr/bin/bash scripts/generate-p1-roots.sh --check`, and `git diff --check`.
The initial full-source smoke compilation exposed one draft parenthesis defect,
which was corrected. Its replacement emitted IR over canonical M3/P1 and the
five wrappers. Independent inspection of that draft IR found compiler-generated
nursery allocations/interrupt polls around mutating named-let loops. The current
source replaces the fixed17/6 store tails with explicit stores, captures root-owned
mode/provider symbols, and checks exact integer phases. It has **not** been
recompiled or proved nonallocating; the older IR is defect evidence only. The
obsolete AOT build was stopped after the publication hold and source corrections;
no smoke execution, current candidate runtime success, resource slope, or supported
acceptance is claimed.

Three independent Astra/high lanes handled implementation, lifetime diagnostics,
and package evidence. The lifetime lane's executed findings were independently
confirmed by root; that agent later encountered a platform safety/model error
before completing final design review. The implementation lane independently
reviewed the source corrections and the [upstream design proposal](e3/E3_P1_CHECKED_PROMOTION_PROPOSAL.md).
It found no additional design blocker beyond that proposal's explicit upstream,
coexistence and evidence conditions. This is conditional design review, not a
patch-ready verdict or approval of the blocked implementation for integration.
