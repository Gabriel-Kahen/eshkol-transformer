# E3-D2 #106 bounded prerequisite

This implements only sections 1–3 of the accepted
[E3-D2 contract](E3_D2_CONTRACT.md): deterministic D2 source preparation,
canonical T1 raw/empty identity admission, and the private native idle query.
It is not an E3 evaluator or a production E3 aggregate. Frame-owned restore,
three staging consumers, owned-batch cleanup, shared exclusion, and repeated-frame
retention remain dependent work under the accepted contracts.

## Added private contracts

- `scripts/generate-e3-d2-source.py --output-dir DIR` reads the fixed checked-in
  manifest and original D2 source, validates exact predecessor/form/output hashes,
  then writes `source/e3_d2_dataset.esk` and
  `source/e3_d2_source_provenance.json`. The only source change is the complete
  `d2-tokenizer-identity` definition. No source/include/root/substitution override
  is accepted. Output and provenance bytes are deterministic; generated files
  are build artifacts, not checked-in alternate D2 implementations.
- `e3-d2-byte-identity/1` authenticates through the canonical T1 registry and
  returns a fresh `[fingerprint,256]` vector only for the eight-slot tagged
  raw/empty core. Foreign or wrong-kind values produce `invalid-argument`;
  authentic unsupported profiles produce `unsupported`, both attributed to
  `token-dataset-open`. The fingerprint comes from the canonical accessor and
  remains detached. Existing D1 fingerprint/vocabulary mismatch rejection is
  unchanged and still precedes dataset publication.
- `et_e3_d2_dataset_idle_preflight_v1(const void *owner)` returns and records D2
  status 1 for null/unknown/closed identities, 2 for an enrolled owner with any
  current batch, and 0 for an enrolled idle owner. The pointer is compared, never
  dereferenced. Only existing `last_status` changes. The uninstalled new TU
  includes the original D2 C file once, preserving its static registry authority.

The predecessor pins are the accepted `d16224de…` D2 source,
`645ec4c0…` native C, and `165dc018…` header. Source generation and the focused
closure checker enforce the complete hashes; native symbol equality is not used
as a substitute for the C/header pins.

## Focused proof and exact test tuple

Run `/usr/bin/bash scripts/test-e3-d2.sh` after configuring the pinned toolchain.
The script verifies the compiler checkout, binary and provenance through
`common.sh`. It needs no prebuilt project archive. Native objects and generated
sources are built in a fresh `build/e3-d2/run.*` directory. Failures preserve that
directory; `E3_D2_KEEP_EVIDENCE=1` also preserves successful evidence.

The test root loads the actual T1 Wave1 root, canonical C1 SHA256, D2 semantic
core, new identity adapter, generated D2 source, and the test fixture. It uses
one real E1/P1/D1/X1/C1/T1 identity universe. All 16 ordered compiler dependencies
are frozen in `tests/e3_d2/expected_source_closure.txt`; the seven native objects
and their complete non-system compiler dependencies are frozen in
`expected_native_closure.json`. No original D2 dataset source, T2 root, M3 root,
shared E3 guard, frame, or second aggregate enters this prerequisite tuple.

The pinned compiler emits the real depfile in `--compile-only` mode; executable
mode does not emit it despite accepting `--emit-depfile`. The test therefore
compiles the same exact root separately for dependency evidence and for two
fresh AOT executables. Complete raw depfiles and symbol inventories remain in
the evidence directory. Five structural mutations demonstrate rejection of
missing T1, original D2 substitution, and unexpected source/native dependencies.

The gate covers deterministic generation and fixtures; all authentic
raw/strict/special/prefix/suffix T1 profiles; persisted raw save/load; detached
identity; foreign, wrong-kind and copied-core rejection; unchanged D1 mismatch
behavior without native enrollment; and recovery by a later valid open. Prefix
and suffix profiles necessarily include specials under accepted T1 rules, so
those cases do not independently isolate the adapter's prefix/suffix predicates.

Native tests compare control/allocation/generation/counter snapshots, exercise
null, unknown, unreadable, closed and reused identities, unpublished/sealed/live/
borrowed batches, release and generation exhaustion, and disable allocation and
I/O at the seam. ASan/UBSan/LSan are mandatory. The same production TU passes
self-contained C11 and C++17 header/link probes. Exact globals differ from the
original D2 object by only the new idle function; production objects contain no
test hooks. Original-only linking cannot resolve the new seam, and linking both
original and variant objects fails for duplicate definitions.

Compiled Eshkol probes also call the exact idle FFI on authentic D2 owners:
idle 0, live/borrowed 2, released 0, closed 1. A batch and its lease stay inside
one lexical `with-region`, with poisoning enabled. The final test executable
localizes exactly the three prerequisite functions and their two Eshkol metadata
symbols listed in `expected_localized_symbols.txt`, then reruns successfully.
This is bounded test-artifact isolation, not future E3 package export acceptance.

## Evidence status and limits

Local execution uses CachyOS and Clang/LLVM 22.1.6 with the explicit unsupported
compatibility flag, the clean pinned Eshkol revision
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`, and existing compiler binary SHA256
`3e0b923e2e272a89474dff6739b6a2023d71ae483863a64b685ec8ea71ca1cc0`.
The canonical source/build are reused read-only; no duplicate compiler build or
shared dependency modification is involved. This is not Ubuntu 22.04/LLVM
21.1.8 supported-lane acceptance.

Initial setup attempts exposed an absent `llvm-config-21`; a first override used
the wrong variable, then `LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config` selected
the installed LLVM22 tool explicitly. The first package compile exposed the
missing C1 SHA256 load, fixed by loading the existing canonical source. The next
AOT completed 43 initial identity checks in 72.420 seconds, then the dependency
gate failed on the missing executable-mode depfile; the dedicated compile-only
path above resolves that measured compiler behavior. These were setup/evidence
failures, not concealed runtime or numerical failures.

The final local focused driver exited 0, with evidence retained at
`build/e3-d2/run.AdgyG4` and stdout/stderr captured in
`/tmp/e3-d2-final.{stdout,stderr}`. It passed 16 generator tests, 408 native
status/resource snapshots in normal and ASan/UBSan/LSan builds, and 63 genuine
T1/generated-D2 checks in each of two fresh strict AOT executions. Executables,
stdout and saved canonical tokenizer bytes compare byte-identically. Both exact
closure checks, five adversarial closure mutations, header/link probes, normal
symbol inventories and the localized executable rerun passed. Original D2/T1/T2
production sources/build tuples and the shared compiler checkout remain unchanged.

| Measured command | Elapsed seconds | Child maximum RSS (KiB) |
|---|---:|---:|
| Real compile-only closure | 69.267 | 3,069,036 |
| Fresh AOT compile A | 69.096 | 3,068,676 |
| Fresh AOT compile B | 68.662 | 3,069,756 |
| AOT execution A | 0.104 | 28,080 |
| AOT execution B | 0.095 | 27,672 |
| Localized execution | 0.091 | 29,784 |
| Native idle | 0.001 | 14,904 |
| Native ASan/UBSan/LSan | 0.008 | 23,604 |

These single-host measurements include process startup and linked-runtime costs.
The native observer separately requires zero allocation/free/I/O calls during
idle probes, unchanged control/payload/generation snapshots, and zero live
native datasets, carriers, leases, owned allocations and D2 file descriptors at
teardown. The tested batch payload is exactly 34 bytes for two i64 `[1,2]` planes
and a bool `[1,2]` plane; this is not the process RSS. No long-run retained-memory
claim follows from this small fixture.

Exact local command (existing shared compiler reused read-only):

```bash
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 \
LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
ESHKOL_SOURCE_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-src \
ESHKOL_BUILD_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build \
E3_D2_KEEP_EVIDENCE=1 /usr/bin/bash scripts/test-e3-d2.sh
```

`/usr/bin/bash scripts/check-ci-topology.sh` passes 19 suites/28 commands;
`python3 -m unittest discover -q -s tests/ci -p 'test_*.py'` passes 100 tests;
`python3 -m unittest -q tests.q0.test_python_isolation` passes three tests after
the exact build-generator registration below;
`git diff --check` passes. CI unit-test output containing a synthetic successful
run ID is fixture evidence only, never supported compiler-suite acceptance.

The initial PR CI run `35698295743` failed before launching any full compiler
suite: the Python-isolation gate rejected the newly tracked
`scripts/generate-e3-d2-source.py`. The earlier local two-test pass had run before
staging that generator, while the gate enumerates `git ls-files`; it therefore
did not establish isolation for the complete candidate. The correction admits
only that exact contract-selected build utility outside `tests/`, with negative
checks for nearby names, nested scripts and production roots. The package
manifest still forbids Python/PyTorch, and the actual compiled closure still
excludes all Python files. The corrected tracked-candidate isolation gate passes
three tests. This failure is retained as CI history, not reported as a supported
runtime run or retried without a source change.

Independent source/provenance and native/lifetime agents approved the
adapter/native implementation. Independent package source review reconciled the
actual complete manifests and rejected earlier partial-inventory checks. The
native reviewer separately approved registration, C/C++/link/isolation gates and
failure propagation. The independent native/lifetime reviewer then inspected the
final candidate run, repeated binary/stdout comparisons and exact closure/hash
checks, and approved the local runtime evidence without blockers. The supported
candidate, union and merged-main dispositions are recorded separately below so
that local compatibility evidence is not confused with supported acceptance.

## Integration acceptance

PR #110 candidate `f10acc953b5ab343ed87bdda19a48c3a783b1e40`, tree
`cbf145b6af3f986d76053756d2b07cc0dafa22f8`, passed supported Ubuntu 22.04,
LLVM/Clang 21.1.8 run
[35698763745](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35698763745),
attempt 1. All 19 suites and the then-current 28-command topology ran; suite
evidence and final aggregation passed. The actual PR test merge was
`fb9783a408fcb2e38b5cbc3b7dfb261129ee03dd` and had the candidate tree exactly.

The combined prerequisite union retained every E3-D2 implementation, generator,
closure, test, isolation and toolchain-pin blob byte-for-byte. PR #112 head
`b052385e742626fcfc2557b3b5a9f176f6234b29`, tree
`2256194d2daeb233ec1745d0039b383d4358e3d5`, passed supported run
[35744832879](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35744832879),
attempt 1: all 19 suites, the expanded 29-command topology, suite evidence and
final aggregation. The emitted evidence commit
`4678160e249fb6477bd4b34be2717f8f3fe77682` has the same tree. Its shard-loader
job reran this gate in full, including 16 source tests, 408 native snapshots,
mandatory ASan/UBSan/LSan, 63 compiled identity checks, both exact closure checks
and five closure mutations. The largest compiler child maximum RSS was
3,043,812 KiB; the largest AOT execution child maximum RSS was 29,824 KiB.

PR #112 merged to main as `33a54ef7256f43d9e1ce82152915f7bded51bc24`
with that same tree. Main run
[35798192984](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35798192984)
validated and reused the exact-tree supported evidence; it did not rerun the full
suites and is not reported as fresh compiler evidence. A separate focused test
from a clean detached checkout of the exact merge reused the pinned compiler
read-only and passed the complete E3-D2 driver. It reproduced 16 source tests,
408 native snapshots, sanitizers, 63 compiled checks per required execution,
both exact closures and five mutations. The two fresh AOT executables and output
were identical. Focused merged-head measurements were:

| Measured command | Elapsed seconds | Child maximum RSS (KiB) |
|---|---:|---:|
| Real compile-only closure | 75.978 | 3,069,932 |
| Fresh AOT compile A | 75.930 | 3,069,656 |
| Fresh AOT compile B | 74.205 | 3,069,216 |
| AOT execution A | 0.098 | 29,528 |
| AOT execution B | 0.087 | 28,060 |
| Localized execution | 0.086 | 30,144 |
| Native idle | 0.001 | 14,908 |
| Native ASan/UBSan/LSan | 0.008 | 21,360 |

On the exact merge, `scripts/check-ci-topology.sh` passed 19 suites/29 commands,
the three focused Python-isolation tests passed, and an explicit candidate-to-
merge comparison confirmed byte preservation of the complete E3-D2 component
and closure inputs. Independent merged-head review found no component or closure
blocker. These process measurements do not establish long-run retention.

This closes only the bounded sections 1–3 prerequisite. Full M3/E3 composition,
frame authentication and root-stable construction, staging, rollback, restore,
owned-batch cleanup, shared exclusion and repeated-frame retention remain
downstream E3 obligations. No public evaluator or new-runtime acceptance follows
from this integration.

Development Python is used only for fixture generation, source preparation,
timing and evidence checks. The delivered identity adapter and compiled dataset path are Eshkol;
there is no Python runtime dependency. Timing reports provide wall elapsed time
and child-process maximum RSS, not aggregate concurrent-process memory or a
steady-state retention claim.

Full M3/E3 composition, canonical frame/root-stable dataset construction,
restore invariants, stage transport, terminal cleanup behavior, shared exclusion,
and 1,024/8,192 repeated-frame retention require the future independently reviewed
E3 artifact. No public API, full evaluator, training, GPU, or performance claim
follows from this prerequisite.
