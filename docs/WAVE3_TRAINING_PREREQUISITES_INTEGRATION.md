# Wave 3 training-prerequisite integration audit

This integration-preparation branch starts from frozen reviewed PR #124 head
`79d71086293ccd51bd569785a19c7409d1e070a9`, tree
`81128ec7c8cb7b4627dad0eeb98e10a624e4fbfd`. That base contains the reviewed
TR3-B/O plus G3-C4 candidate, SHARED-R2 candidate, nine unregistered E3
reference files, TR3-O documentation and accepted-main documentation through
PR #123. The accepted E3 integration successor adds the source-approved bounded
native-frame leaf, its CI registration and the exact E3-METRICS consumer
inventory needed for that leaf. This successor repairs both concrete failures
from the supported PR #124 run: the reviewed L3S consumer inventory and the
reviewed TR3-O GNU `nm` query. It also adds the independently approved six-file
native parity witness and registers parity against the existing pinned
development oracle. It does not accept or publish the integrated candidate.

## Authenticated sources

| Component | Source commit | Source tree | Admission |
|---|---|---|---|
| Frozen reviewed integration base | `79d71086293ccd51bd569785a19c7409d1e070a9` | `81128ec7c8cb7b4627dad0eeb98e10a624e4fbfd` | exact parent tree; reviewed PR #124 head |
| TR3-B/O and G3-C4 base | `2006cde9279e52f7d6994ef800a6f1f3cb9c0ec1` | `e1144fec370299b293acfc94e0719d305385a9f2` | prior reviewed implementation union |
| TR3-B implementation | `071abb72f65fa8286bc49c54eafefe042b8c3576` | `74992144041282953ed465eb3de6b6e77d91ef8e` | implementation/test blobs retained; documentation extended with supported evidence |
| TR3-O implementation | `783dc576e534b241523e9b1e987933d5c5e433d0` | `67edabf43470b894ff8df3347ad5017889f2e847` | nine implementation/test blobs retained; package checker advanced by the scoped portability successor |
| G3-C4 integration | `637cec09e00d6f9c2414e051836ab50665709606` | `df022486337c92a560147bf1f0d3a9f7a901b5b8` | provider, fixtures and prerequisite hook blobs retained; shared topology extends the reviewed base union by the two E3 gates |
| SHARED-R2 | `8047cec9ccae8f39ce1e300224659eed3b15fb59` | `e80930797d556edac83fed12ea18c6c4dabcaa2b` | merged with source ancestry |
| SHARED-R2 evidence | `92c2f5853d1ca3f82c6700f07e47bf9f08052b3e` | `0217c51cdd07e8ec544b02f8a74e8446c2f89063` | evidence-only successor; ROADMAP reconciled |
| E3 reference toolkit | `37a8078ce756e7f3c22f14a046142d2f09e98807` | `a2767cbb8e22c10ac29d3e18b076de182653264b` | nine `tests/e3_reference` files only |
| E3 native frame | `38fb9890cd768aad42796d83ec610304b1ee6875` | `43e364b9e9c6efa3856d58b51d0f5376aa3582b6` | five implementation/test/gate paths exact; ROADMAP and CI inventory reconciled |
| L3S inventory repair | `a6d8851cc334e8ae99b3bde9b6a75e4c2f53aa0a` | `acb1d54b4488d579fe78dfe4568b5e1cff2c6f18` | exact one-line consumer admission; evidence reconciled without replacing E3 rows |
| TR3-O `nm` portability repair | `2a9f368783ca54b7c7598f0b73126c900ef38361` | `8158bbf34a551db33171fe838df07f2438bbbb2d` | exact test-only two-file delta; no symbol assertion or runtime surface changes |
| E3 native parity | `e086a61289df13ffcd9778b6cb7f6068a912692d` | `4194e0c8f8af4623959245accd79f46535c594df` | exact six-file parity leaf only; its earlier native-frame import is not replayed |
| TR3-O documentation | `eef6f420f64c88c3821fbc3a79660e7a2a3e4b8b` | `0c03f85bfde3469dfc18a13dddd34beef1873f48` | exact proposal; integration-ledger and ROADMAP unions |
| Accepted main documentation | `e899215cff22afeb00b9a5f56bddad9c4cc6c469` | `eb0de48ecfbb0bbe24994b463d7cb9e81c9f72c3` | PR #123 merge; accepted C4 closeout and E3 prerequisite records retained |

The complete per-path source and integrated mode/blob comparison is recorded in
[`WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv`](WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv).
Mode-and-blob comparison against the component heads gives these results:

- TR3-B: ten implementation/test paths are exact. `docs/ROADMAP.md` is the
  integration status union; `docs/TR3B_OBJECTIVE_BRIDGE.md` differs only by the
  supported focused-evidence record.
- TR3-O: nine implementation/test paths remain exact to `783dc576`; its package
  checker is the scoped successor from `2a9f368`. Its proposal is exact to the
  later documentation source `eef6f420`; `Makefile` and the two CI topology
  files extend the reviewed 19-suite/32-command base union only with the two E3
  gates.
- G3-C4: 36 provider, fixture, workflow and prerequisite-hook paths are exact.
  `Makefile`, `docs/ROADMAP.md` and the two topology files are integration unions;
  the topology delta is only the two E3 gates.
- SHARED-R2: 18 implementation, supporting-document, script and test paths are
  exact to `8047cec`; its evidence document is exact to successor `92c2f585`.
  `docs/ROADMAP.md` and `docs/QUALITY_GATES.md` are documentation unions with
  accepted main. This preserves I2 rank-two admission, the K2 v1.1
  descriptor/report audit, K1 copy semantics and both private I2 plan gates
  without an ABI change.
- E3 native frame: the private source/header, native C fixture, C++ header
  fixture and standalone script retain the exact modes and blobs from
  `38fb9890`. The ROADMAP and shared Make/CI inventory are integration unions.
  No public evaluator, production source composition, runtime pin, P1 seam or
  shared-guard file is imported.
- E3 native parity: all six paths retain the exact modes and blobs from
  `e086a61`. The accepted native-frame files already present from `38fb9890` are
  not replayed from parity parent `b4ea5fc`. The shared Make/CI inventory and
  ROADMAP are integration unions.
- TR3-O `nm` portability: the package test and evidence note retain the exact
  modes and blobs from `2a9f368`. Only the global-defined query changes from the
  ambiguous `-gU` spelling to `-g --defined-only`; the existing `-gu` query and
  all symbol assertions remain unchanged.
- The TR3-O proposal remains byte-for-byte exact to its documentation source.
  Its integration-log record is preserved inside the accepted-main ledger union,
  and its ROADMAP line is combined with TR3-B, accepted G3-C4, SHARED-R2 and the
  current evaluator status.
- Eight accepted-main documentation paths are byte-for-byte exact to `e899215c`.
  `docs/INTEGRATION_LOG.md` and `docs/ROADMAP.md` are explicit conflict
  resolutions; `docs/QUALITY_GATES.md` is Git's clean nonoverlapping union with
  SHARED-R2. All accepted records from both parents remain present.

The exact E3 reference inventory is:

| Mode | Blob | Path |
|---|---|---|
| `100644` | `8cd49448f9b61886eae6ab66ec71ede05a806a2a` | `tests/e3_reference/README.md` |
| `100644` | `6711bdc0170588b1b78508ec1fd6acdfc0c2cc23` | `tests/e3_reference/__init__.py` |
| `100644` | `2760554f6fdd81bd71bbde5bf23f63165502b882` | `tests/e3_reference/corpus.py` |
| `100644` | `e60d58a1d91eee0bb709ff23e5ac81e73ceb6449` | `tests/e3_reference/reference.py` |
| `100644` | `11ff6b65bae1cb41dd7e9270ea63796693857fda` | `tests/e3_reference/test_corpus.py` |
| `100644` | `9a683c9ab2cbf29db180c0c7708cbe25efb26a47` | `tests/e3_reference/test_isolation.py` |
| `100644` | `68d22c12bbe7ca2c2d55751d50bcc6bfd8e2bcda` | `tests/e3_reference/test_reference.py` |
| `100644` | `b3075d621092f9827ff1092528a13de9aa846f7d` | `tests/e3_reference/test_transcript.py` |
| `100644` | `918921f8275693a805fb9a0ac9864244a2ea13c1` | `tests/e3_reference/transcript.py` |

The exact E3 native-frame leaf inventory is:

| Mode | Blob | Path |
|---|---|---|
| `100755` | `c4a5628637154012b90e7b41adc6c886bb2b9ded` | `scripts/test-e3-native-frame.sh` |
| `100644` | `f06b6c8e00c547640e2887ad590917ac23154122` | `src/eshkol_transformer/e3_frame.c` |
| `100644` | `382ff4905f2165feb453acf48bde0cc77d4121d1` | `src/eshkol_transformer/e3_frame_internal.h` |
| `100644` | `69cdeb391ae83fdc2916e9ddf96ef54b481bad58` | `tests/e3_native/header_cpp.cpp` |
| `100644` | `01afcf0d23a3793ce6f8f39dbcfb97ee4eba1540` | `tests/e3_native/test_frame.c` |

The exact E3 native-parity leaf inventory is:

| Mode | Blob | Path |
|---|---|---|
| `100755` | `8c0c40b3c882a7553530a081af227756e5a9c113` | `scripts/test-e3-native-parity.sh` |
| `100644` | `c3a3fe1fc9a60ef6ebba9284cbf038746094a8d0` | `tests/e3_native_parity/README.md` |
| `100644` | `debba6c5203701022a3cfc137fd469704fa86af7` | `tests/e3_native_parity/__init__.py` |
| `100644` | `39251426281f95b39664b999162a6dc2702b6a77` | `tests/e3_native_parity/generate_fixture_header.py` |
| `100644` | `b8d1683a1ecf3ab09ec46ff739dd676230f9cec8` | `tests/e3_native_parity/test_frame_parity.c` |
| `100644` | `118d43a279ecc8725fd24eefd67ce0500e6feb30` | `tests/e3_native_parity/verify_observation.py` |

The exact TR3-O `nm` portability inventory is:

| Mode | Blob | Path |
|---|---|---|
| `100644` | `4849a32d268650d8eae1d298f18f60175c04f22f` | `docs/TR3_O_NM_PORTABILITY.md` |
| `100644` | `1b9da2a9f80ddfb309312142d317d9b1d7a5660d` | `tests/o2/test_tr3_o2_step_clear_package.py` |

## Conflict and topology audit

The SHARED-R2 merge had no unresolved conflict; Git combined its one-line
ROADMAP addition with the reviewed base. Applying the TR3-O documentation delta
produced a `docs/ROADMAP.md` conflict; its resolution retained G3-C4 and combined
the TR3-B/O evidence. Merging accepted main `e899215c` later conflicted in
`docs/ROADMAP.md` and `docs/INTEGRATION_LOG.md`. The ROADMAP resolution preserves
accepted C4 completion and E3/guard state, the SHARED-R2 and TR3-B/O candidates,
and explicit incomplete E3/G3/TR3/CLI scope. The ledger resolution preserves the
accepted E3-METRICS closeout and the complete O2 candidate record in chronological
order. No runtime or ABI file was hand resolved.

The reviewed `Makefile`, `tests/ci/topology.py` and `tests/ci/test_topology.py`
19-suite/32-command union is extended by two commands. The exact standalone E3
native-frame and parity scripts each run once in both aggregates and in the
existing `model-composition` suite, whose M3 source tuple they consume. Both
compile private tuples directly and add no canonical producer or workflow suite.
Parity uses `Q0_PYTHON`, which that suite already provisions from pinned Python
3.14.6 and `tests/n2/requirements-reference.lock`. That lock is byte-identical
to `tests/q0/requirements-oracle.lock`; Python and PyTorch remain development-test
dependencies and do not enter the production runtime. The strict inventory is
therefore 19 suites and 34 unique commands.

The TR3-O `nm` repair changes no CI registration or suite topology. Its package
test remains in the existing `native-optimizer` command, and the explicit
`--defined-only` spelling is portable across the supported GNU nm 2.38 host and
newer compatibility hosts.

The E3-METRICS package audit continues to distinguish its provider source and
ABI header from compiled dependencies. Its repository-wide source-reference
inventory now separately admits only `src/eshkol_transformer/e3_frame.c` as the
reviewed consumer. An injected unknown consumer must still fail. Archive
membership, depfile closure, public exports, undefined symbols, strict-f32
arithmetic, provider allocation prohibition and predecessor hashes are unchanged.

SHARED-R2 changes the admitted `native/f32_tensor.c` and
`docs/I2_F32_TENSOR.md` predecessor bytes. Four live predecessor manifests are
therefore refreshed as part of this integration: one row each in the L3S,
E3-METRICS and M3 shared-call manifests, and two rows in the G3-N manifest. All
other rows in all six repository predecessor manifests retain their reviewed
values and verify against the integrated tree. These five checksum-row changes
are recorded separately from source-component byte identity in the per-path TSV.

The integrated TR3-B bridge is also an exact reviewed consumer of the L3S ABI
and provider accessor. The fail-closed L3S repository scan therefore admits
`native/tr3b_objective_bridge.c` in `native/l3s_source_closure.txt`. This extends
only the explicit consumer inventory: the standalone L3S compiler depfile,
single-member archive, exports, undefined symbols, build recipe and ABI remain
unchanged.

## Lightweight validation

The integration-preparation gate passed:

- `python3 -m unittest discover -v -s tests/ci -p 'test_*.py'`: 107 tests,
  including exact inventory, sanitizer retention and injected exit-73
  propagation through both aggregates and `model-composition`;
- `make test-ci-topology`: 19 suites and 34 unique commands;
- all 130 provenance rows resolve to the declared source and integrated
  modes/blobs, including the five exact E3 native-frame and six exact parity
  paths;
- all six predecessor manifests verify 118 entries against the integrated tree;
- `python3 -m unittest -v tests.m3cg.test_contract`: five shared-call contract
  and package-boundary tests;
- three Q0 isolation tests and 12 E3 stdlib corpus/isolation/transcript tests;
- the actual E3-METRICS package checker passes against a fresh local artifact,
  including the exact provider/ABI inventory, reviewed frame consumer, injected
  unknown-consumer rejection, archive/symbol/depfile closures, strict-f32 IR and
  unchanged predecessors;
- exact source identity passes for the approved E3 native-frame, E3 parity,
  L3S repair and TR3-O `nm` portability paths;
- `bash -n scripts/test-e3-native-parity.sh`, `make -n
  test-e3-native-parity`, Python bytecode compilation and `git diff --check`
  pass.

The package-check artifact above was generated on the unsupported local
LLVM 22.1.6 compatibility host. The already authenticated native-frame normal
and sanitized supported component gate was not duplicated. No full CI,
expensive native build or component full gate was run for this
integration-preparation task.

Supported PR #124 full CI run `35916891673` finished with 17 of 19 suites
passing. Its only failures are the L3S inventory omission and TR3-O GNU `nm`
option ambiguity addressed by the two reviewed repairs below; the final
`model-composition` suite passed the genuine compiled D2/M3, M3CG and TR3-B
gates. This is source evidence for the repairs, not full CI for this successor.

PR #124 head `79d71086293ccd51bd569785a19c7409d1e070a9` exposed the
missing L3S consumer row in supported full CI run `35916891673`. K1, A2 and L2
completed before `tests/l3s/test_package_contract.py` stopped at the exact
repository-inventory comparison; no L3S numerical failure was reported. The
failure log has SHA-256
`3fb02f9feb516e6f18085064aaede537f8efb654fea1c75d5d395814185e5c3d`.

The narrow repair passed a fresh network-disabled Clang 21.1.8 package audit:
the exhaustive three-entry consumer inventory, compiler depfile, one-member
archive, exports, undefined symbols, traversal and immutable predecessors all
matched. Its package log SHA-256 is
`7dba225126db184d38555047cd00b9291b1113f469a8b08ab2b21b44b2cfafed`.
The genuine pinned-`90cbd713` full L3S gate also passed 9,849 native checks,
numerical and finite-difference checks, I1/I2 composition, eleven killed
mutants, private AOT, isolation and ASan/UBSan with leak detection. That full
gate ran on the explicitly unsupported CachyOS/LLVM 22.1.6 compatibility host;
its log SHA-256 is
`49ab2159e978a360aa547f19f99a05da0545b955c197a6d0372d88e2040f3917`.
No repair full CI was dispatched.

The second supported PR #124 failure occurred only after the full O2 gate had
passed, when the TR3-O package test used `nm -gU --format=posix`. GNU nm 2.38
parses `-U` as `--unicode` and consumes the following option. Reviewed repair
`2a9f368` replaces the two global-defined queries with explicit `-g
--defined-only`; it leaves `-gu`, the expected symbols and every runtime source
unchanged. On supported Ubuntu 22.04/GNU nm 2.38/LLVM 21.1.8, all four focused
package tests and the full TR3-O transaction, coexistence, regression, privacy
and sanitizer gate pass.

## Limits

The E3 native source leaf is independently approved, and root authenticated its
supported normal and ASan/UBSan/LSan executions, each reporting 4,381 checks.
The parity leaf is independently approved with supported Ubuntu 22.04/LLVM 21
evidence for five normal, sanitizer and repeat cases, all 21 roles, 15 strict
comparisons, the 4,381-check native regression, 12 stdlib checks and four pinned
PyTorch checks. It explicitly does not exercise a real D2 borrower/cursor, P1
mode, Eshkol wrapper or public E3 API. Those component results are not integration
acceptance. The integrated candidate still requires independent exact-tree
review, supported full CI and root acceptance. It has no public evaluator,
production source composition, final runtime-pin, P1/D2 restoration,
guard-adoption, public trainer, sampler, transport, generation, joint live
restore, one-batch overfit, held-out improvement or interrupted/resumed
equivalence claim. SHARED-R2 owner gates are complete only on the unsupported
local compatibility host. Full-union supported CI remains separate from the
focused component evidence.
