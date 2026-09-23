# Wave 3 training-prerequisite integration audit

This integration-preparation branch starts from frozen reviewed PR #124 head
`79d71086293ccd51bd569785a19c7409d1e070a9`, tree
`81128ec7c8cb7b4627dad0eeb98e10a624e4fbfd`. That base contains the reviewed
TR3-B/O plus G3-C4 candidate, SHARED-R2 candidate, nine unregistered E3
reference files, TR3-O documentation and accepted-main documentation through
PR #123. This branch adds only the source-approved bounded E3 native-frame leaf,
its one-command CI registration, and status/provenance reconciliation. It does
not accept or publish the integrated candidate.

## Authenticated sources

| Component | Source commit | Source tree | Admission |
|---|---|---|---|
| Frozen reviewed integration base | `79d71086293ccd51bd569785a19c7409d1e070a9` | `81128ec7c8cb7b4627dad0eeb98e10a624e4fbfd` | exact parent tree; reviewed PR #124 head |
| TR3-B/O and G3-C4 base | `2006cde9279e52f7d6994ef800a6f1f3cb9c0ec1` | `e1144fec370299b293acfc94e0719d305385a9f2` | prior reviewed implementation union |
| TR3-B implementation | `071abb72f65fa8286bc49c54eafefe042b8c3576` | `74992144041282953ed465eb3de6b6e77d91ef8e` | implementation/test blobs retained; documentation extended with supported evidence |
| TR3-O implementation | `783dc576e534b241523e9b1e987933d5c5e433d0` | `67edabf43470b894ff8df3347ad5017889f2e847` | implementation/test blobs retained; shared topology extends the reviewed base union by the E3 command |
| G3-C4 integration | `637cec09e00d6f9c2414e051836ab50665709606` | `df022486337c92a560147bf1f0d3a9f7a901b5b8` | provider, fixtures and prerequisite hook blobs retained; shared topology extends the reviewed base union by the E3 command |
| SHARED-R2 | `8047cec9ccae8f39ce1e300224659eed3b15fb59` | `e80930797d556edac83fed12ea18c6c4dabcaa2b` | merged with source ancestry |
| SHARED-R2 evidence | `92c2f5853d1ca3f82c6700f07e47bf9f08052b3e` | `0217c51cdd07e8ec544b02f8a74e8446c2f89063` | evidence-only successor; ROADMAP reconciled |
| E3 reference toolkit | `37a8078ce756e7f3c22f14a046142d2f09e98807` | `a2767cbb8e22c10ac29d3e18b076de182653264b` | nine `tests/e3_reference` files only |
| E3 native frame | `38fb9890cd768aad42796d83ec610304b1ee6875` | `43e364b9e9c6efa3856d58b51d0f5376aa3582b6` | five implementation/test/gate paths exact; ROADMAP and CI inventory reconciled |
| TR3-O documentation | `eef6f420f64c88c3821fbc3a79660e7a2a3e4b8b` | `0c03f85bfde3469dfc18a13dddd34beef1873f48` | exact proposal; integration-ledger and ROADMAP unions |
| Accepted main documentation | `e899215cff22afeb00b9a5f56bddad9c4cc6c469` | `eb0de48ecfbb0bbe24994b463d7cb9e81c9f72c3` | PR #123 merge; accepted C4 closeout and E3 prerequisite records retained |

The complete per-path source and integrated mode/blob comparison is recorded in
[`WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv`](WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv).
Mode-and-blob comparison against the component heads gives these results:

- TR3-B: ten implementation/test paths are exact. `docs/ROADMAP.md` is the
  integration status union; `docs/TR3B_OBJECTIVE_BRIDGE.md` differs only by the
  supported focused-evidence record.
- TR3-O: ten implementation/test paths are exact. Its proposal is now exact to
  the later documentation source `eef6f420`; `Makefile` and the two CI topology
  files extend the reviewed 19-suite/32-command base union only with the E3 gate.
- G3-C4: 36 provider, fixture, workflow and prerequisite-hook paths are exact.
  `Makefile`, `docs/ROADMAP.md` and the two topology files are integration unions;
  the topology delta is only the E3 gate.
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
19-suite/32-command union is extended by one command. The exact standalone E3
native-frame script runs once in each aggregate and in the existing
`model-composition` suite, whose M3 source tuple it consumes. It compiles its
private tuple directly and adds no canonical producer or workflow suite. The E3
reference toolkit remains intentionally unregistered. The strict inventory is
therefore 19 suites and 33 unique commands.

SHARED-R2 changes the admitted `native/f32_tensor.c` and
`docs/I2_F32_TENSOR.md` predecessor bytes. Four live predecessor manifests are
therefore refreshed as part of this integration: one row each in the L3S,
E3-METRICS and M3 shared-call manifests, and two rows in the G3-N manifest. All
other rows in all six repository predecessor manifests retain their reviewed
values and verify against the integrated tree. These five checksum-row changes
are recorded separately from source-component byte identity in the per-path TSV.

## Lightweight validation

The integration-preparation gate passed:

- `python3 -m unittest discover -v -s tests/ci -p 'test_*.py'`: 106 tests,
  including exact inventory, sanitizer/header retention and injected exit-73
  propagation through both aggregates and `model-composition`;
- `make test-ci-topology`: 19 suites and 33 unique commands;
- all 122 provenance rows resolve to the declared source and integrated
  modes/blobs, including the five exact E3 native-frame paths;
- all six predecessor manifests verify 118 entries against the integrated tree;
- `python3 -m unittest -v tests.m3cg.test_contract`: five shared-call contract
  and package-boundary tests;
- `python3 -m unittest -v tests.q0.test_python_isolation`: three tests;
- `bash -n scripts/test-e3-native-frame.sh`, `make -n test-e3-native-frame`
  and `git diff --check` pass.

The already authenticated native-frame normal and sanitized component gate was
not duplicated. No full CI, expensive native build or component full gate was
run for this integration-preparation task.

## Limits

The E3 native source leaf is independently approved, and root authenticated its
supported normal and ASan/UBSan/LSan executions, each reporting 4,381 checks.
Those component results are not integration acceptance. The integrated candidate
still requires independent exact-tree review, supported full CI and root
acceptance. It has no public evaluator, production source-composition, final
runtime-pin, P1/D2 restoration, guard-adoption, parity, public trainer, sampler,
transport, generation, joint live restore, one-batch overfit, held-out
improvement or interrupted/resumed equivalence claim. SHARED-R2 owner gates are
complete only on the unsupported local compatibility host. Full-union supported
CI remains separate from the focused component evidence.
