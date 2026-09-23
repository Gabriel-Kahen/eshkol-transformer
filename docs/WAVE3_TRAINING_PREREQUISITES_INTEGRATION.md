# Wave 3 training-prerequisite integration audit

This integration-preparation branch starts from the reviewed TR3-B/O plus G3-C4
candidate `2006cde9279e52f7d6994ef800a6f1f3cb9c0ec1`, tree
`e1144fec370299b293acfc94e0719d305385a9f2`. It adds only the source-approved
SHARED-R2 candidate, the nine unregistered E3 reference files, the TR3-O
documentation delta, and status/provenance reconciliation. It does not accept or
publish any candidate.

## Authenticated sources

| Component | Source commit | Source tree | Admission |
|---|---|---|---|
| TR3-B/O and G3-C4 base | `2006cde9279e52f7d6994ef800a6f1f3cb9c0ec1` | `e1144fec370299b293acfc94e0719d305385a9f2` | exact starting tree |
| TR3-B implementation | `071abb72f65fa8286bc49c54eafefe042b8c3576` | `74992144041282953ed465eb3de6b6e77d91ef8e` | implementation/test blobs retained; documentation extended with supported evidence |
| TR3-O implementation | `783dc576e534b241523e9b1e987933d5c5e433d0` | `67edabf43470b894ff8df3347ad5017889f2e847` | implementation/test blobs retained; shared topology is the reviewed base union |
| G3-C4 integration | `637cec09e00d6f9c2414e051836ab50665709606` | `df022486337c92a560147bf1f0d3a9f7a901b5b8` | provider, fixtures and prerequisite hook blobs retained; shared topology is the reviewed base union |
| SHARED-R2 | `8047cec9ccae8f39ce1e300224659eed3b15fb59` | `e80930797d556edac83fed12ea18c6c4dabcaa2b` | merged with source ancestry |
| SHARED-R2 evidence | `92c2f5853d1ca3f82c6700f07e47bf9f08052b3e` | `0217c51cdd07e8ec544b02f8a74e8446c2f89063` | evidence-only successor; ROADMAP reconciled |
| E3 reference toolkit | `37a8078ce756e7f3c22f14a046142d2f09e98807` | `a2767cbb8e22c10ac29d3e18b076de182653264b` | nine `tests/e3_reference` files only |
| TR3-O documentation | `eef6f420f64c88c3821fbc3a79660e7a2a3e4b8b` | `0c03f85bfde3469dfc18a13dddd34beef1873f48` | two exact documents plus one ROADMAP union |

The complete per-path source and integrated mode/blob comparison is recorded in
[`WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv`](WAVE3_TRAINING_PREREQUISITES_BLOBS.tsv).
Mode-and-blob comparison against the component heads gives these results:

- TR3-B: ten implementation/test paths are exact. `docs/ROADMAP.md` is the
  integration status union; `docs/TR3B_OBJECTIVE_BRIDGE.md` differs only by the
  supported focused-evidence record.
- TR3-O: ten implementation/test paths are exact. Its proposal is now exact to
  the later documentation source `eef6f420`; `Makefile` and the two CI topology
  files are the reviewed 19-suite/32-command base union.
- G3-C4: 36 provider, fixture, workflow and prerequisite-hook paths are exact.
  `Makefile`, `docs/ROADMAP.md` and the two topology files are integration unions.
- SHARED-R2: 19 implementation, supporting-document, script and test paths are
  exact to `8047cec`; its evidence document is exact to successor `92c2f585`,
  and `docs/ROADMAP.md` is the status union. This preserves I2 rank-two
  admission, the K2 v1.1 descriptor/report audit, K1 copy semantics and both
  private I2 plan gates without an ABI change.
- The TR3-O documentation source matches `docs/INTEGRATION_LOG.md` and
  `docs/TR3_O_OPTIMIZER_TRANSACTION_PROPOSAL.md` byte for byte. Its ROADMAP line
  is combined with TR3-B, G3-C4, SHARED-R2 and current evaluator status.

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

## Conflict and topology audit

The SHARED-R2 merge had no unresolved conflict; Git combined its one-line
ROADMAP addition with the reviewed base. Applying the TR3-O documentation delta
produced the sole content conflict, in `docs/ROADMAP.md`. The resolution retained
the G3-C4 row and combined the TR3-B and TR3-O evidence into one TR3 row, then the
same file was reconciled with the accepted E3 prerequisite state and explicit
incomplete states for E3, G3, TR3 and CLI3. No runtime or ABI file was hand
resolved.

`Makefile`, `tests/ci/topology.py` and `tests/ci/test_topology.py` remain byte
identical to base `2006cde`. SHARED-R2 adds no command, and the E3 reference
toolkit is intentionally unregistered, so the strict union remains 19 suites and
32 unique commands.

SHARED-R2 changes the admitted `native/f32_tensor.c` and
`docs/I2_F32_TENSOR.md` predecessor bytes. Four live predecessor manifests are
therefore refreshed as part of this integration: one row each in the L3S,
E3-METRICS and M3 shared-call manifests, and two rows in the G3-N manifest. All
other rows in all six repository predecessor manifests retain their reviewed
values and verify against the integrated tree. These five checksum-row changes
are recorded separately from source-component byte identity in the per-path TSV.

## Lightweight validation

The integration-preparation gate passed:

- `python3 -m unittest discover -v -s tests/ci -p 'test_*.py'`: 105 tests;
- `scripts/check-ci-topology.sh`: 19 suites and 32 unique commands;
- all six predecessor manifests verify against the integrated tree;
- `python3 -m unittest -v tests.m3cg.test_contract`: five shared-call contract
  and package-boundary tests;
- `python3 -m unittest -v tests.q0.test_python_isolation`: three tests;
- the E3 stdlib/schema/isolation selection: 15 tests;
- the pinned PyTorch 2.13.0+cpu E3 mathematical oracle: four tests.

No full CI, expensive native build or component full gate was run for this
integration-preparation task.

## Limits

This candidate has no root acceptance, merge, public evaluator, public trainer,
sampler, transport, generation, joint live restore, one-batch overfit, held-out
improvement or interrupted/resumed equivalence claim. SHARED-R2 still needs its
remaining K2/C2 evidence and supported integration disposition. Full-union
supported CI remains separate from the focused TR3-B supported proof and the
component-local compatibility evidence.
