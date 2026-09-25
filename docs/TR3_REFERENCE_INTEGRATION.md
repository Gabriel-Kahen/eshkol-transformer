# TR3 development-reference integration

This candidate adds the exact ten `tests/tr3_reference` files from independently
approved commit `cc4749003cd65711f0c320d2f3cd398b305928eb`, tree
`e79a71d7b0d8e3ed1772cc1291150ff4bf41692d`, to runtime-union candidate
`d9887fbb1cd3d3b48c51a4d841e8eaa86650726c`, tree
`e4bf84dc0910768d436c7411227a12efc15a83b0`. The base's supported runtime gates and
integration acceptance remain separate requirements. This local successor does
not change the PR #124 head or its ongoing CI run.

The [per-file inventory](TR3_REFERENCE_INTEGRATION_BLOBS.tsv) records exact Git
modes and blobs. Integration also verifies the manifest's ten source SHA-256
identities and the frozen trainer-proposal identity at `3e28db6`. Imported
reference contents match the reviewed leaf. Production APIs, runtime dependencies,
and CI registration retain the base's contents.

The generator produces 33 trajectory points for updates 0 through 32, including
all 14 parameters and 28 optimizer moments, authentic physical D1/D2 corpora,
weighted held-out evaluation, accumulation controls, and cursor/EOS replay cases.
The consumer checks artifact hashes, exact JSON types, tensor shapes, physical
batch identities, and cross-artifact numerical consistency. Bulk generated
artifacts remain outside Git.

## Evidence and limits

The author and original independent reviewer each passed the seven schema and
isolation tests and six pinned Python/PyTorch reference tests. The reviewer also
rejected the original, cross-artifact, and exact-type refreshed-metadata mutation
probes. On this integration base, both commands passed again:

```sh
python3 -m unittest -v tests.tr3_reference.test_schema tests.tr3_reference.test_isolation
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE \
  /home/gabe/.codex/worktrees/552e/eshkol-transformer/.tmp/o2-venv/bin/python \
  -m unittest -v tests.tr3_reference.test_reference
```

Results: seven schema/isolation tests and six pinned reference tests passed,
including frozen-manifest agreement, byte-identical independent generations, and
refreshed-digest semantic/type negatives. PyTorch emitted the existing optional
NumPy-unavailable warning. The ten imported Git blobs, all ten source SHA-256
identities, and the proposal SHA-256 matched. `git diff --check` passed.

The frozen generated identities are:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| corpora.json | 12,133 | `f4bad109f5a9307889e0905955c69d17b52bf5815e37409a2e11cd3895abe6d1` |
| summary.json | 47,626 | `6c3893a549e5687afa8a126c52b5e2c4d77878bee7aec6090a637ad575f31682` |
| trajectory.json | 1,423,659 | `f88997988d4520ae3eba6556e2cf0c234b206ea04e8cd28d2f86835e5207eea7` |

Summed and combined gradients are not bit-identical: their measured maximum
difference is `1.1920928955078125e-7`, within the frozen `2e-6` tolerance. The
incorrect per-batch-mean control remains distinguishable.

These are development references only. Native one-batch overfit, held-out
improvement, exact full-state checkpoint resume, failure atomicity, bounded
retention, and compiled execution still require the implemented trainer and its
acceptance tests. No Wave 3 workstream is marked complete by this import.
