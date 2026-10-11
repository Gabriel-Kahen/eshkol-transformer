# E3 M2 physical-capacity one-use compiled gate

Status: **prepared, not launched**. This gate is a test-only descendant of the
independently source-approved `f666f7a6a29a96f63693126e130a5199337cfd8b`
(tree `af52d5fd6b30ff35361d9893c665c05667235556`). The independent ACCEPT
seal is `e3-m2-physical-capacity-independent-review-f666f7a6-20261011-a/`
`SHA256SUMS-REVIEW`, SHA-256
`2295b9d6fc2162f39f9029e2150664e29b7e0e29eeb7b5e21a38a657c4acd7b2`.
That seal covers the candidate, not this gate descendant. The gate requires a
separate, clean checkout at the exact candidate commit and tree. It builds
packages and optimized AOT executables only from that checkout.

The candidate's ordinary parser rejects `E3BATCH active=2` against authentic
`E3ROW mask=10` before its independent numerical comparator runs. This gate
descendant adds an opt-in `--physical-capacity` parser mode. It admits exactly
three batches and only the third-row `10`/active-2 pair; the default parser
still rejects that mismatch. It preserves raw stdout and the actual active
count. It does not replace, normalize or synthesize the observed row.

In the supported Ubuntu 22.04/LLVM 21, pinned f31 environment, with absolute
`Q0_PYTHON` set to Python 3.14.6/PyTorch 2.13.0+cpu, run the gate script once:

```bash
scripts/test-e3-m2-physical-capacity-one-use.sh \
  /path/to/clean/f666f7a6-source-checkout \
  /path/to/fresh/e3-m2-proof-output
```

The script checks the exact source commit/tree and source review seal, requires
clean source and gate checkouts, and rejects an existing output path. It checks
the pinned compiler before any build. It generates the independent D2 corpus
and M3 reference twice and compares bytes. It builds two normal and one
physical-capacity private packages, checks exact inventories, and compares
normal objects, archives, raw depfiles and AOT binaries. It compiles all three
under the pinned optimized `--strict-types --no-stdlib -O 2` route and runs only
authentic `packed-single`. Raw stdout/stderr, input/toolchain identities,
fixture files, package evidence, compile logs, observations and verifier logs
remain in the fresh output. The exit trap writes `GATE-STATUS` and a sorted
`SHA256SUMS-GATE` for every completed or failed attempt.

For PASS, normal A/B observations and full 63-role rows must match; normal
must pass the frozen reference. The mutant must complete the same M3/D2 run
with emitted masks `11,11,10`, unchanged IDs, next targets, roles, logits and
CE, and K1 active count 2 only on row 2. Its parsed observation must reach the
independent verifier and fail first and exactly at
`batches[2].reduction_f32: exact rational reduction mismatch`, without
publishing a mutant transcript. The normal final counters are `(5,3)` and the
mutant's `(6,3)`; neither substitutes for the required first value failure.

This one-use gate has **not run** and therefore supplies no compiled AOT
admission, mutant kill, sanitizer result or E3 §9 M2 acceptance. It covers the
single packed `10` physical-capacity denominator mutant. Other M2 mutations,
five-layout numerical parity, poison/sanitizer, current-main union, N3 model
controls, edge cases, rollback and complete §9 acceptance remain separate.
