# Opt-in C2 joint SAVE span trial

Status: source and recipe candidate only. This successor to the Q0-clean C2
span-composition witness has **no joint AOT or memory result**. It changes no
public C1/C2/CLI entry, production callback, native default object, checkpoint
bytes, package manifest, or default operational gate.

The accepted SAVE path calls `c2-save-model-encode-internal` once in each of
its measure and encode child stages. The trial stages the existing 16 MiB
joint operational runtime and sets that private seam to either the ordinary
whole-bytevector C2 model encoder (`whole`) or the opt-in C1 span encoder
(`spans`). Both staged runtimes load the same test-only extension, and both
build the same macro-gated native format object, which exports one test-only
span helper alongside the ordinary C2 SHA callback. The staged gate scripts are
byte-identical; the runtimes differ only in that selector. The recipe compares
the two native format objects byte-for-byte and checks both SHA symbols occur
exactly once. The source checkout is mounted read-only. No supported
production source is patched at runtime.

The prior sealed matched pair at
`/home/gabe/.codex/evidence/eshkol-transformer/c2-sha-matched-pair-e3280e3-20261126/RESULTS.md`
measured native whole-buffer versus scalar SHA on the same fixture. This trial
tests a different question: whether skipping the C1 SHA **preimage
concatenations** changes memory use while retaining native SHA in both arms.
The exact saved fixture has 62 C1 model tensors and a 16,774,462-byte nested
model. Per model build, the avoided tensor preimages total 16,776,019 bytes;
the avoided container preimage is 16,774,461 bytes. The three successful
joint SAVEs each build the model twice, so the span arm skips 378 temporary
preimage allocations totaling 201,302,880 bytes. These are cumulative
allocation sizes, not concurrent live bytes or an
RSS estimate. The tensor payload copies, model image, outer SAVE allocation,
writer snapshot, and other C1 concatenations remain. The recipe asserts these
exact per-model and six-build totals against its generated joint fixture.

Light source checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B tests/c2/test_c2_sha_span_save_trial.py --check
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest tests.q0.test_python_isolation
bash -n scripts/launch-c2-sha-span-save-trial.sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest tests.c2.test_c2_sha_span_save_receipt
```

After an orchestrator-assigned build slot and host resource check, the exact
clean commit/tree can be run once with a fresh evidence path:

```sh
C2_SHA_SPAN_SAVE_TRIAL_SLOT=assigned \
  bash scripts/launch-c2-sha-span-save-trial.sh EXPECTED_HEAD EXPECTED_TREE \
  /home/gabe/.codex/evidence/eshkol-transformer/c2-sha-span-save-trial-UNIQUE
```

The launcher acquires orchestration, heavy-build, then pinned-Docker locks
nonblocking. It checks the source, runner, image, and prior matched-pair host
resource floors, then runs `whole` followed by `spans` in the same pinned
network-none Ubuntu 22.04/LLVM 21 image. Each arm uses one Clang AOT, a fresh
joint fixture and artifact directory, the accepted 26-check runtime gate,
strict runtime RSS/watchdog limits, `/usr/bin/time -v`, and phase markers. It
compares both outputs within each arm to its fixture and the fixtures across
arms, compares compiler stdout byte-for-byte and joint stdout after masking
only measured numeric fields, records per-arm hashes, and seals logs. Raw
stdout remains in the evidence. Order is fixed, one run per arm;
host activity and cache state remain possible confounders. A pass would be a
fixture-local observation, not a production dispatch or full C2 admission.
Exit 0 requires a nonempty checksum manifest and a successful
`sha256sum -c`; an existing trial failure keeps its original exit code even
when evidence sealing also fails.

Before integration, review the selector and staged script transformations,
run the supported C1/C2 package and joint operational gates on an exact
candidate tree, verify symbol/manifest closure, and repeat or counterbalance
the memory measurement if an RSS claim is needed.
