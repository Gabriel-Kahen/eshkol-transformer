This directory separates the cheap source contracts from the genuine Eshkol
runtime witness. The runtime fixture source-composes the existing binding gate
with the real O2 extension, the accepted C2-D2 cursor-pair helper, and the CPU4
control producer. It uses one canonical seeded C4 model, its original fourteen
P1 handles and I2 carriers, a real O2 optimizer, the canonical raw-byte tokenizer,
and a real N1/T4/V256 D2 dataset. The tokenizer is constructed by
`(t2-wave2-tokenizer-byte resolved)`; the V259 special-token file fixture is
not used. Dataset/constructor/control errors print category, operation and
message before failing. It defines no native or component doubles.

`runtime_test.esk` checks zero-state construction and component identities,
constructor overlap and native READ contention, snapshot acquisition of native
WRITE, failed WRITE acquisition leaving the trainer idle, failed end preflight retaining its
exact parent/WRITE lease for retry, deep snapshot copies,
and thirteen malformed restore tuples with no canonical writes. It checks exact
stage identity, caller/stage-copy detachment, pending-stage exclusion, precommit,
abort and restaging, consumed-stage rejection, and a zero-state identity commit.
It exercises all seventeen P1 module modes through the existing slots64..68,
then ends controls and releases the durable model binding. Failed construction
rows remain inert registry tombstones as specified by the producer.

The zero-state commit tests the controls producer only: underlying model,
optimizer and dataset state already equals the staged zero state. This fixture
neither authenticates a C2 image nor stages/commits I2 parameter or O2 moment
plans. It does not prove full checkpoint restore or training-step support. G3
must supply the complete authenticated transaction and its own integration gate.

Host checks, with no compiler invocation:

```
python3 tests/cpu4_trainer_control/test_source_contract.py
python3 tests/cpu4_trainer_control/test_runtime_recipe.py
bash -n scripts/test-cpu4-trainer-control.sh
```

After independent source/recipe review, the orchestrator must run the following
inside the pinned supported image under the shared heavy lock (fresh absolute
evidence directories are mandatory):

```
bash scripts/test-cpu4-trainer-control.sh /out/cpu4-controls-normal normal
bash scripts/test-cpu4-trainer-control.sh /out/cpu4-controls-sanitize sanitize
```

The recipe reuses `native/cpu4_binding_native_sources.txt` unchanged and adds only
`native/o2_optimizer.c` and the O2 bridge compiled with
`ET_O2_NATIVE_HELPERS_ONLY`. Existing binding/C4/P1/D2 feature flags are preserved.
It compiles sequentially with strict warnings, strict floating-point flags,
`--strict-types --no-stdlib --optimize 0`, and isolated caches. Each compiled
binary runs twice with arena poison; both runs must have empty stderr and
identical PASS output. Sanitize additionally enables ASan/UBSan for native code
and linking. The script neither takes a second lock nor starts parallel builds.

Runtime and supported-image compilation: NOT RUN at source handoff.

The fe9 runtime deep-equality helper (`lib/core/runtime_deep_equal.cpp`,
vector recursion221..231 and default286..287) has no bytevector content case.
Distinct canonical UTF-8 buffers and detached nested controls therefore require
`d2-core-bytes=?`, not general `equal?`. The producer uses that exact byte
predicate for canonical configuration; the fixture's `controls=?` uses it for
current cursor, epoch cursor and canonical configuration. General RNG, tokenizer
and counter vector predicates remain unchanged. Regression cases require distinct
objects with identical payloads and reject cursor/config payload or length,
fingerprint and tuple-shape mismatches. The host helper regression is:

```
GUILE_AUTO_COMPILE=0 guile --no-auto-compile -s tests/cpu4_trainer_control/test_controls_bytes.scm
```

This repair changes no upstream runtime or checkpoint schema. The genuine
supported-image constructor/snapshot/restore gate remains the execution oracle.
