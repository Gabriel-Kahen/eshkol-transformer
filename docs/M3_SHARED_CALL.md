# Shared M3 invocation and fixed14 pin implementation

Status: **accepted / complete for the bounded shared implementation** in
[issue #101](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/101).
[Root acceptance 5778924718](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5778924718)
and PR #105 merge `19f404cf21632944e1f2d5d4959e1240f9a79f5f` preserve the
independently reviewed and supported-tested tree. The
[accepted contract](E3_SHARED_CALL_CONTRACT.md) defines behavior; this record
describes implementation, evidence and its consumer boundaries.

`native/m3_call_adapters.esk` adds exactly 38 checked diagnostic entries. Each enters
the unchanged `m3-call` guard once and invokes its original trusted M3T operation.
The eight existing M3 entries and trusted internal schedules keep their source bytes.
No public facade or installed artifact changes.

`src/eshkol_transformer/m3_call_pins.h` provides the exact private fixed14 descriptor
and begin/check/end signatures. `m3_call_f32_integration.c` includes the predecessor
f32 integration once, sharing its I2 registries. Complete preflight precedes
acquisition; failed internal prefixes drain in reverse order. A full descriptor is
checked without changing controls. Release validates the complete descriptor before
its first write, drains in reverse order without allocation, and clears only pins.

The canonical CPU-f32 shapes are four `[4,4]` matrices, `[4,8]`, `[8,4]`, four
`[4]` normalization vectors, `[256,4]` tied embedding/head, two final `[4]` vectors,
and `[2,4]` positions, in the accepted order: 1,184 floats / 4,736 bytes. Preflight
validates I2 byte strides. This layer adds no numerical operation or gradient calculation.

The descriptor detects copies; it does not authenticate a consumer. A future fixed
consumer must authenticate its own registered context and derive the arrays and
storage internally. E3 frame/mode/cursor handling, G3 runtime, output publication,
public ABI, and successor production package admission remain downstream work.
Trusted opaque I2 metadata and valid caller-owned storage remain prerequisites;
this implementation does not recover from arbitrary internal memory corruption.
The parameter pin is not a privileged gradient-borrow lock: execution must remain
serialized and closed to gradient mutation and callbacks.

## Required gate and evidence

`scripts/test-m3.sh` now requires `scripts/test-m3cg.sh` after its four existing
gates. Full CI and acceptance retain 19 suites / 27 top-level commands and their
existing budgets and prerequisites. The subgate performs source-contract checks,
native tests, and fixed test-only source composition; it does not recursively
rebuild the canonical M3 package. Mutation checks reject omission or failure
masking. A mocked runner test verifies both child gates execute in order and that
the first failure propagates.

The native gate runs optimized C11 plus ASan/UBSan with LeakSanitizer explicitly
enabled. Tests cover all 14 injected acquisition-failure prefixes, 25 fail-stop
cases, malformed shapes and spans, aliases, stale/copied descriptors, count and
sentinel corruption, and original-error preservation. Failed checks compare pin,
control, shape/stride and payload snapshots taken before invocation. Release and
repeated cycles track attempted malloc/calloc/realloc/free calls.

Absent and present accumulated-gradient phases each pass 1,024 and 8,192 cycles.
Within each phase live counts stay at 28 tensors / 14 parameters and cumulative
retired storage stays at 1,872 bytes absent or 3,336 bytes present, with no borrow
events. Parameter and gradient bytes remain unchanged. Native identity anchors are
explicitly test-only; the source AOT witness uses actual P1 model handles and I2
parameters through a separately authenticated test harness.

The final source-package gate passes: all 46 reentry paths, trusted nesting, real
M3 forward/logits, manual-frame exclusion and a second idle model, plus 8,192
pin cycles using real P1/I2 ownership. The normal artifact retains 93 globals and
contains no test hooks; the instrumented artifact adds one test runner global and
localizes its observers. Raw dependency closures, C11/C++17 declaration probes,
private-link and duplicate-aggregate negatives pass. Two fresh builds produce
identical IR, objects, archive, executable and output. Both combined objects have
SHA-256 `66f4dad3ae2e0320c6f5a866bee4e33c407d7170c11368fd30e17359ba79b682`.

Local native, source-package, contract and CI-topology checks pass on the explicitly declared
CachyOS / LLVM 22.1.6 compatibility host, using pinned Eshkol sources. This is not
supported-host evidence. The measured subgate writes `gate-measurements.json` under
the build directory with monotonic elapsed time and Linux maximum child-process
RSS; RSS is not aggregate memory. Local component gates were executed separately;
the mandatory supported-CI invocation supplies the complete subgate measurement.
The PR evidence records its exact head, supported-CI result and measurement.

## Integration provenance

[Independent implementation approval 5772575224](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5772575224)
covered `e3d8566eaa33eec2c7fb2a4ef278b1c53b0d3acf`, tree
`984bc258d25d4078d8ce7201a3b4fdc92d2e8ab3`; supported run 35693360705
remains historical evidence for that original tree. The independently approved
[documentation union](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5777816153)
produced `9ae35ecbae04031bb655925545d4176681742c3d`. All 926 non-Markdown
blobs and modes stayed identical; nine incoming documents matched main and only
the E3 roadmap row was reconciled.

[Supported run 35736728847, attempt 1](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35736728847)
passed all 19 suites / 27 commands, topology, evidence and final aggregation.
It tested merge `c9e441aace9da4c617ae435923b108d4ce9fe775`, with parents
`ba0e37d06076d0a16c473ff742ab95723cb2cb89` and `9ae35ecbae04031bb655925545d4176681742c3d`.
The candidate, tested merge and actual PR #105 merge `19f404cf` all have tree
`d5e8a801eeaab3ea666f37dfd8d83daaf473f379`. Authenticated evidence job
106802892449 contains the unique matching versioned report, and the repository's
exact-tree verifier independently accepted the complete successful run. The
[final supported handoff](https://github.com/Gabriel-Kahen/eshkol-transformer/pull/105#issuecomment-5778898098)
retains those distinct commit identities.

[Main run 35745000940, attempt 1](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/35745000940)
passed at `19f404cf`. Topology job 106804199849 explicitly verified reuse of
completed PR run 35736728847; the full-suite wrapper was skipped and final job
106804306781 passed with `scope=reused`. This is verified reuse of the above
supported evidence, not a fresh execution of main's full suites.

The model-composition job took 4,050 seconds (67m30s), within its unchanged
105-minute limit. The shared subgate measured native 15.369844295 seconds,
package 925.065518476 seconds and combined 940.435587652 seconds, both child
exit codes zero. Maximum child-process RSS was 3,846,112 KiB, not aggregate
memory; the subgate timer excludes preceding static checks and outer toolchain
verification. These are one-run measurements, not an improvement or flat-total-
training-memory claim.

Focused verification at the actual merged head passed:

- `python3 -m unittest -v tests.m3cg.test_contract tests.m3cg.test_closure tests.q0.test_python_isolation`: 7 shared and 2 isolation tests.
- `python3 -m unittest discover -s tests/ci -p 'test_*.py'`: 100 tests.
- `make test-ci-topology`: unchanged 19 suites / 27 commands.
- `bash scripts/test-m3cg-native.sh`: optimized normal and ASan/UBSan/LSan with
  leak detection enabled; both stderr files empty. All 14 failure prefixes,
  25 fail-stop cases and absent/present-gradient 1,024/8,192-cycle checks passed.

This local native retest explicitly used CachyOS / Clang-LLVM 22.1.6 compatibility
mode (`ESHKOL_ALLOW_UNSUPPORTED_HOST=1`) and pinned Eshkol sources/build read-only;
it is separate from supported Ubuntu 22.04 / Clang-LLVM 21.1.8 evidence. No full
CI or package rebuild was repeated for the identical merged tree. Consumer
implementation, public evaluation/generation and full training remain downstream.
