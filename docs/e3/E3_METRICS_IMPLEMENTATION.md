# E3-METRICS bounded numerical implementation

This implements issue [#107](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/107)
against the accepted [exact BOOL contract](E3_BOOL_METRICS_CONTRACT.md), merged in
PR #104 at `ba0e37d06076d0a16c473ff742ab95723cb2cb89`, tree
`221e00c9c258841995195a051b9f44df515679b9`. Independent exact-head review, supported
candidate CI and root integration are still required. This document does not mark
E3 or its private composition complete.

## Artifact and explicit discovery

`include/eshkol_transformer/e3_evaluation_metrics_abi.h` declares ABI 1.0 and its
sole accessor, `et_e3_metrics_kernel_provider_v1`. The single source
`native/e3_evaluation_metrics_provider.c` produces the single-member archive
`build/e3-metrics/libeshkol_transformer_e3_metrics.a`. Link it explicitly before
K1 and `-lm`. L2, L3S, I1 and I2 are composition-test dependencies only.

The descriptor owns exactly one verified capability, `e3.evaluation-metrics`,
with three sorted operations: `accumulate`, `correct.bool` and `finalize`, each
prefixed by the capability name. Request admission is exactly CPU f32
`[1,2,256]`. Operand shapes, BOOL bytes, i64 counters, disjoint storage and error
precedence are unchanged from the accepted contract. No canonical K1 provider
symbol, K2 discovery change, tensor owner, public Eshkol evaluator, serialized
format, graph or backward operation is introduced.

Correctness uses lowest-index argmax and canonical BOOL weights. Accumulation
uses physical f32 state after every batch, with exact token/batch controls capped
at 16,384/8,192. Finalization returns loss, mask weight, perplexity and accuracy in
that order; it computes perplexity once from the rounded global loss. Validation
stages every result before output publication, preserves the complete incoming
x87/MXCSR environment and `errno`, and rejects incompatible FP controls.
Invocation preserves `errno` and may accrue ordinary arithmetic flags. The
provider retains no pointers and allocates no heap storage. Its result union is
32 bytes; additional locals are bounded scalar storage.

## Focused evidence gate

`scripts/test-e3-metrics.sh` consumes the canonical K1/I1/I2/L2/L3S and metrics
archives from the current build directory. Shared prerequisite and mandatory CI
registration are coordinated by the root task. It requires the pinned oracle
through absolute `Q0_PYTHON` and verifies the compiler/source pin through
`scripts/common.sh`. No compiler build or shared dependency mutation occurs in
this focused test script.

The gate checks:

- Warning-clean C11/C++17 consumers, exact exports/undefined symbols/member and
  source/depfile inventories, unchanged predecessor hashes, source traversal
  anchors and optimized LLVM IR/object exclusion of binary64/FMA/fast arithmetic.
- An unchanged provider-free K1 report, plus the exact metrics report in `C` and
  `C.UTF-8`. K1 retains its eleven baseline entries; the explicit metrics provider
  adds its sole verified capability, for twelve report entries.
- Independent rational binary32 rounding and high-precision Decimal exponential
  reference, pinned deterministic development fixture generation twice, real
  L2/L3S composition through genuine I1/I2 synchronous borrows, malformed-call
  and complete byte/FP-control/errno checks, and compiled mutation controls.
- Two fresh strict pinned-Eshkol private AOT builds and runs, with identical
  executable/stdout bytes. Their inventoried test-only bridge invokes native
  error/atomicity checks and real L2/L3S/I1/I2 composition. A negative link proves
  the bridge is absent from normal production archives.
- Optimized sanitizer variants with ASan, UBSan and LSan; leak detection is fixed
  at `detect_leaks=1`. Save/restore failure precedence is exercised by test-only
  linker wrappers around `fegetenv` and `fesetenv`.

The rational model explicitly separates CE tolerance against the mathematical
reference from bit-exact ordered aggregation of observed f32 L2 outputs. The
Decimal oracle permits at most two binary32 ULP for finite `expf` results and
requires correct finite/overflow classification at frozen neighbors. It does not
claim bitwise libm agreement across platforms. Compiled mutant evidence records
numeric, admission and atomicity kills separately from equivalent local controls;
compilation failure, crashes and unsupported dispatch do not count as numeric
kills.

## Evidence status and limits

The complete local compatibility gate passed on September 22, 2026:

```bash
ESHKOL_SOURCE_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-src \
ESHKOL_BUILD_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build \
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 LLVM_CONFIG_EXECUTABLE=llvm-config \
Q0_PYTHON=/home/gabe/.codex/worktrees/552e/eshkol-transformer/.tmp/o2-venv/bin/python \
  /usr/bin/bash scripts/test-e3-metrics.sh
```

It passed 70,231 native checks, 135 save/restore-fault checks, five rational model
tests and five reference tests. The independent Decimal oracle measured maximum
0 ULP across 102 exact-f32 inputs, including the frozen exponential overflow
neighbors. All twelve provider numeric, five admission/atomicity and five
orchestration mutants were killed, and all five compiled equivalent controls
passed. Both fresh private AOT executable/stdout comparisons and the complete
ASan/UBSan/LSan runs with `detect_leaks=1` passed. Native and private AOT compilations
were warning-clean. The pinned development PyTorch environment emits its existing
missing-NumPy warning; this fixture does not use NumPy.

The run took 24.215 seconds wall, 20.580 seconds user and 3.574 seconds system CPU,
with a peak child-process RSS of 312,424 KiB. `/usr/bin/time` was unavailable: the
first launch failed before running tests and was preserved. A development-only
Python wrapper measured `time.monotonic` and Linux
`resource.getrusage(RUSAGE_CHILDREN)` for the successful run. This is complete-gate
host resource evidence, not kernel throughput or tensor-retention evidence.
Local logs are `.tmp/e3-metrics-evidence/focused.stdout`, `focused.stderr`,
`time.json` and `launch-failure.stderr`. The optimized local object/archive sizes
were 17,160/17,424 bytes; the deterministic oracle/provenance fixtures are
22,727/411 bytes.

An independent provider subreview found no blocker in the package driver,
source/depfile/IR inventories, exact bridge boundary, negative linkage, fresh AOT
comparisons or sanitizer registration. Independent package review inspected
production alias/control flow, full sentinel coverage and genuine carrier
borrow/release paths; identified test coverage gaps were fixed before this run.
The integrating task must record the exact candidate commit/tree and external
review/CI disposition. No supported-lane result is inferred from local evidence.

The local compiler is the unmodified read-only pinned Eshkol
`90cbd7130f47b8184bcc77b8d5c1b0026da980de` (`1.3.4-evolve`), built with LLVM/Clang
22.1.6 on CachyOS. This requires the explicit unsupported-host compatibility
override. Ubuntu 22.04 x86-64 / LLVM-Clang 21.1.8 remains the supported lane.
The prerequisite I2 aggregate produced existing LLVM 22 vectorization warnings;
they are not evidence of supported compiler behavior.

Unsupported cases remain other devices/dtypes, non-BOOL weights, arbitrary
shapes, strided/offset storage, empty successful finalization, graph gradients,
in-place state, hidden higher-precision arithmetic or fallback. Native live and
readable pointer preconditions, immutable borrowed metadata during dispatch and
disjoint error storage are caller obligations. This is not a native pointer
sandbox or protection against concurrent mutation.

Real E3 frame ownership, genuine M3/D2 no-grad traversal, transactional restore,
trusted metric-history authentication, publication, retention and eventual public
execution remain downstream E3 obligations. This numerical AOT bridge proves
private compiled reachability only and supplies no frame or generic authority.
