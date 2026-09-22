# Shared M3 call gate

`/usr/bin/bash scripts/test-m3cg-package.sh` builds only fixed development
compositions. It does not admit a production E3/G3 package or change E1B policy.
It uses the verified compiler and compiles the inherited native source owners
once each, replacing the f32 slot with the common TU and, only in the witness,
the M3 transport slot with `model_harness.c`.

The normal composition has the inherited 93 globals and no test hooks. The
instrumented composition adds only `et_m3cg_test_run_v1`; its model/pin observers
and I2 instrumentation are LOCAL. The native harness authenticates the genuine
same-root model owner and derives its canonical 14 arrays into one stable context.
There is no test API accepting arbitrary pin storage or parameter arrays.

The witness checks all 38 M3T and eight M3 reentry paths, preservation of the outer
guard after rejection, trusted-original nesting, and genuine M3 forward/logits.
It then performs 8,192 pin cycles through separate Eshkol/native boundaries using
the same context. Every cycle checks parameter bytes and token ownership. Existing
I2 live and retired control counts/bytes at cycles 1, 1,024 and 8,192 must match.
This is pin-cycle retention evidence, not flat total-process memory or an E3/G3
consumer implementation claim.

The gate checks C11/C++17 declarations, exact raw source/native dependency
closures, normal/test symbol isolation, duplicate-registry and private-link
negatives, and two fresh instrumented source compilations with identical IR,
localized objects, archives, executables and output. Compiler and runtime output
remain under `build/m3cg-package`; the enclosing gate records elapsed time/RSS.

A test-only named-let enclosing guard expansions triggered an LLVM dominance
error on the local pinned Eshkol/CachyOS-Clang22 compatibility lane. Separate fixed
cycle/recursion functions retain the same runtime assertions and avoid that
compiler construct. The supported Ubuntu22.04/Clang21.1.8 lane remains required.
