# N4 private K1 sticky-status witness candidate

This test-only descendant of `03b698a7` retains its genuine C1 checkpoint,
D1 `03 04` corpus, D2 `[3,0] -> [4,0]` mask `[1,0]`, complete M3 sparse
forward oracle, and real private K1 dispatch. It changes no model, logit,
target, mask, CE, public ABI, or provider. The frozen subnormal packet remains
separate. This source and its host tests provide **no compiled admission**.

`E3_N4_STICKY_MODE` is accepted by the private overflow package builder only
when the sparse observer is enabled. Modes 0/1 form a clear/seed pair; 2/3
deliberately clear the seeded status before/after dispatch. The macro exists
only in that noninstalled native package. Immediately before K1, mode 0
clears x87 and MXCSR sticky status, and mode 1 additionally raises masked
x87 divide-by-zero and sets MXCSR invalid/inexact sticky bits. Neither path
changes x87 or MXCSR control bits. The observer samples x87 control/status and
MXCSR before and after the actual `et_kernel_runtime_dispatch` call, before
formatting its `E3N4FENV` record. Mode 2 erases the seed before the first
sample; mode 3 erases it after dispatch but before the second sample.

The new parser requires supported, unchanged control bits, both seeded x87
and MXCSR flags before K1 and their preservation afterward. Arithmetic may
raise additional flags. It delegates the remaining transcript to the existing
sparse parser, which checks the authentic D2 row, all 21 M3 roles, 512 K1
input words, CE, K1 count/output, accumulator, final result, and footer.
Host tests reject both flag-clearing mutants, changed controls, forged D2
mask, and changed K1 word. The paired clear control must enter K1 with zero
sticky status. These tests do not prove what flags the natural M3 path leaves
before the test seam, nor unsupported timed control changes.

The next gate needs independent source review, then a fresh exact-source
normal/repeat/sanitized package and AOT run with separate clear and seed
processes plus both mutant kills, source/fe9/Q0 custody, and independent
transcript/IR review. Until then sticky-status preservation at private K1 is
**NOT RUN**.
