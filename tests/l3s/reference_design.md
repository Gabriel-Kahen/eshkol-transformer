# L3S independent numerical review and evidence

This development test design follows the bounded contract
[accepted by integration](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5747250069).
It uses only merged A0/L2/K1/Q0 and I1/I2 carriers, with no M3
API. The earlier exact-rational probe remains independent of runtime execution.

## Proposed domain and exact arithmetic

The profile contains exactly two CPU-f32 per-token CE values, logically `[1,2]`.
Each is finite and nonnegative; signed zero is accepted and locally canonicalized
to positive zero. Masks are either canonical one-byte bool values 0/1 or finite
nonnegative f32 values; negative zero is likewise accepted as positive zero.
Noncanonical bool bytes, negative nonzero masks, and every nonfinite value reject.
All inputs, including zero-masked CE, are validated. No mask short-circuits a bad
CE value. These are mathematical CE VJPs evaluated in binary32, not derivatives of
the piecewise-constant rounded machine implementation. Masks are nondifferentiable.

For increasing token indices 0,1, compute `W = f32(W + weight[i])` from positive
zero, and require positive finite W. For reduction compute each
`p[i] = f32(weight[i] * loss[i])`, then `N = f32(N + p[i])` from positive zero.
The reported mean is `f32(N / W)`. Every intermediate must remain finite even when
a later exact-real quotient would be finite. The numerator seed is the converted
weight vector; the mean seed is each separate `f32(weight[i] / W)` division.
Computing one reciprocal and multiplying is a different binary32 contract.

Gradual underflow is accepted, including positive mathematical N or mean rounding
to zero, and a positive weight's mean seed rounding to zero. A positive subnormal W
is admissible. An all-zero objective with positive W is successful. An all-zero
mask is invalid. No epsilon, mask clamp, numerator clamp, or higher-precision sum
is introduced. Canonicalized nonnegative inputs yield positive-zero outputs.

Require the reviewed x86-64 IEEE binary32 lane, `FLT_EVAL_METHOD == 0`, explicit
noncontracting arithmetic, round-to-nearest-even in x87 and MXCSR, masked floating
exceptions, and FTZ/DAZ disabled. Validate without repairing controls. Preserve the
complete incoming floating environment during validation; successful invocation
may accrue ordinary flags. All borrowed bytes and controls stay stable throughout
dispatch. Dry evaluation precedes any caller-output write.

## Independent exact probe

`python3 tests/l3s/reference_probe.py -v` uses `Fraction` arithmetic and explicit
ties-to-even rounding, without C arithmetic, NumPy, Torch, or any L3S implementation.
It tests denominator distinctions, bool/f32-equivalent mathematical values, zero
entries, zero objective, subnormal W, underflow, overflow at each reduction stage,
division-versus-reciprocal and contraction witnesses. It cannot establish provider
behavior, byte atomicity, ABI admission, or L2 transcendental accuracy.

Useful exact witnesses:

- Loss `[2,10]`, mask `[0.5,2]`: N=21, W=2.5, mean=8.4 (rounded to f32), seed
  `[0.2,0.8]` (each independently rounded). This kills missing/repeated
  normalization, division by token count, and reversed token pairing.
- Mask `[3,4]`: first mean seed is `0x3edb6db7`; multiplication by the rounded
  reciprocal of 7 incorrectly yields `0x3edb6db8`.
- First product `2^-24`, second product `(1+2^-23)^2`: separate multiply/add gives
  numerator `0x3f800002`, contraction gives `0x3f800003`.
- Minimum subnormal loss times 0.5 rounds to +0; minimum subnormal W remains
  valid. `[FLT_MAX,FLT_MAX]` weights reject overflow, as do loss `FLT_MAX` with
  weight 2 and two `FLT_MAX` products with unit weights.
- Final-mean-only overflow: losses both `FLT_MAX`, masks with bits
  `3df9528a/3e1c9ee4` produce finite products `7df95289/7e1c9ee3`, finite W
  `3e8ca414` and finite N `7e8ca414`. The exact quotient of those rounded values
  is `2^128`; only the final division overflows. The independent rational probe
  checks every intermediate, establishing that the final-result check is needed.

## Two-token accumulation-order limitation

There is no honest numerical mutation test that distinguishes reversing **only
the accumulation traversal** of two finite nonnegative terms, with a positive-zero
initial accumulator and round-to-nearest-even. The first addition is exact and
the remaining two-operand addition is commutative. The overflow outcome is likewise
the same. This applies independently to N and W. The exact probe checks examples
and demonstrates the three-term counterexample `[2^24,1,1]`; that third term is
outside the proposed runtime profile and supplies no three-token claim.

Preserve the specified increasing index order by source review. Mutation tests
must distinguish reversed **token pairing** or reversed **seed ordering** and label
those mutations precisely. Expanding runtime support merely to create a failing
order mutation requires a new accepted shape contract. Treat accumulation-only
reversal as an equivalent mutant at this scope, not as a detected bug.

## Real L2/PyTorch/finite-difference evidence

The development fixture uses pinned Python 3.14.6 / PyTorch 2.13.0+cpu, Q0 canonical
bytes and source/lock identities. Generation requires `ATEN_CPU_CAPABILITY=default`
and `MKL_CBWR=COMPATIBLE` before Torch import. Deterministic
CPU logits `[1,2,256]` should have visibly different token loss and target indices,
with nonuniform values within each row. Include bool/all-ones f32 agreement,
weighted `[0.5,2]`, asymmetric zeros, and masks scaled by a positive common factor.
Scaling leaves the mathematical mean gradient unchanged and scales the numerator
gradient; use tolerances where f32 rounding prevents bit identity.

Use real accepted L2 forward, L3S reduction/seed, then real L2 backward. Compare
per-token loss, N, W, mean, both seeds and both logit gradients against the frozen
reference. Independently evaluate double-precision stable log-sum-exp and analytic
`mask * (softmax-onehot)` and `(mask/W) * (softmax-onehot)` on the development side.
Use Q0 metadata checks, existing L2 justified forward `atol=2e-6, rtol=2e-5` and
gradient `atol=3e-6, rtol=3e-5` as initial limits, recording measured errors before
acceptance. Exact rational comparisons cover arithmetic-only boundary witnesses.

Central finite differences of an independent double objective should cover both
targets and non-target logits in both tokens; use explicit epsilon and error
bound. Also perturb representative production f32 logits and observe real composed
mean/numerator differences with an empirically justified f32 FD tolerance. Avoid
near saturation and severe cancellation for FD. Analytic parity still covers all
512 output entries. Zero-masked gradient entries must be exact zero in magnitude;
L2's frozen signed-zero behavior is preserved rather than redefined by L3S.

Mutations must run against observed real composed outputs, not solely compare two
copies of reference formulas: omit normalization, apply W twice, divide by token
count, interchange mean/numerator seeds, reverse seed token positions, reverse CE
and mask pairing, and replace per-element division by reciprocal multiplication.
Document accumulation-only reversal as equivalent at two terms. The independent
adversarial workstream owns complete input/output byte snapshots and numerical
failure admission, including poisoned output buffers and signaling NaNs.

## Local measured results

On the explicitly unsupported local LLVM 22.1.6 lane, the warning-clean optimized
native runner passed exact arithmetic and real L2 composition for ones, weighted,
zero and commonly scaled masks. Eight representative logit perturbations per mask
and seed kind gave maximum absolute native-f32 finite-difference errors
`5.99622726e-05` (numerator, bound `1.6e-4`) and `1.19954348e-05` (mean, bound
`8e-5`), with central step `0.03125`. The independent double analytic comparison
over every gradient entry gave maximum absolute error `3.74637521e-08`; all four
frozen-reference/identity/analytic/scaling tests passed.

`test_mutations.py` compiles modified copies of the actual provider, links real
L2/K1, and runs the independent native runner. Eleven arithmetic mutants fail
numeric assertions: reversed pairing, misplaced seed outputs, missing/double/count
normalization of each mean or mean seed, numerator/mean seed swap, reciprocal
substitution and contraction. The unchanged baseline succeeds. A separate compiled
addition-reversal control succeeds with byte-identical complete reference output.
That same compiled control also runs the full native adversarial suite against
the baseline, requiring identical successful stdout across its signed-zero,
subnormal, extreme-value, admission, environment and failure-atomicity checks.
Compilation errors, dispatch rejections or crashes do not count as mutant kills.

Two pinned fixture generations were byte-identical. The 59,047-byte fixture contains
30 typed tensors and eight Q0 cases; SHA-256 is
`3cec8eec7574bcc78a0ccc369b631d45e5172e70c06122649201557b794009fa`.
The generator's existing optional-NumPy warning does not affect bytes; NumPy is not
imported or required by this test code. These local results do not substitute for
supported exact-head Ubuntu/LLVM 21.1.8 CI.

The normal focused gate also passed four repeated I1/I2 carrier compositions and
two fresh pinned-Eshkol private AOT builds with identical binaries/output. The
carrier harness dispatches all six L3S schemas, uses genuine scoped I1 target and
I2 floating borrows, checks blocked destruction/copy during borrowing, compares
owned results to raw borrowed-buffer L2 composition, and verifies an invalid mask
preserves all three independent scalar outputs. Every owner and borrow is released.
The AOT bridge invokes this same private transport; it is not installed ownership,
a model graph, or a production aggregate.
