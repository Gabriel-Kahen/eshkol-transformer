# Exact proposed E3 BOOL metric provider contract

Design-only proposal for root acceptance. Base inspected: merged `27c99f7c9f27a227f0e237f1571ac89d5cdfa225`. This fixes concrete choices for a later bounded implementation; it does not authorize production edits, publish VERIFIED evidence, change A0/Eshkol APIs, or modify accepted K1/L2/L3S/I1/I2. The three operations are correctness with BOOL masks, accumulation and finalization only.

## 1. Files, ABI and discovery

| Item | Exact proposed choice |
|---|---|
| Independent ABI | 1.0 |
| Header | `include/eshkol_transformer/e3_evaluation_metrics_abi.h` |
| Source / single object member | `native/e3_evaluation_metrics_provider.c` / `e3_evaluation_metrics_provider.o` |
| Native archive | `build/e3-metrics/libeshkol_transformer_e3_metrics.a` |
| Sole exported symbol | `const et_kernel_provider_v1 *et_e3_metrics_kernel_provider_v1(void);` |
| Header macros | `ET_E3_EVALUATION_METRICS_ABI_MAJOR 1u`, `ET_E3_EVALUATION_METRICS_ABI_MINOR 0u` |
| Header guard | `ESHKOL_TRANSFORMER_E3_EVALUATION_METRICS_ABI_H` |
| Provider name and capability implementation | `e3.cpu-f32-bool.serial` |
| Provider and capability version | `1.0` |
| Provider and capability evidence | `E3:cpu-f32-bool-metrics-v1` |
| Capability | `e3.evaluation-metrics` |
| Operations, sorted exact order | `e3.evaluation-metrics.accumulate`, `e3.evaluation-metrics.correct.bool`, `e3.evaluation-metrics.finalize` |
| K1 provider prefix | ABI1.0; required_features0; one capability, sizeof capability stride/bytes |
| Capability request admission | compute `f32`, device `cpu`, rank3, exact extents `[1,2,256]`; deterministic1; both valid deterministic request booleans eligible |

The immutable descriptor/accessor has process lifetime. The eventual tested provider advertises VERIFIED only for these rows/profile; this proposal itself supplies no verification. Explicit resolver access only; do not define `eshkol_transformer_kernel_provider_v1`, search/load providers, add a global registry, or edit K1 baseline/K2 discovery. No bool/f32 storage owner, Eshkol facade, graph/backward, trainer, model implementation, serialized format or public evaluator is added. Each composed L2/L3S/E3 provider uses its own explicitly resolved K1 runtime. The aggregate later localizes the E3 accessor as it does the other providers.

Header includes kernel_abi.h and wraps the single accessor in extern-C for C++. No new public structs, functions or error enums. Build links K1 and libm explicitly, not L2/L3S for the standalone provider; those are test/composition dependencies. Proposed build driver `scripts/build-e3-metrics.sh`, focused test driver `scripts/test-e3-metrics.sh`, and test tree `tests/e3_metrics/`. No implementation of these paths exists or is authorized by writing this design.

## 2. Exact call schemas and storage

All three rows retain request `[1,2,256]`; scalar operands remain rank0, not request-shaped. Exact ordered schemas:

| Operation suffix | Inputs (count) | Outputs (count) |
|---|---|---|
| correct.bool | logits f32[1,2,256]; targets i64[1,2]; mask bool[1,2] (3) | C_b f32[]; active i64[] (2) |
| accumulate | N f32[]; W f32[]; C f32[]; tokens i64[]; batches i64[]; N_b f32[]; W_b f32[]; C_b f32[]; active i64[] (9) | next_N f32[]; next_W f32[]; next_C f32[]; next_tokens i64[]; next_batches i64[] (5) |
| finalize | N f32[]; W f32[]; C f32[] (3) | loss f32[]; mask_weight f32[]; perplexity f32[]; token_accuracy f32[] (4) |

Every view: CPU, dense row-major, offset0, naturally aligned, exact byte_length. Correctness inputs have2048/16/2 bytes and alignments4/8/1. f32[] has4 bytes/alignment4; i64[] has8 bytes/alignment8. Rank0 requires shape=NULL. Rank3/2 shapes match exactly; no cast, broadcast, materialization, stride or offset variant. Data pointer span arithmetic must not wrap; all nonzero spans require readable/live storage and output spans writable. Native readable-pointer preconditions are not turned into a sandbox guarantee.

All input data spans pairwise disjoint. Outputs pairwise disjoint and disjoint from every input. Every output additionally excludes these entire live metadata spans: call `struct_size`; request `struct_size`; input/output table declared byte spans including compatible-minor extensions; request/each-view rank*sizeof(uint64_t) shape arrays; call capability and request operation/dtype/device strings and each-view dtype/device string through terminating NUL. Adjacent spans are valid. No in-place accumulate; fixed disjoint ping-pong state is required.

Error storage remains disjoint caller-owned control memory under K1's existing trusted precondition: K1 writes it before provider validation, so error/data alias cannot be promised recoverably rejected. Borrowed inputs, metadata and FP controls must remain immutable through complete validate+invoke. Neither phase retains pointers or allocates. Success overwrites every output in table order after all prospective results have been validated. Any recoverable error leaves all data/descriptor/guard bytes unchanged; only the independent error record changes. Native process/hardware faults and racing mutation remain outside the rule.

## 3. Numerical environment and side effects

Require x86-64, CHAR_BIT8, sizeof(float)4, IEEE radix2/significand24/maxexp128, FLT_EVAL_METHOD0, exact int64 storage. Compile C11 with `-ffp-contract=off -fexcess-precision=standard -fno-fast-math -frounding-math`, `#pragma STDC FENV_ACCESS ON` and FP_CONTRACT OFF, warning-clean optimized O2 and sanitized O1 variants as L3S does. No binary64 arithmetic, excess precision, approximate reciprocal/exp, FMA or reassociation.

Reuse L3S's admitted x87/MXCSR controls exactly: x87 exception masks bits0..5 all set and rounding bits10..11 zero; MXCSR exception masks0x1f80 all set, rounding/FTZ/DAZ bits selected by0xe040 zero. Do not repair controls. x87 precision-control bits are not newly constrained because actual arithmetic uses the accepted binary32/SSE lane. Save complete validation fenv before any float classification/arithmetic; fegetenv failure is unsupported. Preserve independent incoming x87/MXCSR sticky flags and controls on validation success/error using fesetenv. Failed environment restoration is internal error; no tensor output has yet changed, but successful environment restoration cannot then be claimed.

Save/restore `errno` across both validation and invocation, including expf, so rejected overflow does not leak ERANGE. This is a bounded E3-native side-effect contract, not an amendment to L2. Invocation re-evaluates the same fixed operations into stack results and then commits, with inputs/controls unchanged. It may accrue ordinary fenv arithmetic flags just as L3S does; it does not repair/clear sticky flags or return recoverable errors. No output write occurs before all arithmetic results exist. Internal violation of the validated invocation preconditions is outside recoverable dispatch; no partial-output rollback is claimed.

Finite nonnegative scalar zero values accept +/-0 and canonicalize locally/output to+0. Positive subnormal N/N_b and a correctly rounded division underflowing to+0 are valid. Bool weights/counts cannot be subnormal, and nonzero accuracy is at least1/16384. No rejection solely for underflow/inexact sticky flags. Perplexity is expf(nonnegative loss), hence>=1 when finite; it cannot underflow.

## 4. Exact validation and arithmetic order

K1 generic call/table/request/view/output-alias validation and capability matching run first and retain their exact current precedence. Provider validation then follows this fixed order:

1. Save errno/fenv. Validate call/request prefix/null defensively under the direct-callback readable-descriptor precondition, then recognize capability/request row/profile.
2. Exact operand counts; inputs in index order, then outputs in index order. Per-view check order: dtype, device, layout/offset, exact rank/shape including rank0 NULL, exact data span/alignment.
3. Input aliases with ascending pair `(i,j)`; then each output's metadata exclusions in the list order in section2.
4. FP-control compatibility, before floating classification.
5. Row-specific domains/order below, followed by every prospective result; no output mutation.
6. Clear error on success, restore complete fenv and errno, return category. Restoration failure takes precedence over an otherwise pending result and reports internal/provider-rejected.

### correct.bool

Domain order: validate both target indices position0 then1 in[0,256); scan all logits position0/j0..255 then position1/j0..255 for finiteness; validate mask bytes0 then1 are0/1; reject00. No masked position skips any target/logit validation.

For positions0 then1, initialize winner0 and maximum=logit[0]; visit j1..255, replace only on strict `logit[j] > maximum`. Thus equal maxima and +/-0 ties select lowest vocabulary index. Convert canonical mask and `(winner==target)` to exact f32 values0/1. Compute each separate `p_i=fl(mask_i*hit_i)`, then `C_b=fl(fl(+0+p0)+p1)`; active is exact i64 mask0+mask1. Publish C_b then active. C_b is integral0..active; active is1 or2. No softmax, epsilon/tolerance or gradient.

### accumulate

Domain order:

1. Check floats in input table order N,W,C,N_b,W_b,C_b: finite/nonnegative; canonicalize zeros locally. Then counters in input order tokens,batches,active: nonnegative; require active1 or2.
2. Profile bounds: reject batches>8192 or tokens>16384 as unsupported **before** float conversion or cross-field equality. This fixed precedence also applies to a cap-exceeding state containing a later cross-field defect.
3. Cross-field checks in this order: batches<=tokens; tokens-batches<=batches (overflow-safe form of tokens<=2*batches); W==exact-f32(tokens); C<=W and integral; batches0 implies N0; W_b==exact-f32(active); C_b<=W_b and integral. Any failure invalid-argument. Integral tests convert only proven bounded nonnegative values to i64 and back exactly, or an equivalent reviewed integer-bit test; do not invoke f64.
4. Check i64 next_tokens=tokens+active and next_batches=batches+1 without overflow. In the admitted bounded domain overflow is unreachable; the checked path still reports invalid-argument/integer-overflow if triggered. Reject next_batches>8192 or next_tokens>16384 as unsupported before float arithmetic. Thus a valid full8192-batch state is admissible for finalize but cannot accumulate another batch.
5. Compute next_N=fl(N+N_b), check finite; next_W=fl(W+W_b), check finite; next_C=fl(C+C_b), check finite. The last two are exact under the bool bounds. Publish next_N,next_W,next_C,next_tokens,next_batches in table order.

The empty state is exactly N=W=C=tokens=batches=0 after local zero canonicalization. N_b is any finite nonnegative row numerator; the closed Eshkol transcript, not raw scalar values, proves it came from accepted L3S. No batch mean, extra multiply or division appears in this row.

### finalize

Check N,W,C in input order for finite/nonnegative. Reject W==0 invalid-argument. Reject W>16384 unsupported before integrality; then require W integral, C<=W and C integral or return invalid-argument. Compute loss=fl(N/W), check finite; accuracy=fl(C/W), check finite; perplexity=expf(loss), check finite. Canonicalize zero outputs. Stage and commit in output-table order loss,W,perplexity,accuracy (arithmetic order deliberately differs from output order). Exact N0 returns loss+0/ppl1. No clamp, infinity sentinel, f64 retry or approximate `loss<=logf(FLT_MAX)` overflow cutoff.

The raw row's W0 error is invalid-argument; Eshkol's empty/entry-EOS traversal raises invalid-state before calling it. With only3 inputs, finalize cannot establish counter/history authenticity. The trusted frame proves prior successful accumulator transitions and prevents fabricated orchestration.

## 5. Exact error category/code assignment

Provider errors carry full recognized operation string; early unrecognized/prefix/fenv-bookkeeping failures carry `e3.metrics.provider`. Messages are bounded nonempty fixed English literals; neither message text nor index interpolation is a format/API. Category/code and precedence below are contractual after acceptance. Names abbreviate the existing ET_KERNEL_ERROR_/ET_KERNEL_CODE_ constants; introduce no new enum.

| Rejection | Category / code |
|---|---|
| Provider call/request NULL (direct-callback defensive case) | INVALID_ARGUMENT / NULL_ARGUMENT |
| Provider call/request truncated prefix | VERSION_MISMATCH / INVALID_STRUCT_SIZE |
| Provider unknown operation/capability or wrong request profile/dtype/device | UNSUPPORTED / PROVIDER_REJECTED; normal K1 matching may reject earlier with its existing CAPABILITY_NOT_VERIFIED |
| Exact operand count mismatch | INVALID_ARGUMENT / INVALID_BUFFER |
| Operand dtype mismatch | DTYPE_MISMATCH / INVALID_TEXT |
| Operand device mismatch | DEVICE_MISMATCH / INVALID_TEXT |
| Layout not dense or offset nonzero | NONCONTIGUOUS / INVALID_BUFFER |
| Exact rank/shape, rank0 shape pointer or target index mismatch | SHAPE_MISMATCH / INVALID_SHAPE |
| Provider data alignment/null-span/range/exact-span error | INVALID_ARGUMENT / INVALID_BUFFER; generic K1 byte-length/shape errors retain their earlier categories |
| Pairwise input alias or output-live-metadata alias | INVALID_ARGUMENT / ALIASING_OUTPUT |
| Unsupported x87/MXCSR controls or fegetenv failure | UNSUPPORTED / PROVIDER_REJECTED |
| Negative/nonfinite scalar/logit, bad bool/allzero mask, active not1/2, negative counter, fractional count-weight/C, cross-field mismatch, W0 | INVALID_ARGUMENT / PROVIDER_REJECTED |
| Existing/next counter or final W beyond the fixed cap | UNSUPPORTED / PROVIDER_REJECTED with precedence in section4 |
| Checked i64 arithmetic overflow (unreachable after admitted caps) | INVALID_ARGUMENT / INTEGER_OVERFLOW |
| Nonfinite prospective sum/division/expf | INVALID_ARGUMENT / PROVIDER_REJECTED |
| fesetenv restoration failure | INTERNAL / PROVIDER_REJECTED |

K1's generic malformed table/descriptor/request/enum/text/byte-length and input/output or output/output alias errors occur before this provider and are unchanged. Tests must assert generic-first results for deliberately multiply-invalid cases rather than rewriting K1 precedence. Direct callbacks require all complete generic readable-metadata preconditions and are not a separate unsupported caller interface.

## 6. Focused independent gate for later implementation

Proposed `scripts/test-e3-metrics.sh`, with exact canonical pin verified by common.sh and explicit local compatibility override only when applicable. It builds/reuses K1/I1/I2/L2/L3S once plus this single provider; no M3 aggregate/fullCI/duplicate toolchain build or Eshkol public evaluator is required to prove this numerical component. Root integrates the focused command into the existing prerequisite planner/CI topology separately after implementation approval.

1. **ABI and isolation:** warning-clean C11/C++17 header consumers; exact single accessor/global/archive-member allowlists; provider-free baseline unchanged; explicit one-capability discovery in C and C.UTF-8; exact3 operations/profile/dtype/device/determinism metadata; no canonical provider symbol, allocator/dlopen/Python/test-bridge dependency. Compile recipes and raw depfiles fixed. Strict optimized LLVM IR/object excludes binary64 arithmetic/FMA; structural operand/traversal/rounding anchors remain.
2. **Independent reference:** `tests/e3_metrics/reference_probe.py` uses exact rational binary32 RN-even arithmetic independently of production and explicitly models each accumulation/division rounding point. A pinned Python/PyTorch development generator creates small deterministic BOOL-only logits/targets for at least3 distinct batches, all01/10/11 masks, ties and padded last row; preserve version/checksum/generator/toolchain source identity via the existing Q0 conventions. Regenerate twice byte-identically. CE compares against the existing accepted L2 tolerance (atol2e-6,rtol2e-5); N/W/C aggregation is bit-exact against the rational model given observed f32 L2 CE. Mathematical frozen CE/loss reference remains separate from that exact-order check.
3. **Independent exp oracle:** use development-only high-precision Decimal.exp on exact binary32 loss input, then independent rational-to-binary32 rounding. For finite results require <=2 binary32 ULP versus this oracle; boundary/domain finite-vs-overflow uses oracle round-to-f32 finiteness. Freeze below/at/above overflow-neighbor cases; never determine the expected boundary by querying the provider or host expf. PPL0loss exactly1. No across-platform bitwise-libm claim. Supported lane measurement exceeding2ULP or wrong finite/overflow classification blocks acceptance rather than loosening tolerance silently.
4. **Real composition:** independent native runner dispatches real L2 forward + L3S reduce.bool + new correct.bool/accumulate/finalize over multiple BOOL batches through separate explicit runtimes. Genuine I1 targets and I2 logits/CE/scalars borrowed synchronously, canonical native bool staging and exact i64 output controls; release all owners/scopes. Numerical provider test storage is clearly private, not a production carrier API. No backward/gradient test is required for new nondifferentiable rows; predecessor numerical-gradient gates remain their own accepted evidence.
5. **Values/bounds:** allcorrect/allwrong, lowest-index ties at multiple positions and +/-0, both target boundaries, all3 valid masks and invalid00/byte2/255, masked invalid target/nonfinite logit, signedzero/subnormal logits and N, exact cap and one-over, fractional W/C, mismatched W/tokens/active, b/t range defects, malformed initial N, real L3S sum overflow, cumulative N overflow, finite huge N with final exp overflow, underflowing N/W and exact-zero outputs. The rational order witness N_b=[64,2^-18,2^-18], W_b1 gives total0x42800000 vs regrouped0x42800001 and loss0x41aaaaab vs0x41aaaaac with finite PPL. Mean-of-means witness(1,1),(6,2) gives7/3 versus2.
6. **Atomic adversarial coverage:** every schema/dtype/device/layout/alignment/span/metadata-alias case, adjacent spans, compatible-minor descriptor extensions; output and descriptor sentinels unchanged on early/late failure. Test validation with x87-only/MXCSR-only and mixed preexisting flags, RN alternatives, FTZ/DAZ and unmasked exceptions, signaling NaNs, exact errno restoration; never induce unsafe pointers or uncontrolled traps. Test multiply-invalid cases to lock section4 precedence. No recoverable failure injection inside validated invoke. Normal and ASan/UBSan/LSan with leak detection enabled.
7. **Compiled meaningful mutants:** mutate actual provider source, recompile and require baseline success. Provider-level value kills include highest-index tie, ignore/flipped mask, target offset/wrong axis, active=2 padding count, omit/double N_b, incorrect next counter, extra denominator normalization, wrong output slot and clamp-to-finite on exp overflow. Bad-masked-value admission skip, omitted finite checks and early-output writes are separately labeled admission/byte-atomicity mutants. Numeric mutants require meaningful numeric assertion failure, not compilation failure/crash/unsupported dispatch. A separate composed-driver/orchestration mutation layer uses actual providers but changes call order/number, mean-of-means, exp-before-global-loss or retaining a wider accumulator across batches; the three-batch witness belongs to that layer. A standalone accumulate call contains only one addition per destination and returns physical f32, so a local f64-add-then-cast need not be numerically distinguishable; no fabricated kill is required. W-to-exact-f32(tokens),2-term reversal and FMA of canonical0/1 weighting are equivalent controls. Strict IR/source checks still forbid binary64/FMA. A f64 retry mutant must actually alter a permitted observable to count; an identical-error retry is equivalent, not a kill. No weighted-mask or correct.f32 mutant/fixture is required.
8. **Private compiled transport:** two small fresh pinned-Eshkol AOT numerical callers invoking an inventoried test-only C bridge exercise real provider outputs, errors and byte-atomicity with genuine I1/I2 borrows, compare binary/stdout where fixed-path toolchain permits. Bridge is absent from normal provider archive and negative-link test proves absence. This establishes private numerical compiled reachability only; real model/D2 no-grad/transaction/publication/retention and eventual public execution remain separate E3 composition gates.

## 7. Boundaries and review disposition

No GPU, alternative dtype/device, f32 mask row, arbitrary shape, strided/offset input, zero-weight success, approximate fallback, graph, generic owner, hidden f64 or public Eshkol operation. Native kernels own hot numeric traversal, Eshkol owns orchestration. Provider dispatch is stack-only; exact stack result maxima are3 floats+2 i64 for accumulate,4 floats finalize,1 float+1 i64 correct, apart from scalar locals. Any additional scratch must remain bounded and documented, never proportional allocation. No retention or end-to-end performance claim follows from this provider contract.

**Recommended for independent review and root acceptance as concrete ABI1.0 design.** After acceptance, the exact specified files/strings/rows/domains/error ordering/fenv/atomicity and focused gate permit bounded numerical implementation. This writing itself is not that acceptance. No build, test suite, production edit or new probe ran in this phase; only merged K1/L3S/source and existing focused gate/reference documents were inspected.
