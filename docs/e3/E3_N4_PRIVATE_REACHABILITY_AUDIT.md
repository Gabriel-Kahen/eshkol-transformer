# E3 §9 N4 private reachability audit

Source baseline: `44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef` (tree
`70706c0a25cf2055815de363eec6539823a8fca8`). This is a source audit,
not a compiled N4 result. The direct K1 E3-METRICS cases remain provider
evidence, as specified by the [§9 matrix](E3_SECTION9_COMPILED_ACCEPTANCE_MATRIX.md).

The genuine evaluator takes a sealed M3 model and a D2 dataset. The existing
noninstalled uniform and finite-overflow source candidates use the accepted
`c1-checkpoint-load-module-internal!` API before frame binding,
including byte comparison, parameter shape/tie/identity and initializer checks
(`tests/e3_private/uniform_driver.esk`, `overflow_driver.esk`). Their
development fixtures build checksummed C1 records for all 15 paths with one
tied head/embedding group (`tests/e3_reference/uniform_tie.py`,
`finite_overflow.py`). This route can stage controlled parameter bytes without
a new native parameter setter. It does not by itself prove a requested logit
bit pattern after M3 forward.

| N4 case | Genuine C1/M3/D2 path at this baseline | Next required proof |
|---|---|---|
| Mask `01` | **Unreachable from admitted D2.** D2 writes a positive `true-count` at the row's first destination and leaves later `calloc` mask bytes zero. Packed and unpacked rows therefore emit active prefixes (`11` or `10` for T=2). A parameter change cannot alter the mask. | Retain K1 provider-only `01`. A private `01` admission test would need a separately reviewed D2 test-only staging seam and must be labelled injected, not authentic D2. |
| Masked invalid target | **Unreachable from admitted D1/D2.** D1 validates every shard token `< vocabulary-size`; D2 copies only those tokens and leaves padded targets zero. E3 stage-6 validates **both** targets in `[0,255]` before the mask is consumed (`src/eshkol_transformer/e3_frame.c`). An invalid target in the masked position cannot be made by a C1 parameter checkpoint. | Retain direct-provider rejection. A private malformed-target case needs a reviewed post-D2 staging seam with owner and rollback checks; a post-stage mutation would also have to respect the earlier E3 validation boundary. |
| Masked nonfinite logit | D2 can supply an authentic `10` row, and E3-METRICS checks all 512 logits before applying the mask. The zero-state development oracle predicts finite `+0` logits, while N3K validates finite operands/intermediates during M3 forward (`native/n3k_primitives_provider.c`). A finite C1 state reaching E3-METRICS with a nonfinite masked logit is **not established**. | Keep direct K1 coverage. If exploring a C1 state, freeze an independent M3 forward oracle and the *first* predecessor error; do not call a post-forward corruption an authentic model output. Any post-forward test-only seam would be a separately reviewed admission probe. |
| Unsupported fenv controls; sticky-flag preservation | The external floating environment is not a C1 parameter or D2 datum. E3 executes M3/L2/L3S before the E3-METRICS correct/finalize call; N3K can reject altered masks/rounding/FTZ/DAZ on the first M3 role, and L3S also validates controls. Sticky exception flags are separately permitted and preserved. No closed private evaluator fixture currently owns a timed fenv change at the N4 provider boundary. | Retain direct K1/N3K/L3S provider gates. A private first-error or sticky-preservation witness needs a reviewed test-only environment control around a named preinvoke boundary, with complete restoration and source/compiled proof of which provider is reached. No parameter seam addresses this. |
| Subnormal or negative-zero f32 input | C1 can encode raw finite f32 parameter words through the existing load route, but neither the zero-state nor finite-overflow fixture proves a subnormal or negative-zero *output* logit, CE numerator or cumulative metric operand. M3/L2/L3S operations can round, underflow, or change zero's sign. | Generate candidate C1 states and authentic D1/D2 rows; compare observed M3 logits, CE and metric inputs with independent forward/numerical oracles and exact bits under supported fenv before adding a private N4 assertion. Use the existing C1 load route first. |

The provider source is precise about the difference: `evaluate_correct` rejects
out-of-vocabulary targets and nonfinite logits before mask weighting
(`native/e3_evaluation_metrics_provider.c`), while genuine E3 calls
`e3-stage-batch!`, all M3 forward roles, CE, L3S reduce, then E3 correct
(`native/e3_private_extension.esk`). The direct provider may reach malformed
rows, descriptors and environment states that this traversal does not.

No N4 production or test-only API is added by this audit. The next bounded
compiled gate should first prove candidate finite C1-loaded model logits and
actual D2 row/mask bytes under normal/repeat/poison/sanitizer execution, with
an independent M3 oracle and explicit first-error stage. Only then can an N4
private assertion be selected. The `01` and masked-invalid-target rows remain
provider-only until separate D2 injection contracts are reviewed. N4 and full
§9 acceptance remain **HOLD**.
