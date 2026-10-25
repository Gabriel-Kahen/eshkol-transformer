# E3 §9 N4 sparse subnormal candidate

Baseline `44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef`, tree
`70706c0a25cf2055815de363eec6539823a8fca8`. This is a noninstalled
source and host oracle candidate, not compiled E3 K1 reachability or N4
admission.

`tests/e3_reference/n4_sparse_bits.py` builds a checksummed C1 record for all
15 logical paths, with the accepted tied head/token embedding alias (10, 14).
The tied row for token 3 is `(1,-1,1,-1)`; row 4 has f32 word `0x00000001`
at dimension 0 and zeros elsewhere. Position embedding row 1 is
`(1,-1,1,-1)`; final norm gamma is 1; all projections and other parameters
are zero. This uses the existing `c1-checkpoint-load-module-internal!` route
before E3 frame binding. The admitted D1 bytes `03 04` and D2 configuration
used by the finite-overflow fixture emit input `(3,0)`, target `(4,0)`, mask
`10`.

The independent sparse graph oracle computes two nonzero model input rows:
token 3 supplies the first axis and position 1 supplies the second. Zero Q/K/V,
attention output and FFN projections leave both residual rows equal to the
axis. The final norm feeds the tied head matmul. Under RN-even binary32 with
FTZ and DAZ clear, it predicts the 512 logit words as `0x407fffac` at indices
3 and 259, `0x00000001` at indices 4 and 260, and positive zero elsewhere.
Index 260 is in the masked D2 position. The oracle rounds sparse f32
arithmetic separately from the M3 schedule/provider; runtime agreement
remains to be observed. Host tests check C1 payload/alias, genuine D1/D2 row
and these independent bits.

## Compiled gate to discharge this row

The one-use noninstalled source gate extends the existing overflow private
tuple with a `sparse` scenario. The accepted C1 load/byte/owner checks and
existing D2 path remain in that tuple. Its compiled observer captures the D2
row, all 21 M3 roles including every word of `Z`, CE, K1-derived batch/state
and final output. A test-only `ET_E3_N4_SPARSE_TESTING` observer records all
512 K1 input words and x87/MXCSR controls immediately before dispatch. The
parser compares both K1 input and retained `Z` to the complete independent
oracle and checks successful K1/accumulate output. `et_e3_private_correct_v1`
binds its K1 input directly from `frame->forward[20]`; the numerical observer
reads that same retained handle after K1. Source and result review must verify
this identity and lack of mutation between those points. The fixture writes no
post-forward logit or row bytes. A first M3/N3K rejection invalidates a K1
claim; captured failure logs must identify the predecessor, not be counted as
an N4 pass.

The compiled run must verify RN-even, FTZ=0, DAZ=0 and supported exception
masks at runtime; sticky flags are allowed. The existing N3K/N2/K1 provider
checks enforce their admitted controls. A changed environment or failed first
role invalidates this candidate. The external launcher accepts the exact
commit and tree plus a fresh evidence directory and takes the shared lock
without waiting; it must run only after the other scheduled lock turns.

Run the reviewed gate on the supported fe9/f31 Clang 21 lane with the shared
`heavy.lock`: normal, repeat, poison and ASan/UBSan/LSan. Seal exact source,
compiler/runtime identities, fixture hash, stdout/stderr, exit status and
resource checks. The launcher and gate are prepared but **NOT_RUN**;
independently review source, then the result before updating the N4 private
row. No Docker gate is requested here.

Negative-zero logit is a separate case. The final N3K tied-head linear role
starts each sum at positive zero and has no bias. With admitted RN-even,
`+0 + -0` and exact cancellation yield positive zero; a negative product
that underflows to `-0` cannot turn that initial sum negative. This source
obstacle does not discharge other negative-zero E3 operands. A finite C1
state also cannot manufacture a nonfinite logit at E3 K1 if an earlier N3K
role rejects its nonfinite input, prospective intermediate or output.
Timed fenv changes require their own boundary control and first-error proof.
Those rows stay provider-only/open.
