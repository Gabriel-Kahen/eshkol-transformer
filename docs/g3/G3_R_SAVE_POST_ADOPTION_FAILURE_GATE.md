# G3-R private SAVE caller: compiled failure gate preparation

This gate is **pending**. The source-only successor roots the returned O2
shell in `shell-cell[0]` before taking the one-use bundle. A compiled Eshkol
run must still prove that a failure after adoption reaches the same shell
through the guard after longjmp. It must use the real adoption, P1, O2 and C2
leaves on one original generator/model/optimizer/capsule graph.

## One-use post-adoption cut

In a test-only build of the exact production caller, insert a typed failure
immediately after `(vector-set! shell-cell 0 (g3r-save-o2-adopt-held-internal
...))` and before `g3r-save-o2-take-for-c2-internal`. Do not replace the
adoption or abandon leaves. The injected failure must run once, after the
adoption return is observed, with a unique error code/message so the first
error can be checked exactly. The test instrumentation may count calls to the
real `g3r-save-o2-abandon-internal!` without changing its behavior.

Assert all of the following in the compiled process:

1. The handler calls the real abandon exactly once with the exact returned
   shell; O2 release precedes P1 release, both complete, and no C2 owner or
   output is published. The caller rethrows the injected first typed error.
2. Original fourteen value/gradient bit images, tied alias, G3 RNG/history,
   seven-field journal, token count, D2 cursor and caller outputs are unchanged.
3. Reusing the consumed admission/one-use grant denies before any second O2
   issue, P1 clone or C2 claim. A fresh valid grant can capture again, proving
   no leaked owner or retained busy state after certain cleanup.
4. A separate test-only abandon failure fails stop (`exit 134`) and retains
   roots; it must not be reported as a recoverable typed error.

For the later partial-claim cut, inject a typed failure into C2's
`c2-training-state-claim-o2!` after its real P1 claim. Verify C2 rolls that
claim back, then the SAVE caller abandons the still-detached O2 and P1 once,
in that order. Check the same original images and one-use denial. Do not infer
this result from a preclaim refusal or source parsing.

Run normal, repeat, and supported sanitizer modes against one pinned union;
record exact commit/tree, compiler/runtime versions, command exit codes,
assertion counts, and a checksummed evidence directory. The source-only tests
in this branch are preparation, not compiled failure-cut evidence.
