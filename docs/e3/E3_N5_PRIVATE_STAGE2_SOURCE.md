# Private E3 N5 stage-2 source candidate

This test-only successor uses the existing noninstalled retention bridge and
authentic E3 frame, P1 token, D2 dataset, model and four I2 destinations. It
changes no production ABI. The caller supplies the established packed-single
three-batch corpus and the separately generated 8,193-row cap corpus. The
driver first completes a real three-batch evaluation, establishing four exact
destination words and two counters. It then sets the authentic 17-mode P1
token to the fixed mixed mask and checks both absent and seeded nonzero-present
gradient variants across all 14 parameter values, gradients and metadata.

The cap dataset's private D2 core must report exactly 8,193 packed rows at
cursor zero. After open, a one-shot D2 read-fault hook is armed: if the frame
reaches its first batch read, the result would be `io`. The witness instead
requires typed `unsupported` at stage **2**, then checks the original six
caller outputs, D2 cursor, 17 modes, full model/gradient image, initializer
and live native counters. It repeats the denial on the same frame and dataset
without the hook and makes the same checks. Source ordering puts the one-over
guard before D2 prepare publication, native acquire, mode enter, and
traversal. The public N5 preflight cannot inspect these private fields.

Offline tests check that ordering and reject removal of the read-fault hook,
an off-by-one D2 row assertion, wrong stage, and dropped model/output
comparisons. The exact derived first-evaluation mutant is regenerated from
the changed retention driver, and its and overflow's retention-source SHA
anchors are repinned only to this source. Earlier compiled retention,
first-evaluation mutant and overflow results remain evidence for their own
original trees; all three compiled package closures and their exact manifests
need fresh validation on this tree. No AOT or runtime result is claimed here.

This is only the **8,193-row private stage-2 denial slice** of N5. The public
cap gate is also unrun. Empty/singleton/EOS/one-token cases, exact 8,192,
normal/poison/sanitizer runs, resource measurements and full owner/lifetime
closure remain open. A fresh, resource-gated private package and compiled
normal/poison/sanitizer gate on the reviewed exact tree is the next proof;
none should be inferred from source checks.
