# E3 L sanitizer diagnosis and isolated P1 source repair

The immutable one-use L run from `7322632224f08dc123f585f9893519aa389eb6c2`
failed before E3 evaluation. Normal A/B three-batch correctness passed, but
sanitized all-correct initialization raised an ASan heap-buffer-overflow
READ 1 in `numerical_driver_root_lambda_169`. The read was seven bytes before
the 264-byte token allocated by `native/p1_identity.c:create_token` at line 403.
The sealed raw run is
`/home/gabe/.codex/evidence/eshkol-transformer/e3-section9-first-eval-oneuse-73226322-20261012-l/run`
(`SHA256SUMS-RUN` SHA-256
`b5a632d26d2027c391824da9db6e0d0e2888876ea77faac9350fd7d9267a1ec2`).

The sanitized package's generated LLVM function for
`tensor-provider-register-internal!` loads
`tensor-provider-admission-active-shell`, checks its heap tag, forms
`getelementptr i8, ptr %shell, i64 -7`, and reads one byte before storing `#f`
to that global. The original source assigned the native `ptr` from
`shell-for-raw` directly to the global and then cleared it after admission.
That pointer denotes a P1 native token, not an Eshkol heap value with a header
at offset -7. The emitted borrow check therefore explains the exact ASan
address and access width. The isolated source repair keeps the raw shell in
a managed one-element vector while the admission guard is active and reads
its element for the existing same-shell reentrancy comparison. It changes
neither the native token nor the provider admission/publication protocol.

This is a source-level causal repair, not a passing sanitizer result. The next
independent gate must inspect the new compiled guard IR and run the authentic
sanitized P1/E3 package, including same-shell and distinct-shell reentrancy,
before replaying a fresh one-use E3 detector. The original L run remains a
failed prerequisite and supplies no first-evaluation or §9 acceptance.

The host launcher independently reported `unsafe-guest-output`, killed the
Docker client, and returned 74. Its `output_bytes` traversal catches
`OSError` and `ValueError` but records only the generic reason, discarding the
exception class, path, and bounded message. The final tree had 253 ordinary
entries and 6,883,528,604 bytes, below the 20 GiB cap; no special entry was
found in the final read-only scan. The transient traversal failure cannot be
reconstructed from the sealed monitor row. A future one-use launcher should
retain a bounded error code and path at the first exception, without relaxing
its fail-closed behavior. The sealed launcher and L output are unchanged.
