# Independent representative rerun

An independent agent used `/usr/bin/bash` and fresh scratch directories under
`/home/gabe/.cache/r0-independent.XjmRuW` to validate the same clean canonical
source, F0 compiler version, and compiler SHA-256. It ran the checked-in harness
separately for `tensor_core`, `precision_f64`, `matmul`, `broadcast`,
`autodiff_forward`, `autodiff_attention`, and `rng`, then directly reran the
index-OOB, reshape-mismatch, and matmul-mismatch malformed cases with the same
timeouts and memory caps.

Results:

- Tensor core, the narrow precision discriminator, 2-D matmul, the one broadcast
  shape, and scalar forward AD passed repeated AOT/JIT execution.
- Attention AD printed `FAIL attention gradient` in all four executions.
- Fixed-seed RNG repeated within each backend but AOT and JIT sequences differed.
- Index OOB and matmul mismatch rejected explicitly with exit 1.
- Reshape mismatch exited 0 and returned `#((1 2) (3 1e-323))`.
- Available full-run stdout matched the independent stdout byte-for-byte for tensor
  core, broadcast, scalar AD, and attention AD.

The reviewer required these claim boundaries: the f64 discriminator is not proof
of physical storage; Scheme-vector mutation is not tensor mutation; broadcast and
matmul support are limited to tested shapes; scalar differentiation is not proof
of reverse mode; determinism/parity claims cover stdout, not compiler diagnostics;
and malformed-input conclusions remain per-case.

The review's initial malformed preflight omitted `ESHKOL_PATH` and was explicitly
renamed `INVALID_missing_ESHKOL_PATH_not_evidence`; it is not used here.
