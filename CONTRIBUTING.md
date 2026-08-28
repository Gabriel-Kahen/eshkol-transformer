# Contributing

Development is organized by the workstream IDs in `docs/ROADMAP.md`.

Before implementation:

1. Confirm upstream contracts are merged or explicitly pinned.
2. State shapes, dtypes, device behavior, error behavior, and serialization impact.
3. Add or identify the test oracle.

Before handoff:

1. Run focused unit tests and the relevant integration gates.
2. Add finite-difference or reference parity checks for numerical code.
3. Check deterministic behavior with a fixed seed when randomness is involved.
4. Update documentation and roadmap evidence.
5. Record benchmarks only with hardware, backend, dtype, shapes, and command included.

The main branch must remain buildable. Experimental kernels stay behind explicit
capability checks until their correctness and fallback behavior are verified.

