# Independent full CPU4 numerical reference

`reference.py` uses only Python's standard library. It imports no production
model, descriptor, schedule, provider, PyTorch, NumPy, or environment-specific
runtime. Python is a development oracle; delivered execution remains native.

The fixed profile is CPU f32 parameters, N1/T4/V256/D4/Hq2/Hkv2/Dh2/L1/F8,
learned position `[4,4]`, pre-LayerNorm attention and FFN, final LayerNorm,
epsilon binary32 bits `3727c5ac`, exact-erf GELU, no projection biases/dropout,
and tied token/head weights. Its direct mathematical equations run in
binary64. Four token IDs are `[3,197,3,41]`; the repeated token tests scatter
accumulation, and all positions have distinct learned vectors.

The fixture contains fourteen canonical parameter arrays in path order,
1,192 binary32 words in total. It uses a deterministic arithmetic pattern and
nonzero conditioned norm weights. This is a strict state-load fixture, not a
production seeded initializer. One physical array supplies both embedding and
head edges. The 1,024 upstream binary32 words are nonuniform.

Generate the temporary header and run self-checks:

```sh
python3 tests/cpu4_transport/reference.py --self-check --header /tmp/cpu4-reference.h
python3 tests/cpu4_transport/reference.py --header /tmp/cpu4-reference-repeat.h
cmp /tmp/cpu4-reference.h /tmp/cpu4-reference-repeat.h
```

Generated data must remain outside Git. The header exposes:

| Symbol | Contents |
| --- | --- |
| `cpu4_ref_parameter_offsets[15]` | Canonical array boundaries |
| `cpu4_ref_parameter_bits[1192]` | Exact binary32 parameter words |
| `cpu4_ref_ids[4]` | Copied input token IDs |
| `cpu4_ref_upstream_bits[1024]` | Exact binary32 output cotangent words |
| `cpu4_ref_logits[1024]` | Binary64 token-major forward output |
| `cpu4_ref_gradients[1192]` | Canonical binary64 parameter VJP |
| `cpu4_ref_parameter_paths/ranks/shapes` | Independent tensor metadata |

Every parameter derivative uses central differences of the complete independent
model objective `sum(logits * upstream)`, with step
`1e-5 * max(1, abs(parameter_value))`. Perturbations stay binary64 rather than
being rounded back to binary32. The tied array is perturbed at both uses in
the same forward; position tail rows are fully included. Each perturbation
restores the exact original value in a `finally` block.

`--self-check` verifies all values/gradients are finite, every parameter family
has a nonzero signal, a future-token change preserves earlier causal logits,
and deliberate noncausal and wrong-head-layout mutations change outputs.
Three mixed-parameter directional probes and fourteen sampled derivatives
at a second step check gradient assembly. Four tied token-row probes compare
the sum of separate head and embedding edges; the embedding edge also equals
the sum of separately perturbed repeated-token occurrences.

Suggested native binary32 comparison tolerances, declared before execution,
are `abs_error <= 4e-5 + 4e-5 * abs(reference)` for logits and
`abs_error <= 3e-4 + 5e-4 * abs(reference)` for VJP elements. Corresponding C
macros are included in the header. These envelopes compare f32 native
arithmetic with a binary64 mathematical oracle. Reference self-consistency
checks use tighter tolerances; neither establishes native execution or
transport ownership correctness without the separately compiled gate.

Generation is deterministic for the same Python/platform math implementation;
the header prints hexadecimal binary64 values and exact binary32 words.
`math.erf`, `math.exp`, and `math.sqrt` use the host math implementation, so
cross-platform byte identity is not claimed. The tool prints the generated
header SHA-256 and elapsed time; elapsed time is not included in header bytes.
