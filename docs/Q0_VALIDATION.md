# Q0 validation evidence

Validation date: 2026-08-28.

## Environment

- Python: 3.14.6.
- Development-only reference framework: PyTorch 2.13.0+cpu in
  `.tmp/q0-venv`.
- Fixture SHA-256:
  `a56cf559536b1fa7be0d7959abd168f352b66a2990f47dd4a5883d55decff217`.

## Deterministic generation

```bash
/usr/bin/bash -c 'set -euo pipefail; mkdir -p .tmp/q0-generation; .tmp/q0-venv/bin/python tests/q0/generate_scalar_add.py --output .tmp/q0-generation/first.json; .tmp/q0-venv/bin/python tests/q0/generate_scalar_add.py --output .tmp/q0-generation/second.json; cmp --silent .tmp/q0-generation/first.json .tmp/q0-generation/second.json; cmp --silent .tmp/q0-generation/first.json tests/q0/fixtures/scalar_add_v1.json; sha256sum .tmp/q0-generation/first.json .tmp/q0-generation/second.json tests/q0/fixtures/scalar_add_v1.json'
```

Result: exit 0. Both fresh generations and the checked fixture were byte
identical. All three files had the SHA-256 above. PyTorch emits a development-only
warning that optional NumPy is not installed; the generator does not import or use
NumPy and the warning does not affect the deterministic bytes.

## Focused test suite

```bash
/usr/bin/bash -c '.tmp/q0-venv/bin/python -m py_compile tests/q0/*.py; .tmp/q0-venv/bin/python -m unittest discover -s tests/q0 -p "test_*.py" -v'
```

Result: exit 0; 23 tests passed. Tests cover canonical
bytes, exact generator identity, checksum corruption, valid-checksum version and
boolean-version rejection, malformed dtype/shape/device/keys/roles/tolerances,
duplicate keys, all required case kinds, repeated inputs, NaN/Inf, boundaries,
empty tensors, strict numerical metadata and scalar types, central finite
differences (including unrepresentable perturbations), and Python isolation.

## Compiled Eshkol probe

The probe is genuine Eshkol source at `tests/q0/probes/scalar_add.esk`. The parity
test requires compiler version 1.3.4 or newer, compiles with `--no-stdlib -o` in a
temporary directory, executes the emitted binary, requires exactly one scalar
output, and compares it to the frozen PyTorch fixture. Compile and run calls have
explicit timeouts. It never invokes JIT `-r` mode and has no Python or scalar
substitute.

The accepted local evidence uses the canonical F0 source pin
`https://github.com/tsotchke/eshkol.git` at
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`. The compiler identifies as
`Eshkol Compiler v1.3.4-evolve`. This compiler was built on the explicitly
unsupported CachyOS/LLVM 22 compatibility lane, so the result proves Q0 parity but
does not replace F0 supported-host CI evidence:

```bash
TMPDIR=/home/gabe/.cache/eshkol-q0-final2 \
ESHKOL_RUN=/absolute/path/to/f0/.deps/eshkol-build-minimal/eshkol-run \
/usr/bin/bash scripts/test-q0.sh
```

Result: exit 0; all 23 tests passed. The compiled Eshkol program emitted exactly
`3.75`, matching the frozen PyTorch 2.13.0+cpu float64 fixture.
