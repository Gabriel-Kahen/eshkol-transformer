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

Result: exit 0; 23 tests ran, 22 passed, and one skipped with the explicit
`BLOCKED: compiled Eshkol runner unavailable` reason. Tests cover canonical
bytes, exact generator identity, checksum corruption, valid-checksum version and
boolean-version rejection, malformed dtype/shape/device/keys/roles/tolerances,
duplicate keys, all required case kinds, repeated inputs, NaN/Inf, boundaries,
empty tensors, strict numerical metadata and scalar types, central finite
differences (including unrepresentable perturbations), and Python isolation.

## Compiled Eshkol probe

The probe is genuine Eshkol source at `tests/q0/probes/scalar_add.esk`. The parity
test requires compiler version 1.3.4 or newer, compiles with `-o` in a temporary
directory, executes the emitted binary, requires exactly one scalar output, and
compares it to the frozen PyTorch fixture. It never invokes JIT `-r` mode and has
no Python or scalar substitute.

An older locally cached compiler demonstrates that the operation and probe syntax
are reachable, but is not acceptance evidence because it is below the repository
minimum:

```bash
/usr/bin/bash -c 'set -euo pipefail; old=/home/gabe/.cache/eshkol-r0/work3/build/eshkol-run; temp_dir=$(mktemp -d .tmp/q0-old.XXXXXX); "$old" --help 2>&1 | tail -n 2; "$old" -o "$temp_dir/scalar-add" tests/q0/probes/scalar_add.esk; test -x "$temp_dir/scalar-add"; "$temp_dir/scalar-add"; rm -rf -- "$temp_dir"'
```

Result: exit 0; compiler identified as `v1.1.13-accelerate`, compilation emitted
an executable, and the executable printed `3.75`. The Q0 parity test explicitly
rejects this version as below 1.3.4.

The available compatible compiler identifies as `v1.3.4-evolve`, parses and
code-generates the probe, and emits a 42,968-byte temporary object. Its final link
is blocked by the staged compiler's absolute host-library paths:

```bash
/usr/bin/bash -c 'runner=/tmp/eshkol-a0-build.z9rbhM/release/eshkol-v1.3.4-evolve-linux-x64-lite/bin/eshkol-run; libs=/tmp/eshkol-a0-build.z9rbhM/llvm21-deb/usr/lib/x86_64-linux-gnu:/tmp/eshkol-a0-build.z9rbhM/blas-root/usr/lib:/tmp/eshkol-a0-build.z9rbhM/ubuntu-deps/usr/lib/x86_64-linux-gnu; ESHKOL_RUN="$runner" LD_LIBRARY_PATH="$libs" LD_PRELOAD=/tmp/eshkol-a0-build.z9rbhM/blas-root/usr/lib/libcblas.so.3 .tmp/q0-venv/bin/python -m unittest tests.q0.test_scalar_add_parity.CompiledEshkolParityTests.test_scalar_add_matches_frozen_pytorch_oracle -v'
```

Result: exit 1 with explicit `BLOCKED: compatible Eshkol compilation failed`.
The linker reports:

```text
/usr/bin/ld: cannot find /usr/lib/x86_64-linux-gnu/libblas.so
/usr/bin/ld: cannot find /usr/lib/x86_64-linux-gnu/libcrypto.so
ERROR: Linking failed with exit code 1
```

This is an F0/A0 compiler packaging/link-path dependency. Q0 does not hide it,
patch host libraries, or accept the older compiler as parity evidence.

## Deferred acceptance rerun

After F0/A0 supplies a compatible runner whose native link paths resolve:

```bash
/usr/bin/bash -c 'ESHKOL_RUN=/absolute/path/to/eshkol-run scripts/test-q0.sh'
```

Acceptance requires all 23 tests to pass with no skip or blocked failure.
