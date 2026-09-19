# M3 development reference

This is an independent mathematical reference for the accepted fixed
N1/T2/V256/D4/Hq2/Hkv2/Dh2/L1/F8 CPU-f32 diagnostic model. It imports no native
role table or production model schedule. Direct PyTorch equations define the
complete logits graph; the separate N3K integer Philox oracle supplies exact
initializer bits in the explicitly written canonical model path order. Its
compact `reference_manifest.json` freezes the dedicated development-reference
format v1, profile/cases/shapes, source/dependency identities and generated JSON
SHA-256 `fd53baf6c42e893b71a747e59d05cfc976aa1f5f1c2b7713b1f4bfcaecad8267`.
Independent reference review regenerated those bytes and checked the identities;
this is separate from Q0 registration, native VJP and supported-platform evidence.

## Pinned environment and commands

Run from the repository root using `/usr/bin/bash` without login startup files.
The required versions are **Python 3.14.6** and **PyTorch 2.13.0+cpu**. Dependencies
come from the existing development-only `tests/q0/requirements-oracle.lock`;
NumPy is not used. Prepare a local environment if the pinned oracle is unavailable:

```bash
python3.14 -c 'import sys; assert sys.version_info[:3] == (3, 14, 6), sys.version'
python3.14 -m venv .tmp/m3-oracle
.tmp/m3-oracle/bin/python -m pip install -r tests/q0/requirements-oracle.lock
M3_REFERENCE_PYTHON="$(pwd)/.tmp/m3-oracle/bin/python"
```

An existing environment installed from that lock may be selected by setting
`M3_REFERENCE_PYTHON` to its absolute interpreter path. Reference import rejects
other Python/PyTorch versions and requires both reference-only CPU controls below
before torch import. There is no fallback framework, version or device.

```bash
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$M3_REFERENCE_PYTHON" -m unittest tests.m3.test_reference tests.m3.test_public_parity tests.m3.test_native_retention -v
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$M3_REFERENCE_PYTHON" -m tests.m3.reference --output build/m3-reference/development-a.json
ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE "$M3_REFERENCE_PYTHON" -m tests.m3.reference --output build/m3-reference/development-b.json
cmp build/m3-reference/development-a.json build/m3-reference/development-b.json
```

The composed gate selects the same interpreter through `M3_ORACLE_PYTHON` (or
`N2_ORACLE_PYTHON`) and sets both CPU controls before invoking it:

```bash
M3_ORACLE_PYTHON="$M3_REFERENCE_PYTHON" /usr/bin/bash scripts/test-m3-reference.sh
```

Generation prints the exact output SHA-256. Records include source hashes for the
graph generator, independent initializer and dependency lock, plus Python/PyTorch
versions and both CPU controls. Generated records remain under ignored `build/`;
do not commit them. Ordinary generation rejects a mismatch with the checked-in
manifest before writing output. A deliberate reference change uses
`--candidate-manifest PATH` and requires renewed independent review; ordinary
tests never rewrite the frozen manifest.
The initial development run used the existing interpreter
`/home/gabe/.codex/worktrees/552e/eshkol-transformer/.tmp/o2-venv/bin/python`.
That local path is evidence provenance, not a repository dependency.

## What the tests establish

The seven numerical tests cover canonical 14-parameter construction and
290-block continuation, three initializer seeds, distinguishable/repeated tokens,
all unique VJPs, separate tied head/embedding edges and lost/duplicated-edge
mutations, causal isolation, q/k/layout/noncausal wiring mutations, and zero-seed
VJPs. They also reject malformed parameters, extra/missing paths, scalar or
wrong-shape upstream, nonfinite values, mismatched dtypes and invalid tied edges.
The reference does not silently broadcast or cast these operands.

Two additional result-decoder tests cover public numerical output.
They feed synthetic f32-rounded observations to the comparator and reject missing
or duplicate rows, zeroed Q/K gradients, lost tied embedding contributions,
incorrect metadata, changed initializer bits and nonfinite logits. These validate
the comparison machinery; synthetic observations are not compiled parity evidence.
Three native-retention parser tests check the exact field schema, missing or
duplicate reports, and mutated count/byte trajectories. Together the driver runs
**twelve tests**; actual compiled arena reports remain a separate gate.

The finite-difference test checks **all 1,184 unique values at each of two step
sizes**, for **2,368 central-difference derivative comparisons per fixture** and
4,736 perturbed forward evaluations per fixture. It now covers both the
distinguishable conditioned-state fixture and the exact seed-1729/input-[3,197]
public parity fixture: **4,736 comparisons and 9,472 evaluations in total**.
These are not additional distinct model parameters. Values are rounded to f32
before binary64 differentiation. Every canonical gradient has a nonzero witness.
A tied perturbation changes both uses of the same matrix; separate untied edge
VJPs must sum to its unique gradient. The public fixture's finite differences
connect actual Eshkol parity to checked derivatives at the same parameter point.

Zero upstream produces fourteen zero *contributions*. Adding those to existing
nonzero I2 accumulators would preserve their numerators while advancing count and
weight; present-zero accumulation itself requires an absent/already-zero prestate.
This Python reference does not implement or test I2 transactions or metadata.

## Compiled caller and comparison

`public_runtime.esk` calls the actual installed model API for three unconditioned
initializer/input cases. Its integer-only test ingress serializes the same three
finite seed tensors without computing model numerics. It emits all 512 logits per
case as exact bytes, releases the output, invokes VJP through the surviving logits
lease, and explicitly releases caller owners. The seven-facade package gate owns
its two fresh production AOT compilations and executions.

After a successful compiled run, compare its captured stdout with the current
generated reference (the decoder itself needs only standard-library Python):

```bash
python3 -m tests.m3.check_public_parity --reference build/m3-reference/development-a.json --public-output build/m3-reference/public.stdout
python3 -m tests.m3.generate_instrumented_caller --output build/m3-reference/instrumented_runtime.esk
```

The generated instrumented caller is linked separately against an `ET_M3_TESTING`
artifact with explicit test-only gradient/parameter-copy and metadata observers.
It compares all 14 accumulated numerators per case, exact initializer bytes, and
present/count-one metadata using `check_public_parity --instrumented`. Separate
case weights are 1, 1/2 and 5/2; expected VJP numerators do not change with the
weight. This detects accidental multiplication or division by normalization weight.
`scripts/test-m3-numerical.sh` builds this separate artifact from the exact trusted
Eshkol root and native source closure, validates 97 globals (93 production plus
four test observers), and links the same installed public wrappers into one
registry. The fourth observer reports arena/native retention counts. Test-only
package-builder counters are enabled separately and remain localized. The script
requires the canonical `build/m3` package and generated reference, and retains
its test artifact, facades, closure inventories and stdout under
`build/m3-numerical`. It never changes the canonical package policy.
The ordinary artifact must not expose these observer symbols. This test variant
does not substitute for production artifact/source/symbol/public AOT evidence.

The comparator verifies current generator/initializer/lock hashes and pinned
reference provenance. Numerical bounds apply independently to each tensor:
`max_error <= 1e-10 + 3e-4 * max_abs_reference`. Every gradient path is compared;
large gradients cannot conceal a missing smaller Q/K gradient. It prints measured
per-path errors for review. These mathematical f64-versus-f32 bounds require actual
native evidence; they make no bitwise f32/libm claim.

## Current limits

The generated oracle computes the mathematical graph and gradients in **f64**,
starting from exact f32 parameter values. It does not reproduce every contracted
serial f32 arithmetic statement, reciprocal/multiply order, or native libm result.
Its dedicated manifest is frozen; it is not a Q0 fixture or, by itself, an Eshkol
execution test, native provider parity, public model-output/VJP evidence, JIT
evidence, or supported Ubuntu/LLVM CI.

Actual compiled Eshkol/native comparison, per-path f32 tolerances, independent
statement-order/libm bit witnesses, broader schedule mutation coverage, public
seed/ownership/stale-primal/I2 atomicity tests
remain implementation evidence. No Python or PyTorch may enter the delivered
runtime or package. Tests alone do not authorize a new public ABI or model shape.
