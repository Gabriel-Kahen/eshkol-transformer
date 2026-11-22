# B0 benchmark format and measurement contract

Status: **B0 foundation complete; supported attention report v2 measured and
independently reviewed on its exact candidate head**.

## Scope

B0 establishes a reproducible smoke-benchmark definition, a Linux host-process
measurement runner, and a versioned report format. It measures the F0 native smoke
artifact. It does not benchmark a tensor kernel or training loop and makes no
acceleration, device-memory, or performance-threshold claim.

The only supported B0 execution contract is:

- backend `host-cpu`, device `cpu`, with direct process execution as evidence;
- dtype `not-applicable` and an empty shape list because the F0 smoke workload has
  no tensor inputs;
- `CLOCK_MONOTONIC` elapsed time in nanoseconds;
- Linux `pidfd` polling for child-completion notification, with no timed polling or
  alternate completion source;
- Linux `wait4(2)` `ru_maxrss` in kibibytes for the direct child process;
- an exact expected exit status, empty stderr, and SHA-256 of stdout.

Requests for another backend, device, dtype, shape contract, clock, memory source,
or acceleration claim are rejected. No substitute implementation is selected.

## Stable definition

[`benchmarks/smoke_v1.json`](../benchmarks/smoke_v1.json) is the checked-in source
of benchmark facts. It is canonical UTF-8 JSON: sorted keys, compact separators,
no duplicate keys, and exactly one final newline. Its envelope has these fields:

```text
format = eshkol-benchmark-definition
version = 1
checksum.algorithm = sha256
checksum.digest = SHA-256(canonical JSON of format, version, and payload)
payload = the benchmark contract
```

The payload fixes the argv without a shell, working directory, workload count,
warmup and measured repetition counts, timeout, backend/device evidence, dtype,
shapes, output expectations, clock, units, memory source, and measurement scope.
Changing any fact requires a new valid checksum and an explicit format/version
decision when compatibility changes.

## Generated report

`make benchmark` writes `build/benchmarks/smoke-v1-report.json`. Reports use the
same canonical, checksummed envelope convention with format
`eshkol-benchmark-report` and version `1`. The report payload deliberately separates:

- `stable`: benchmark-definition identity and fixed contract facts (backend, dtype,
  shapes, warmup, repetitions, and work unit), project revision and dirty state,
  target hash, measurement-launcher source and binary hashes, canonical Eshkol
  repository/revision/version/compiler hash, and exact workload outcome hashes;
- `volatile`: observation time, host/OS/CPU/compiler metadata, supported-host versus
  compatibility-only status, raw measured samples, and derived summaries. Fixed
  contract facts are not duplicated here.

The volatile section is expected to differ between runs. It must not be checked in
as a universal baseline or compared byte-for-byte. Consumers first validate the
report checksum and schema, then compare only fields appropriate to their question.
The stable section is reproducible only for the same project tree, target binary,
toolchain, and definition; a dirty working tree is recorded explicitly.

Each measured sample records repetition number, elapsed nanoseconds, peak RSS in
kibibytes, work-unit count, and throughput in runs per second. Latency summaries are
the integer minimum, median, and maximum elapsed nanoseconds. Peak RSS is the maximum
sample high-water mark. Throughput is rendered as a decimal string to avoid
platform-dependent JSON floating-point encodings: `(work_count * 1_000_000_000) /
elapsed_ns`, rounded half-even to six decimal places. Aggregate throughput uses total
work divided by total elapsed time. Validators recompute and require every value.

## Running and validating

On the F0 supported lane:

```bash
/usr/bin/bash -c 'make clean && make configure && make build'
/usr/bin/bash -c 'make test && make smoke'
/usr/bin/bash -c 'make benchmark'
python3 tests/b0/run_benchmark.py verify \
  benchmarks/smoke_v1.json build/benchmarks/smoke-v1-report.json
```

`make benchmark` reuses F0 verification, so a missing, dirty, wrong-revision, or
wrong-version Eshkol checkout is rejected before execution. The canonical compiler
source for this format is `https://github.com/Gabriel-Kahen/eshkol.git` at
`81298b4a9608fb92eb6f351a2eabd8392da7d9ef`. The benchmark validator rejects every
other repository or revision.
F0 target compilation and measurement-launcher compilation are setup phases that
finish before timing begins and are excluded from elapsed/RSS samples. Their source
and binary identities are recorded by stable hashes; the exact measured argv, working
directory, timeout, warmup, and repetition configuration are sealed by the definition
whose digest is recorded in the report.
Report replacement is atomic after all warmups, repetitions, output checks, schema
validation, and checksum validation succeed.

## Measurement limitations

- Wall time includes process creation, dynamic loading, and workload execution. The
  smoke workload is short, so scheduler, cache, frequency, and system load effects
  can dominate it. The end timestamp is taken when the Linux `pidfd` becomes ready;
  kernel wakeup and parent scheduling latency are therefore included. B0 rejects an
  unavailable `pidfd_open` instead of reverting to interval polling.
- `ru_maxrss` is the Linux high-water resident set for the measured direct process,
  reported by the kernel in KiB. It is not an allocation count, aggregate process-
  tree memory, accelerator/device memory, or a portable cross-OS metric.
- Warmups reduce first-run effects but do not make different hosts comparable.
- `ESHKOL_ALLOW_UNSUPPORTED_HOST=1` produces compatibility-only evidence. Such a
  result is never presented as supported-lane evidence. The benchmark invocation
  derives this label from the current verified run, not stale configure metadata.
- B0 observes only host CPU process execution. It provides no evidence that a GPU or
  other accelerator executed work.

## A2 attention extension v1

The separate `eshkol-attention-benchmark-definition` and
`eshkol-attention-benchmark-report` formats (both version `1`) add real A2 CPU-f32
attention measurement. They reuse the canonical/checksummed envelope codec above;
the smoke definition/report schemas are unchanged and reject these formats.
`benchmarks/attention_v1.json` seals all ten exact attention rows admitted by A2,
forward and analytic backward, deterministic requests, ten warmup dispatches,
1,000 timed dispatches, three fresh-process repetitions per case, and a 30-second
per-process timeout. Counts may be changed only by a newly checksummed definition
within the documented validator bounds. Device, dtype, provider, shape set,
operations, input recipe, and measurement sources cannot be substituted.

`benchmarks/attention_v1.c` links the accepted A2 provider and K1 directly. It
performs K1 provider discovery, capability resolution, and dispatch; every timed
iteration includes K1 validation plus A2 execution. No production ABI changes,
provider from an unmerged branch, Eshkol compiler invocation, Python numerical
implementation, or GPU execution participates. This is development benchmark
tooling, independent of the delivered Eshkol-native training path.

For each shape `[N,Hq,Hkv,Tq,Tk,Dh]`, Q/K/V/upstream use dense row-major f32
values `((7*i + 3*role) mod 23 - 11) / 16`, with roles 0/1/2/3 and the flat index
restarting at zero for each buffer. Query positions are `q+1+batch`; key positions
are `k+batch`. The exact `[N,Tq,Tk]` mask keeps entries when
`(q+k+batch) mod 3 != 1`, except the last query is fully masked when `Tq > 1`.
These dyadic inputs are exactly representable. A2 applies its causal position
constraint in addition to the keep mask. Shapes and all tensor tables follow
[A2's accepted ABI](A2_ATTENTION.md); outputs remain caller-owned and disjoint.

Each fresh process allocates/prepares its fixed bounded stack buffers, discovers
K1, resolves the capability, and performs warmup before starting
`CLOCK_MONOTONIC`. `dispatch_elapsed_ns` times only the requested batch of
synchronous K1 dispatch calls. It excludes compilation, initialization, warmup,
output checks/serialization, and process startup/teardown. It includes loop,
validation and clock-call overhead; tiny admitted shapes make those costs
material. This is dispatch latency, not isolated arithmetic-kernel latency.
`process_elapsed_ns` separately measures the complete child using the existing
pidfd helper, including setup, warmup, output serialization and termination.
`process_peak_rss_kib` is that child's `wait4` high-water RSS including those
phases, stack, runtime and code. It is not an allocation count, incremental
attention working set, process-tree peak, or device-memory measurement. No
subtraction of a baseline or invented per-dispatch memory value is performed.

The native child checks finite results and emits every final output f32 word as
eight hexadecimal digits, most-significant digit first, in output-table order
(result or dQ/dK/dV). Python hashes the decoded bytes with SHA-256. All measured
repetitions for a case must produce the same witness; nonfinite, malformed or
all-zero final buffers, nonzero exits, signals, timeouts, or stderr reject the
run. Output witnesses prove observed repeatability on this host/toolchain; they
are not independent numerical correctness or cross-platform bitwise claims.
The existing bounded A2 provider gate retains numerical/finite-difference proof.

The strict report schema stores the full definition and its digest, project
commit/dirty state, source-closure hashes, compiler path/binary hash/version and
fixed flags, and target/helper hashes under `stable`. Sources must have identical
hashes before compilation and after measurement. `volatile` records UTC time,
OS/kernel/architecture/CPU metadata, compatibility-only status, and ordered cases
with raw per-repetition dispatch/process/RSS samples and output witnesses.
Summaries recompute minimum, lower median, maximum batch dispatch nanoseconds,
maximum process RSS, and total dispatches/total dispatch time (six-decimal,
half-even throughput string). Validators reject unknown fields, invalid types,
wrong sample/case order, mismatched witnesses, and inconsistent summaries.
Reports replace their destination atomically only after every case succeeds and
validates. A failed run preserves the previous report.

Run the small native build and benchmark without building the core toolchain:

```bash
make benchmark-attention
python3 -B tests/b0/attention_benchmark.py verify \
  benchmarks/attention_v1.json build/benchmarks/attention-v1-report.json
python3 -B -m unittest tests.b0.test_attention_benchmark -v
```

`CC` selects the native C compiler and `BENCHMARK_REPORT` selects the report
path for the Make target. Compilation uses `-O2 -ffp-contract=off` and strict
warnings; source/compiler/binary identities remain explicit. The wrapper checks
available disk space with `df` and holds the shared heavy-build lock. Every report
is conservatively labeled `compatibility-only`; this standalone benchmark does
not verify or establish the project's supported Eshkol toolchain lane. Supported
CI/admission and PA4 candidate parity remain
separate gates. No speedup, reduced memory, large-context scalability, training
throughput, or admission of PA4 draft #215 follows from these measurements.

### Reviewed supported-lane attention report v2 extension

`make benchmark-attention-supported` requests report format version `2`, while
reusing the exact version-1 attention definition, native workload, and measurement
semantics above. Version-1 reports remain compatibility-only; their validator
continues to reject a `supported` label. Version 2 requires `supported` and adds
the exact toolchain lock and build-provenance TSV bytes, their SHA-256 identities,
and explicit host ID/version/architecture evidence. Its validator requires the
Ubuntu 22.04 x86-64, Clang/LLVM 21.1.8 lock, checks the embedded provenance
against that lock and the recorded compiler, and rejects a dirty project tree.
Neither format permits another device, shape, provider, or numerical workload.

Before compilation and again after the final sample, the v2 runner calls the
project's F0 `verify_toolchain` gate and binds the benchmark compiler to the
verified provenance C compiler. It rejects any
`ESHKOL_ALLOW_UNSUPPORTED_HOST` setting, checks the project tree and source
hashes, and publishes only after all checks pass. A failed check leaves an older
report untouched. The checksum and embedded evidence make the report auditable;
they are not a cryptographic attestation that the host executed it. Supported
admission requires an exact-head supported-lane run. Independent source and
test review accepted this contract. Exact-head hosted run
[43930911395](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/43930911395)
at `9af5513d` passed the supported report verifier for all ten shapes, forward
and backward, with three repetitions each (60 samples). The sealed report's
SHA-256 is `ab8bc9ba1f92ce2e92fcd58a34b5c876b714f7e79dd00b4bb8780e135709ab52`.
The separate ordinary full CI run `43930811027` passed its exact head. Neither
run establishes a PA4 speed or memory improvement.

An explicit `workflow_dispatch` mode in `.github/workflows/ci.yml` selects only
the supported attention benchmark. It requires the exact
`codex/b0-attention-supported-v2-20261116` ref and a matching full
`expected_attention_head` SHA. The Ubuntu 22.04 job installs pinned LLVM,
verifies the locked toolchain, runs `make benchmark-attention-supported`, and
uploads the version-2 report plus checksummed setup and benchmark logs. Its
default-false selector leaves ordinary CI and ordinary manual dispatch on the
full-coverage path. This diagnostic job is explicitly non-admission.
