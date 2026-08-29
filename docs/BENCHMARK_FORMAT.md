# B0 benchmark format and measurement contract

Status: **B0 implementation complete; independent integration review pending**.

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
wrong-version Eshkol checkout is rejected before execution. The canonical upstream
is `https://github.com/tsotchke/eshkol.git` at
`90cbd7130f47b8184bcc77b8d5c1b0026da980de`; no fork or alternate ref is accepted.
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
