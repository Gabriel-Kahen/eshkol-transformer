# AMD4 HIP observability ABI 1.0

Build with `bash scripts/build-amd4.sh`. Link
`build/amd4/libeshkol_transformer_amd4.a` and `-ldl`, and include
`eshkol_transformer/amd4_hip_abi.h` from C11 or C++17. The archive has four
exports and uses libc and the Linux dynamic loader. It requires no ROCm headers
or link-time libraries. The archive SHA-256 is emitted beside the archive.

## Device and environment probes

`et_amd4_probe_v1(ordinal, report)` checks read/write access to `/dev/kfd`
using `access(2)`. The result is `absent` for a missing path, `inaccessible`
for other access errors, or `accessible` if the path passed the permission
check. This check uses the caller's real IDs and cannot establish whether the
driver will open the device in the current container.

The probe opens `libamdhip64.so` and `librocblas.so` independently using
`RTLD_NOW | RTLD_LOCAL` and closes each acquired handle. Library state is
`unavailable` when the named library fails to open, including due to
unresolved dependencies, `missing-symbols` when an open library lacks a
required export, or `ready` when all required exports resolve. The LP64 HIP
adapter uses the versioned
`hipGetDevicePropertiesR0600` entry point. The byte layout was checked against
the AMD HIP 6.0.0, 6.4.3, and 7.2.0 declarations. Older runtimes without the
R0600 export report an unavailable runtime. [HIP 6.0.0 ABI](https://github.com/ROCm/hip/blob/rocm-6.0.0/include/hip/hip_runtime_api.h), [HIP 6.4.3 ABI](https://github.com/ROCm/hip/blob/rocm-6.4.3/include/hip/hip_runtime_api.h), [HIP 7.2.0 ABI](https://github.com/ROCm/hip/blob/rocm-7.2.0/include/hip/hip_runtime_api.h)

Pass `ET_AMD4_ALL_DEVICES` for ascending HIP-visible ordinals or a nonnegative
ordinal for one device row. The report always carries the requested ordinal,
any observed visible count, and the number of rows populated. A count above
`ET_AMD4_MAX_DEVICES` (64) yields `ET_AMD4_ERR_OUT_OF_RANGE` with the observed
count and zero rows. An explicit ordinal outside a nonempty visible set yields
the same status. HIP visibility and ordinal ordering are inherited from the
process environment.

The report records raw HIP runtime and driver version integers and the rocBLAS
version string. Device rows carry name, GCN architecture including any feature
suffixes, PCI domain/bus/device, compute units, wavefront size, and total
VRAM from HIP properties. AMD4 retains both values from `hipMemGetInfo` on the
selected device. Its total is reported separately from the property total.
Free memory is a snapshot taken before AMD4 creates the temporary rocBLAS
handle for that row. An actual free value of zero is valid when the memory
query succeeded. [HIP device management](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/device_management.html), [rocBLAS helper functions](https://rocm.docs.amd.com/projects/rocBLAS/en/docs-6.2.2/reference/helper-functions.html)

The probe saves the calling thread's HIP current device before any change.
For each row with usable properties and a known original device, it selects
that ordinal as needed, queries memory, and checks rocBLAS by creating and
destroying a temporary handle on that device. A failed memory query still
allows the handle check. The probe attempts to restore the original ordinal
if it tried to switch devices and cannot establish that the original is now
active. A return to the original device during enumeration needs no separate
restore call. Restore failure is visible in `restore_device_result`. Probe
calls can initialize HIP runtime and rocBLAS state.

### Results and ownership

The report is caller-owned, fixed-size storage. Each query result records an
AMD4 status and the raw HIP or rocBLAS return code. A zero raw code with a
non-OK status identifies a locally rejected response or a skipped stage.
Fields produced by a successful stage are valid only when its result is
`ET_AMD4_OK`. `ET_AMD4_NOT_RUN` is a stage value.

| Return | Condition |
| --- | --- |
| `ET_AMD4_OK` | Requested observations and rocBLAS handle checks succeeded |
| `ET_AMD4_ERR_INVALID_ARG` | Invalid pointer, storage overlap, ordinal below `-1`, or invalid ops/report descriptor |
| `ET_AMD4_ERR_UNAVAILABLE_RUNTIME` | HIP library unavailable or missing required exports |
| `ET_AMD4_ERR_NO_DEVICE` | Zero HIP-visible devices or HIP error 100 |
| `ET_AMD4_ERR_DEVICE_UNSUPPORTED` | HIP error 801 or unusable GCN architecture string |
| `ET_AMD4_ERR_ROCBLAS_UNAVAILABLE` | rocBLAS library unavailable, missing exports, unsuccessful version or handle check, or invalid version/handle response |
| `ET_AMD4_ERR_DRIVER_FAILURE` | Other HIP error or invalid HIP response |
| `ET_AMD4_ERR_OUT_OF_RANGE` | Invalid HIP ordinal (error 101), ordinal outside the visible set, over-capacity visible count, or short format buffer |
| `ET_AMD4_ERR_ALLOC` | HIP out-of-memory (2) or rocBLAS memory error (5) |

A valid probe invocation fills a report even if it returns an error. The
top-level result prioritizes HIP library availability, count and selection
bounds, then the first HIP error in version/current-device/row/restore order.
With those observations successful, the first rocBLAS error determines the
result. rocBLAS version discovery is independent of the HIP library state.
Invalid arguments leave caller storage intact.

`et_amd4_probe_with_ops_v1(ops, ordinal, report)` uses the same probe logic and
marks the report `injected` so fixture data is identifiable. Initialize
`ops->struct_size` to `sizeof(et_amd4_ops_v1)`. Supply all callbacks for each
`READY` library, using zero for native success and native error codes
otherwise. The table and callback context must stay valid for the call and use
separate storage from the report. The probe calls destroy once for every
non-null handle returned by create before changing devices, including when
create reports an error. The report records both raw statuses and prioritizes
a create error over a destroy error for the same handle. Report fields and ABI
layout are fixed for 1.0.

`et_amd4_format_report_v1` emits LF-terminated ASCII beginning with
`eshkol-transformer-amd4-observability:v1`. It uses a fixed field and row order,
decimal integers, and quoted strings. Bytes outside printable ASCII, quotes,
and backslashes are encoded as lowercase `\xhh`. Query lines use
`key=status,native-code`. Unobserved values are omitted and their query
results remain present. Pass `NULL, 0` to measure the required capacity,
including the terminal NUL. A short buffer returns
`ET_AMD4_ERR_OUT_OF_RANGE`, sets the required capacity, and preserves the
buffer. Report, buffer, and capacity output use separate storage.

## K1 capability discovery

`et_amd4_kernel_provider_v1()` returns an immutable K1 ABI 1.0 provider for
`kernel.amd-hip.gemm` and `kernel.amd-hip.attention`, both with device
`amd`. AMD4 names the unavailable state `ET_KERNEL_CAPABILITY_UNAVAILABLE`,
an alias of the pinned K1 value `ET_KERNEL_CAPABILITY_UNSUPPORTED`
(`"unsupported"` in the K1 JSON report). Both entries are always unavailable,
including after a successful host probe. K1 dispatch rejects these entries
with `ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED`. Direct provider validation
always returns `ET_KERNEL_ERROR_UNSUPPORTED` and records
`ET_KERNEL_CODE_PROVIDER_REJECTED` when the supplied error output is disjoint
from call, request, tensor table, and readable tensor storage. Direct
invocation has no effect. Both paths preserve tensor data and have zero CPU
fallback.

## Measured validation

In `eshkol-r0-debug:20260928` on x86-64 with Clang 21.1.8, the live probe
returned `ET_AMD4_ERR_UNAVAILABLE_RUNTIME`. It reported `kfd=absent`,
`hip.library=unavailable`, `rocblas.library=unavailable`, a device count query
of `not-run,0`, and zero device rows. Both K1 capabilities were unavailable.

`CC=clang-21 CXX=clang++-21 AMD4_ASAN_DETECT_LEAKS=1 bash scripts/test-amd4.sh`
passed the strict C11/C++17 build, archive symbol and reproducibility checks,
live absence assertions with `AMD4_EXPECT_NO_HIP=1`, and the injected and K1
tests in normal and ASan/UBSan/LSan builds. The Python isolation and CI
topology checks also passed.

Live device enumeration and rocBLAS handle checks on hardware, direct AMD GPU
execution, numerical parity, and GPU benchmarks remain pending a
ROCm-provisioned runner.

## Eshkol AOT bridge and compiled validation

Build the Eshkol AOT bridge archive with `bash scripts/build-amd4-aot.sh` to
produce `build/amd4_aot/libeshkol_transformer_amd4_aot.a` alongside its SHA-256
file. Include `native/amd4_eshkol_bridge.h` from C11 or C++17 and link
`libeshkol_transformer_amd4_aot.a`, `libeshkol_transformer_amd4.a`,
`libeshkol_transformer_k1.a`, and `-ldl`. The archive exports 24
`et_amd4_aot_*` entry points covering:

- Live host probing (`et_amd4_aot_probe_live` via `et_amd4_probe_v1`),
  report field accessors, formatting (`et_amd4_aot_format_report`), and byte
  inspection (`et_amd4_aot_read_byte`).
- Four deterministic `injected` mock scenarios via `et_amd4_probe_with_ops_v1`:
  a two-device MI250X (`gfx90a:sramecc+:xnack-`, 110 CUs, wavefront 64) plus
  RX 7900 XTX (`gfx1100`, 96 CUs, wavefront 32) host, an OOM memory-query fault
  (`ET_AMD4_ERR_ALLOC`), a HIP driver failure prioritized over a rocBLAS handle
  error (`ET_AMD4_ERR_DRIVER_FAILURE`), and a rocBLAS missing-symbols state
  (`ET_AMD4_ERR_ROCBLAS_UNAVAILABLE`).
- K1 runtime registration for `et_amd4_kernel_provider_v1()`, capability status
  and JSON report queries, and explicit dispatch rejection for
  `kernel.amd-hip.gemm` and `kernel.amd-hip.attention` with 32-bit canary
  marshaling (`et_amd4_aot_store_u32` / `et_amd4_aot_load_u32`).

### Measured AOT validation

In `eshkol-r0-debug:20260928` on x86-64 with Clang 21.1.8 and pinned Eshkol
(`--strict-types --optimize 0 --no-stdlib`),
`CC=clang-21 CXX=clang++-21 bash scripts/test-amd4-aot.sh` passed:

- Reproducible bridge archive `libeshkol_transformer_amd4_aot.a`
  (SHA-256 `6afb0c891d49c029a7aaabf49530772829bf277e54c83c47ee793dd75239d5c6`),
  exact 24-symbol export audit, import allowlist audit, and C++17 linkage check
  (`tests/amd4/header_bridge_cpp.cpp`).
- Direct C11 bridge tests (`tests/amd4/test_amd4_aot_bridge.c`, 153 checks) in
  both normal and ASan/UBSan/LSan (`detect_leaks=1:halt_on_error=1`) builds.
- Two fresh AOT compilations of `tests/amd4/test_amd4_aot.esk` from isolated
  output and cache directories produced byte-identical ELFs
  (SHA-256 `0754b33e058338bf1f2bcaab0d335b3f0f81a5ea4ec4996b60bcb8020e6d3efe`)
  and identical stdout with empty stderr:
  - Live host absence: `status=unavailable-runtime`, `kfd=absent`,
    `hip.library=unavailable`, `rocblas.library=unavailable`, `devices=0`
    (460-byte formatted report).
  - Injected two-device report: `status=ok`, `devices=2` (1164-byte formatted
    report) verifying exact `eshkol-transformer-amd4-observability:v1` lines.
  - Injected failure scenarios: `oom=alloc`, `driver=driver-failure`,
    `rocblas=missing-symbols`, plus invalid scenario, out-of-range ordinal, and
    untouched short-buffer checks.
  - K1 rejection: both AMD capabilities reported `unsupported` (in capability
    queries and K1 JSON), dispatch rejected with `ET_KERNEL_ERROR_UNSUPPORTED`
    and `ET_KERNEL_CODE_CAPABILITY_NOT_VERIFIED` (`15`), zero silent CPU
    fallback, and output canary `0xdeadbeef` preserved unchanged.

Live ROCm device execution and hardware validation remain pending a
ROCm-provisioned runner.
