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

On 2026-11-25, Linux x86-64 with Clang 22.1.8 reported `kfd=accessible`
and `hip.library=unavailable` independently. rocBLAS was also unavailable;
the live probe returned `ET_AMD4_ERR_UNAVAILABLE_RUNTIME`, a device-count
result of `not-run,0`, and zero device rows. Both K1 capabilities remained
unavailable. KFD access is a permission observation, not evidence of HIP
execution.

`CC=clang CXX=clang++ AMD4_ASAN_DETECT_LEAKS=1 bash scripts/test-amd4.sh`
passed the strict C11/C++17 build, deterministic archive and symbol checks,
injected device and failure tests, live host observations, and K1 fail-closed
checks in normal and ASan/UBSan/LSan builds (923 checks per C11 run).

Live device enumeration and rocBLAS handle checks on hardware, direct AMD GPU
execution, numerical parity, and GPU benchmarks remain pending a
ROCm-provisioned runner.
