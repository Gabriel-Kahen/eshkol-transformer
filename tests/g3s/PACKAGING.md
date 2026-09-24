# G3-S package gates

The normal artifact contains exactly `g3s_sampling_provider.o`, compiled as C11
with the verified Eshkol toolchain's C compiler, `-O2`, PIC, strict warnings,
stack protection, and the accepted strict binary32 flags. `-Wvla -Walloca` are
errors; compiler stack-usage output is retained alongside the dependency file.
Staged outputs publish the archive last using atomic same-filesystem replacement.
The sole defined external symbol is `et_g3s_kernel_provider_v1`; there are no
installed Eshkol exports. Normal defined/undefined and actual non-system depfile
closures are compared with exact manifests. Instrumented dependencies are reported
separately and cannot enter the canonical archive.

`provider_report.json` is constructed independently from the accepted two rows
and the frozen K1 eleven-row baseline, before running the provider. It is 3,571
bytes, SHA-256 `c7eb54448f4e2a0dde003f832f83a0893fb073a2dde7d2a838ea99528942bcaa`.
Both fresh reports must match every byte. Linking the complete archive into the
provider-free K1 report consumer must preserve its established baseline hash.
Predecessor source hashes cover K1, I1/I2, N2/N3K, A2, the oracle lock, and the
accepted sampler ABI text.

The stack proof uses normal-object compiler output including spills. All emitted
frames must have static sizes. The optimized LLVM internal call graph must be
acyclic and contain no indirect helper calls. Summing every emitted provider frame
plus a conservative 136 bytes per frame for return-address/red-zone allowance
bounds every possible simultaneously live provider/helper path, overcounting
mutually exclusive calls. The bound must not exceed 16,384 bytes. Caller/K1 and
external libc/libm frames are excluded from this compiler claim. An independent
painted alternate-stack test measures the real normal object, including its small
measurement callback and external library frames, and also enforces 16,384 bytes.
Neither test asserts a sanitizer-build stack bound.

Two fresh deterministic archive builds are compared with the canonical object,
depfile, stack report and archive. Actual pinned-Eshkol interop builds one private
object and two independent executables with separate disabled-cache directories;
executable bytes and output bytes must match. The private transport exercises both
ordered tensor schemas, greedy exhausted-state preservation, categorical exhausted
failure preservation, and successive categorical counters. Its symbol must fail
to link from production archives. This is development transport only, without an
authenticated shell, public Eshkol API, or generation-completion claim.

`make test-g3s` builds the focused prerequisites and executes these gates plus
independent numerical, mutation, adversarial/fenv/carrier and sanitizer tests.
Normal build, local full/predecessor tests and the dedicated 75-minute g3s-sampling CI
suite include G3-S; all predecessor commands remain covered. Supported-lane
Ubuntu 22.04 / LLVM 21.1.8 results must be recorded separately from compatibility
execution on the local LLVM 22/CachyOS host.

The LLVM 22.1.6 compatibility object currently measures seven undefined symbols:
`__stack_chk_fail`, `et_kernel_error_clear`, `expf`, `fegetenv`, `fesetenv`,
`snprintf`, and `strcmp`. Its separate compatibility manifest does not establish
the supported LLVM 21.1.8 subset; the supported manifest must pass on that exact
compiler before acceptance. The accepted nine-symbol ceiling remains a separate
immutable comparison, including `memcpy` and `memset` even when optimized away.
