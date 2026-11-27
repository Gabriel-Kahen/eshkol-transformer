# Toolchain compatibility

`eshkol.lock` is the authoritative F0 compiler/runtime and host-tool compatibility
lock. It is tab-separated data and is never executed. The bootstrap verifies exact
Eshkol, LLVM, and Clang versions plus explicit minimum CMake, Ninja, Make, Bash, and
Git versions. It records the actual host-tool versions in provenance, binding the
compiler binary hash and CMake source directory to the pinned checkout. Configure
revalidates that record before project code is compiled. All project commands use
the workspace-local compiler, and AOT linking explicitly reuses the C++ driver in
its provenance; there is no PATH, Python, or alternate-runtime fallback. Minimums
are intentional because GitHub-hosted Ubuntu images update these utilities in place;
the exact build-critical compiler/runtime inputs remain pinned.

The supported CI lane is Ubuntu 22.04 x86-64 with LLVM/Clang 21.1.8. Install the
apt.llvm.org Jammy LLVM 21 repository and these packages:

```text
build-essential clang-21 cmake git gnupg llvm-21 llvm-21-dev lld-21 make
ninja-build pkg-config wget libjpeg-dev libncurses-dev libopenblas-dev
libpcre2-dev libpng-dev libreadline-dev libsqlite3-dev libssl-dev libwebp-dev
```

The bootstrap disables Eshkol tests, examples, agent FFI, XLA, quantum, and
TensorCore integrations. Those are outside F0 and disabling them avoids unneeded
FetchContent dependencies; it does not claim those capabilities are unavailable in
Eshkol generally. The delivered smoke uses `--no-stdlib`, loads its internal library
through an explicit `-I src` path, emits a depfile in compile-only mode, then performs
a separate AOT executable build and execution.

CI can restore the workspace-local source and build cache. Bootstrap still runs
CMake configure and verifies the pinned source and supported toolchain.
See the [CI efficiency notes](../docs/CI_EFFICIENCY.md).

`ESHKOL_ALLOW_UNSUPPORTED_HOST=1` exists only for explicitly labeled compatibility
probes. A configure manifest records `host_supported=false`; such a probe is useful
diagnostic evidence but does not expand the supported matrix or satisfy its CI gate.

## Local F0 warm replay

Run the opt-in probe with a trusted, populated `SEED_DEPS` containing
`eshkol-src` and `eshkol-build` originally configured at `/workspace/.deps`.
Use rootless Docker with a locally retained Ubuntu 22.04/LLVM 21.1.8 image
identified by its full image ID. The checkout must include F0 commit
`0d8bd51a` and its parent. The output directory must be new.

```bash
scripts/probe-f0-warm-replay.sh "$SEED_DEPS" "$IMAGE_SHA256" "$NEW_RECEIPT_DIR"
(cd "$NEW_RECEIPT_DIR" && sha256sum -c SHA256SUMS)
```

The probe holds both shared build locks and runs the same `make toolchain`
offline at the same container paths. Each run starts with a fresh copy whose
contents, modes, ownership and nanosecond mtimes match the seed. The
unmodified source and lock stay pinned. The two exact bootstrap script
revisions differ by the header mtime fix.

The receipt records the header digest and timestamp before configure, after
CMake returns, and before and after Ninja. It also records the CMake cache
and Ninja graph digests, compiler provenance, Ninja logs and phase timings.
The gate requires equal post-configure header bytes, a changed mtime before
Ninja in the predecessor, a restored mtime in the fixed run, and build
commands only in the predecessor. CMake's glob check appears in both raw
Ninja logs. An exit status of zero and a verified
`SHA256SUMS` complete the gate. Single-pair elapsed times describe this
local replay. Measurement of hosted savings requires a matched hosted run.
