# Toolchain compatibility

`eshkol.lock` is the authoritative F0 compiler/runtime and host-tool compatibility
lock. It is tab-separated data and is never executed. The bootstrap verifies exact
Eshkol, LLVM, Clang, CMake, Ninja, Make, Bash, and Git versions, then writes provenance
binding the compiler binary hash and CMake source directory to the pinned checkout.
Configure revalidates that record before project code is compiled. All project
commands use the workspace-local compiler; there is no PATH, Python, or
alternate-runtime fallback.

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

CI does not restore a compiled dependency cache. Correct caching would need to bind
the OS image, architecture, compiler/LLVM versions, CMake flags, Eshkol SHA, and all
generated artifacts. Until such provenance is checked, rebuilding the source pin is
slower but safer.

`ESHKOL_ALLOW_UNSUPPORTED_HOST=1` exists only for explicitly labeled compatibility
probes. A configure manifest records `host_supported=false`; such a probe is useful
diagnostic evidence but does not expand the supported matrix or satisfy its CI gate.
