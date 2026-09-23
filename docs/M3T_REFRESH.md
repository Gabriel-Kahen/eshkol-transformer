# M3T current-main proposal refresh — 2026-09-19

Status: **historical pre-implementation refresh; superseded by
[M3T implementation acceptance](M3T_TRANSPORT.md)**. The
[binding disposition](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5740555652)
accepts exact proposal `e7ad6b0d82f4cf7f5d45efff04179433379965bb` and lifts the
implementation hold. The evidence below predates that disposition; it does not
establish implementation acceptance. At that point M3T and M3 were incomplete;
M3T has since merged, and [bounded M3 acceptance](M3_MODEL.md) supersedes its
historical downstream/incomplete status. All pre-implementation
statements below describe the refresh date, not the current runtime.

Accepted base: `7d9ad1cf6d997fc3c1f489f3955ff97c97281f68`.
Local history-preserving merge: `207d0e95047fe3f6b73880a0c163369c5b375bf0`.
Preserved proposal/probes: `8b72fde`.
Tracking: [#68](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/68).
Original [proposal](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5587828007)
and [refinement](https://github.com/Gabriel-Kahen/eshkol-transformer/issues/1#issuecomment-5587947469).
The accepted design is [the amended contract](M3T_TRANSPORT_PROPOSAL.md),
with [public witness design](M3T_PUBLIC_AOT_TEST_DESIGN.md).

## Actual amendments

Three independent `gpt-6-astra` / high audits covered contracts, ownership and
compiled public/package closure. All found the bounded **85-global / 79-package-
export sibling** sufficient; no correctness dependency requires the 119-global
full C2 successor. The integration preference for the sibling is incorporated
for disposition, not treated as production authorization.

- Install only config/module/byte-tokenizer/error_public/error_consumer and the
  proposed diagnostic facade. One registry-owning artifact per process; no
  D2/O2/K2/C2 interoperability. The existing 47 globals plus 38 M3T wrappers remain
  exact. Current C2's additional 34 globals are 12 K2, 11 D2, 6 O2 and 5 C2.
- Correct Eshkol P1 expectations from 41 to 46 private, 59 to 64 total; preserve
  all ten C2 seams and slots 0..58, appending construction at 59..63. Native
  identity remains a separate 31 to 36 private change. No generator changes are
  made before acceptance.
- Construction must preserve merged P1 promotion barriers and canonical identity
  readback, including repaired parameter-plan edges. Complete promotion before
  nonfailing publication and preserve affected P1/C2 regressions. I2's scoped-guard
  requirement and private constructor revoke/seal requirement remain unchanged.
- Strengthen planned memory evidence with exact retained Eshkol arena accounting;
  RSS is supplementary. Released aliases require live Eshkol allocations. C2-owned
  direct copies neither replace K1 guards nor grant authority over M3T storage.
- E1B's proposed native-input extension must be an exact repository-owned tuple,
  with object/depfile provenance and localization, not generic external objects.
  Future gate routing must extend CI-E2's accepted 16-suite / 23-command engine.

The K2/C2 limitation described below is historical M3T acceptance evidence and is
superseded for current persistence admission by
[SHARED-R2](SHARED_R2_PERSISTENCE.md).

There are no changes to the 38 public names/arities, profile, typed initializer,
X1, numerical providers or complete-model scope. Full training composition and
capability widening remain explicit downstream dependencies. Current K2's fixed
I2 report and C2's rank-two public-load rejection remain unchanged; even the
119-global composition would not establish diagnostic checkpoint support.

## Fresh bounded evidence

Native checks used x86-64 CachyOS, Clang 22.1.6. Public probes used clean pinned
Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de`; compiler binary SHA-256
`3e0b923e2e272a89474dff6739b6a2023d71ae483863a64b685ec8ea71ca1cc0`.
Recorded toolchain/compiler bindings passed `verify_toolchain`, with explicit
unsupported-host warnings. All results are local compatibility evidence.

| Check | Result |
|---|---|
| `git diff --check` | PASS |
| `/usr/bin/bash scripts/generate-p1-roots.sh --check` | PASS; current generated roots unchanged |
| Existing native I2 borrow probe, 1,000 cycles | PASS: 1,000 retired 96-byte shells, 96,000 bytes, zero live borrows |
| Same probe, 30,000 cycles with 20-second bound | Exit 124, no output; no refreshed 30,000-cycle result |
| Existing C2 model-copy test | PASS: 225 checks, including allocation-disabled repeated copy and flat live/retired/borrow-event counts |
| Four current public import-only compile probes | PASS; symbol counts below |
| Required facade caller linked against reused I2 archive | PASS: exact 47-global definitions and `M3T-BASE-IMPORTS-OK` |
| Data-only import linked against same I2 archive | Expected failure: missing D2 wrappers |

| Imported facades | E1B undefined symbols | Missing from I2's 47 globals |
|---|---:|---:|
| config/module/tokenizer/error_public | 38 | 0 |
| data | 25 | 11 |
| persistence | 11 | 4 |
| capabilities | 18 | 12 |

Top-level imported safe closures retain those references even without an API
call. The required caller's depfile contains only the six public sources,
including error_consumer. Archive reuse proves link compatibility, not fresh
artifact reproducibility. No M3T artifact exists yet. No full CI, numerical
reaudit, public AOT lifetime test or supported-host acceptance was run. The
September 8 30,000-cycle native/region results remain historical evidence only.

## Commands and retained evidence

Commands ran from this worktree using `/usr/bin/bash`, without a login shell.
Native output is retained in the task transcript; binaries are temporary.

```bash
clang -std=c11 -Iinclude -Inative \
  tests/probes/m3t_lifetime/borrow_retention.c native/kernel_abi.c \
  -lm -o /tmp/m3t-refresh-borrow-retention
timeout 20s /tmp/m3t-refresh-borrow-retention 30000
timeout 10s /tmp/m3t-refresh-borrow-retention 1000
clang -std=c11 -O2 -Iinclude -Inative \
  -DET_I2_NATIVE_HELPERS_ONLY -DET_C2_I2_MODEL_COPY -DET_F32_TENSOR_TESTING \
  tests/i2/test_c2_model_copy.c native/i2_wave2_package_bridge.c \
  native/f32_tensor.c native/kernel_abi.c -lm -o /tmp/m3t-refresh-c2-model-copy
timeout 10s /tmp/m3t-refresh-c2-model-copy
```

The prior `41a1` toolchain is absent and this worktree has no `.deps`. Provenance
verification used the existing `7fca` toolchain:

```bash
env ESHKOL_ALLOW_UNSUPPORTED_HOST=1 \
  ESHKOL_SOURCE_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-src \
  ESHKOL_BUILD_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build \
  LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config /usr/bin/bash -c '
source scripts/common.sh
verify_toolchain
resolve_provenance_compilers probe_cc probe_cxx /usr/bin/clang /usr/bin/clang++
printf "PROVENANCE_OK %s %s\n" "$probe_cc" "$probe_cxx"
'
```

Probe sources, depfiles, objects, symbol lists and logs remain in
`/tmp/m3t-interface-refresh.ZYrti3`. Each source requires the facade(s) listed
above, displays a fixed diagnostic and calls newline. The required source prints
`M3T-BASE-IMPORTS-OK`. The initial unsupported `--depfile` invocation failed;
the successful runs used `--emit-depfile`:

```bash
probe_dir=/tmp/m3t-interface-refresh.ZYrti3
probe_compiler=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build/eshkol-run
for name in required data persistence capabilities; do
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$probe_dir/cache-$name" ESHKOL_LIB_DIR="$PWD/lib" \
    ESHKOL_CXX_COMPILER=/usr/bin/clang++ timeout 45s "$probe_compiler" \
    --strict-types --no-stdlib --compile-only -I "$PWD/lib" \
    --emit-depfile "$probe_dir/$name.d" \
    "$probe_dir/$name.esk" -o "$probe_dir/$name.o" > "$probe_dir/$name.log" 2>&1
  nm -u "$probe_dir/$name.o" | awk '/et_e1b_/ {print $2}' | LC_ALL=C sort \
    > "$probe_dir/$name.e1b-undefined.txt"
  comm -23 "$probe_dir/$name.e1b-undefined.txt" native/i2_wave2_defined_symbols.txt \
    > "$probe_dir/$name.missing-i2.txt"
done
artifact_dir=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/build/i2
nm -g --defined-only --format=posix "$artifact_dir/i2_wave2.o" | awk '{print $1}' \
  | LC_ALL=C sort > "$probe_dir/reused-i2-defined.txt"
cmp native/i2_wave2_defined_symbols.txt "$probe_dir/reused-i2-defined.txt"
for name in required data; do
  env -u ESHKOL_PATH -u ESHKOL_JIT_CACHE_DIR ESHKOL_JIT_CACHE=0 \
    XDG_CACHE_HOME="$probe_dir/link-cache-$name" ESHKOL_LIB_DIR="$PWD/lib" \
    ESHKOL_CXX_COMPILER=/usr/bin/clang++ timeout 45s "$probe_compiler" \
    --strict-types --no-stdlib -I "$PWD/lib" -L "$artifact_dir" \
    --lib eshkol_transformer_wave2 "$probe_dir/$name.esk" -o "$probe_dir/$name" \
    > "$probe_dir/$name.link.log" 2>&1
done
timeout 10s "$probe_dir/required"
```

These temporary paths describe this run, not portable artifact dependencies.
Future implementation still requires its own fresh builds, lifetime/failure
proofs, exact packaging negatives, review and supported CI at the final head.
