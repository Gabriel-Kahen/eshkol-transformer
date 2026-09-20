# G3 compiled/package review — proposed only

Reviewed merged M3 ancestor `aa9e78f96b5545b52b77c238b92b26c281b41af0`.
This review and the development probes add no production facade, public operation,
ABI, kernel capability, serialization format, or accepted G3 implementation.

## A0 surface and actual reachability

The declaration fixture `tests/fixtures/a0/transformer/generation.esk` re-exports
nine names from the declaration-only `transformer.public` fixture:

| Name | Fixed arity |
|---|---:|
| `generator-create` | 3 |
| `generator-prefill!` | 2 |
| `generator-decode-step!` | 2 |
| `generator-generate!` | 2 |
| `generation-output-ids` | 1 |
| `generation-output-lengths` | 1 |
| `generation-output-text` | 1 |
| `generation-output-rng` | 1 |
| `generation-output-cache-lengths` | 1 |

All nine correctly shaped calls compile against those fixtures; nine zero-argument
negative calls produce the expected exact arity diagnostic. Every fixture body
raises if called. The production-only probe cannot import `transformer.generation`
and verifies the compiler's specific missing-module diagnostic. Declaration
compilation is not runtime implementation evidence.

The compiled production caller imports only the seven installed M3 facades plus
its own source. The existing closure checker verifies this exact eight-file set.
Its undefined symbols contain no private M3/M3T/I2/F32/P1/kernel/E1B authority.
The caller runs the existing Eshkol model schedule through public boxed calls:

- CPU f32 model with N=1, T=C=2, V=256, D=4, Hq=Hkv=2, Dh=2, L=1, F=8;
  learned positions, tied embedding/head, no dropout/cache.
- Input is the M3T-owned `i64[1,2]` diagnostic carrier; output logits own
  `f32[1,2,256]` and expose a detached 2,048-byte observation. This is not the
  generator's required last-token `f32[N,V]` transport.
- T=1 and T=3 input copies reject with `shape-mismatch`; an ID equal to 256 also
  rejects. A subsequent forward matches the earlier logits bytes, showing these
  rejected admissions did not change the input/model for this witness.
- `:rng #f` rejects as `unsupported`; `:cache #f` and `:no-grad? #t` reject as
  `invalid-argument` because they are unknown model options.
- Loss/RNG accessors return `#f`. Eval still builds an owned graph: VJP succeeds
  from a retained logits handle after output release. Eval is therefore not a
  proved no-grad model path. There is no accepted generation RNG ingress here.
- Two executions of the same freshly compiled caller produce identical complete
  logits-byte output. This is one-profile, one-seed repeatability evidence, not
  stochastic generation determinism or repeated fresh-package reproducibility.

Test-only byte comparison/emission is serialization observation; it is not a
scalar implementation of sampling or model numerics.

## Source composition and carrier boundary

The trusted chain is `m3_package_root.esk` → `m3t_package_root.esk` →
`i2_wave2_root.esk` → `t1_wave1_root.esk`, with the M3 schedule/model extension
loaded into that same root. M3's native-object inventory includes N2, N3K and A2
providers. Presence of A2's native cache code in an archive does not furnish an
Eshkol generator/cache carrier or a callable public transport.

M3's admitted source-private seams are localized. Its graph/owned-logits registry,
M3T input/workspace registry, I2/P1 parameter identities, T1 token tensors, and E1
errors are all part of the one aggregate. An independently linked G3 aggregate
plus the existing M3/I2/T1 aggregates would duplicate authority; use one successor
source-composed registry owner. The future package needs exact facade, export,
defined/undefined-symbol, source-closure, native-object, and archive inventories;
private-link and wrong-arity negatives; real installed public AOT callers;
reversed-import identity checks; and duplicate/cross-aggregate link rejection.
No new private extern names or boxed public ABI are frozen by this review.

Required decisions before that successor exists:

1. A bounded no-grad model transport and shape schedule, including all admitted
   prompt lengths, T=1 decode, learned-position offsets, context bounds and a
   last-token `[N,V]` owned result. M3's fixed `[1,2]` roles cannot be relabeled.
2. An explicit whole-model cache owner coupling every layer's A2 cache, model
   parameter/mode identity, position, and prefix lengths with one transaction.
   A2 raw cache pointers are not M3T, I1, I2 or T1 carriers.
3. Reviewed sampling kernels/capability rows and an immutable sampling RNG
   carrier. The diagnostic Philox4x32-10 initializer is not an accepted sampling API.
   No scalar Eshkol loops, dtype casts, CPU fallbacks, or fixture data may bridge
   the missing numerical/carrier operations.
4. Owned generated-token ingress/readout interoperable with the genuine T1
   tokenizer identity, plus explicit lifetimes for generator/output/RNG/cache.
   The A0 generation surface has no release names; bounded reclamation must be
   dispositioned before adding names or claiming steady-state operation.
5. Serialization decisions: A2 caches are non-serializable; current restricted
   M3 facade installation excludes checkpoint/trainer/optimizer facades. Model
   state dictionaries alone do not establish generator/RNG resume. Specify an
   explicit non-serializable first subset or separately review reproducible
   reconstruction and versioned RNG state; do not silently add a checkpoint ABI.

## Reproduction and provenance

`/usr/bin/bash scripts/probe-g3.sh [M3_ARTIFACT_DIR [EVIDENCE_DIR]]` verifies the
pinned compiler/provenance, archive inventories and actual symbols, copies the
installed package into its evidence directory, compiles only development callers,
and runs the fixed M3 witness twice. It never builds prerequisites or runs full
CI. If no M3 package exists, the prerequisite after the repository's normal kernel
prerequisites is `/usr/bin/bash scripts/build-m3.sh`; no such build was needed here.

This run reused the read-only package from worktree `7a9f`, whose current source
HEAD was `10449bbefb2e2061ce5e0d9bbb1b26676e17d618`. All 61 project source/facade files
in its source closures byte-match this worktree; pinned upstream header identity
is covered separately by toolchain verification. This is source correspondence,
not a cryptographic claim that the old object was built from these exact bytes.
The archive's actual `m3_package.o` member matches the adjacent object; its actual
global symbols and member inventory match the checked-in inventories. Seven
companion evidence inventories also match. Archive SHA-256:
`b5cb08b9fcaa4d17b22b08fb0652b194973e84619349c41da6374e248de4eda0`.

The caller object/executable was freshly compiled in this worktree using the
read-only pinned Eshkol `90cbd7130f47b8184bcc77b8d5c1b0026da980de` compiler from
worktree `7fca`; compiler SHA-256
`3e0b923e2e272a89474dff6739b6a2023d71ae483863a64b685ec8ea71ca1cc0`.
The host is CachyOS with LLVM/Clang 22.1.6, explicitly unsupported versus the
Ubuntu 22 / LLVM 21.1.8 acceptance lane. No supported-host result is claimed.

Exact full probe command (the source-root flag preserves the reused-source audit):

```sh
ESHKOL_ALLOW_UNSUPPORTED_HOST=1 \
LLVM_CONFIG_EXECUTABLE=/usr/bin/llvm-config \
ESHKOL_SOURCE_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-src \
ESHKOL_BUILD_DIR=/home/gabe/.codex/worktrees/7fca/eshkol-transformer/.deps/eshkol-build \
G3_PACKAGE_SOURCE_DIR=/home/gabe/.codex/worktrees/7a9f/eshkol-transformer \
/usr/bin/bash scripts/probe-g3.sh \
  /home/gabe/.codex/worktrees/7a9f/eshkol-transformer/build/m3 \
  /home/gabe/.codex/worktrees/df21/eshkol-transformer/build/g3-probe-final
```

The final complete script passed as one invocation with the archive/source guard,
fresh caller compilation, exact public closure, both runtime observations and all
declaration negatives. The guard can also be rerun separately without recompiling:

```sh
python3 tests/probes/g3/check_package.py \
  /home/gabe/.codex/worktrees/df21/eshkol-transformer \
  /home/gabe/.codex/worktrees/7a9f/eshkol-transformer/build/m3 \
  /home/gabe/.codex/worktrees/7a9f/eshkol-transformer \
  /home/gabe/.codex/worktrees/df21/eshkol-transformer/build/g3-probe-final
```

`build/g3-probe-final/` retains compiler logs, exact missing-facade and nine arity
rejections, public depfile, two byte-identical runtime observations, toolchain
provenance, archive/member hashes, actual member/symbol inventories, and
`source-equality.tsv` with both SHA-256 columns for all 61 files. These large
artifacts stay outside Git. `bash -n scripts/probe-g3.sh` also passed.

No cache/no-cache parity, sampling distribution, stochastic RNG advancement,
EOS commit, variable-sequence execution, sanitizer/leak bound, supported-host
CI, fresh-package reproducibility, or generation save/reload test was run; those
remain prerequisites for any G3 implementation acceptance.
