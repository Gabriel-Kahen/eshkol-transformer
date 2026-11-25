# AC4 native foundation extraction gate

The standalone AC4 implementation is preserved at `8ae54793` plus the
floating-point guard at `c020e8d9` (tree
`99c79d09bed1724726829e2203343d4661d850be`). Those commits cannot yet be
extracted alone onto main `8d185344`: AC4 includes
`mod5_primitives_abi.h`, calls `et_mod5_kernel_provider_v1()`, and its test
script invokes `scripts/build-mod5.sh`. Main has none of that MOD5 header,
provider, or build script. The historical parent `9a816504` has all three.
The next integration gate is a separately admitted MOD5 provider contract,
followed by an exact combined-tree AC4 build and test. No AC4 files are
copied into main by this evidence-only change.

## Reproduce the standalone gate

The historical candidate is independently runnable without merging its
stack. The supported image below is the Clang 21.1.8 image with digest
`sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6`.
Use an allowed logical CPU in place of `0` if the host's affinity differs.
The pinned image runs under rootless Docker here, so container UID 0 maps to
the host owner of the worktree.
Run from the repository root. The subshell removes its detached worktree on
exit, including a failing test run.

```bash
(
  set -e
  ac4_tree=$(mktemp -d /tmp/eshkol-ac4-c020.XXXXXX)
  trap 'git worktree remove --force "$ac4_tree"' EXIT
  git worktree add --detach "$ac4_tree" c020e8d9769632fba9124663f54e67320541637e
  flock -x /home/gabe/.cache/eshkol-transformer-orchestration-build.lock \
    docker run --rm --network none --cpuset-cpus=0 --user 0:0 \
    -v "$ac4_tree":/work -w /work \
    eshkol-checked-promotion-llvm21@sha256:f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6 \
    bash -lc 'CC=clang-21 CXX=clang++-21 AC4_ASAN_DETECT_LEAKS=1 bash scripts/test-ac4.sh'
)
```

The test script builds the C11 archive, audits its exports/imports, links a
C++17 caller, runs optimized numerical/shape/replay/negative tests, reports
arena peaks and median microbenchmarks, then runs ASan/UBSan with leak
detection. The original independent source-and-test review requested one P1
repair on `8ae54793`: unsupported floating-point controls could trap before
rejection. The focused re-review passed `c020e8d9` after integer-bit config
validation and x87/MXCSR guards closed that case. This approval applies to the
historical MOD5 stack. The external exact-tree Clang 21 log is
`/home/gabe/.codex/evidence/eshkol-transformer/ac4-c020-clang21-20261125/test-ac4.log`
(SHA-256 `233baab7cb6850b9d392d3846ce4f0cffdefdd2e52bb5b97ad6ec2706ed03183`).
It reports 5,440 central differences, maximum absolute error
`8.10525438e-06` under the test's mixed absolute/relative criterion, exact
policy parity through eight caller-side SGD updates at 4, 8, and 16 layers,
and optimized plus ASan/UBSan/LSan passes. This receipt qualifies the
historical tree, not an AC4-only extraction onto main.

## Memory and timing interpretation

For the receipt's 16-layer GQA shape (`N=2`, `T=3`, `Hq=4`, `Hkv=2`,
`Dh=2`), `NONE` reserved 31,872 activation-arena bytes and 57,472 total
workspace bytes. `EVERY_K` with `k=16` reserved 3,264 activation-arena bytes
(89.76% less) and 28,864 workspace bytes (49.78% less), at 136 replay
forwards and 4.337 times the measured call time. Each policy staged 25,600
parameter-gradient bytes. These are byte counts from AC4's own arena and
workspace, not process RSS or end-to-end trainer memory. Caller tensors,
provider-local scratch, stack storage, and allocator effects are excluded.
The receipt's timings are medians on one logical CPU for small fixed shapes;
they are not a throughput or scaling claim.

The foundation supplies only the listed CPU binary32 shapes, a half-mean-
squared-error objective, and caller-side SGD smoke updates. It has no public
TR3 trainer integration, optimizer/resume proof, GPU path, or mixed-precision
contract. Those remain separate gates.
