# Private G3-R snapshot codec adapter: supported-host gate recipe

Status: **prepared only**. No compiled aggregate, hosted job, or integration
claim follows from this recipe. An independent review must accept this recipe's
source and offline checks before the orchestration root releases a host run.

## Immutable inputs

| Input | Pin |
|---|---|
| Main merge base | `6f6ef9ada4c586573d495c40283ad4b9410d0957` |
| Accepted private adapter HEAD | `615114236150253d2fc04e0129a608ed1ab78bc6` |
| Accepted adapter tree | `eda477c449c520223165644150a6c4fda69d1335` |
| Host | Ubuntu 22.04 x86-64, Clang/LLVM 21.1.8 |
| Eshkol | `toolchain/eshkol.lock` at the accepted adapter tree: `90cbd7130f47b8184bcc77b8d5c1b0026da980de` |
| Oracle Python | 3.14.6, as in the supported hosted acceptance workflow |

The accepted tree includes the exact `tests/g3r/snapshot_codec_adapter_aggregate_test.esk`
digest words and `tests/g3r/check_snapshot_codec_adapter_oracle.py` fixture
generator. Do not regenerate or edit either input. The Python oracle derives
the whole-image SHA-256 from `make_m3_artifact((2, 4))`; the Eshkol fixture's
32-byte digest must continue to match
`c6231d711ca7593630b1940cc790fee1157e96c7ec4bb0b5915476fc9bce6399`
(the 25,410-byte fixture image at the accepted head). This digest is
test-authenticated, not production C2 authority.

## One-use host procedure after release

Use the Ubuntu 22.04 x86-64 package installation and pinned Python setup in
`.github/workflows/acceptance.yml` (pinned checkout, apt.llvm.org key
fingerprint, `llvm-toolchain-jammy-21`, Python 3.14.6). Use a fresh checkout;
do not dispatch CI or use Docker for this gate. Fetch the private adapter branch
only to obtain its commit object, then detach at the immutable commit:

```bash
git fetch origin codex/g3r-snapshot-codec-adapter
git worktree add --detach "$HOME/g3r-adapter-6151142" \
  615114236150253d2fc04e0129a608ed1ab78bc6
mkdir -p "$HOME/.cache/g3r-adapter-gate-tmp"
export TMPDIR="$HOME/.cache/g3r-adapter-gate-tmp"
export CC=clang-21 CXX=clang++-21 LLVM_CONFIG_EXECUTABLE=llvm-config-21
# Put Python 3.14.6 first on PATH, as in the pinned hosted workflow.
scripts/run-g3r-snapshot-codec-adapter-host-gate.sh \
  "$HOME/g3r-adapter-6151142" "$HOME/g3r-adapter-evidence-6151142"
```

Run the last command from this recipe branch's root. The script rejects a
different HEAD, tree, merge base, dirty source checkout, unsupported-host
override, compiler selection, Python version, unavailable task-specific
`TMPDIR`, or reused evidence path before compiling. It invokes the accepted
branch's own `make toolchain` and aggregate runner. That runner performs Q0,
source closure, normal and repeated native/Eshkol execution, ASan+UBSan with
leak detection, independent wire oracle, stdout equality, empty stderr,
feature-off symbol absence, and input hashes. The wrapper records the host,
exit status, source cleanliness, and a SHA-256 manifest even on failure. Keep
all failure evidence; do not rerun into the same evidence path.

The gate passes only if the wrapper exits zero, `run.tsv` records exit status
zero, `post.status` is empty, `SHA256SUMS` verifies, and the independent evidence
review accepts the exact pin and outputs. The orchestration root then decides
whether to proceed to integration. This gate does not prove a production C2
receipt, SAVE writer, file pair, restore, replay, or resumed generation.
