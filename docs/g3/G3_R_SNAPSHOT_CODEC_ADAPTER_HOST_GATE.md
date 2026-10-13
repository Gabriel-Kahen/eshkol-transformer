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
| Recipe checkout | Exact full HEAD/tree in the separately reviewed ROOT release-pin file; its SHA-256 comes from the same immutable handoff. The file is created after this recipe commit is published. |
| Host | Ubuntu 22.04 x86-64, Clang/LLVM 21.1.8 |
| Eshkol toolchain lock | `toolchain/eshkol.lock` at the accepted adapter tree: Git blob `0743c71a1ce749aa70eea3f8cc043148ee4e06be`; file SHA-256 `8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe` |
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
`.github/workflows/full-coverage.yml` (pinned checkout, apt.llvm.org key
fingerprint, `llvm-toolchain-jammy-21`, Python 3.14.6). Use fresh checkouts;
do not dispatch CI or use Docker for this gate. The orchestration root supplies
`G3R_RELEASE_PIN_FILE` and `G3R_RELEASE_PIN_SHA256` from a separately reviewed,
immutable release handoff after the recipe commit is published. The file must
contain exactly `recipe_head<TAB><40 lowercase hex>\n` and
`recipe_tree<TAB><40 lowercase hex>\n`, in that order. Copy the digest from
the handoff; never calculate it from the local recipe checkout at invocation.
The published commit and tree cannot be embedded in their own tracked source.

```bash
: "${G3R_RELEASE_PIN_FILE:?ROOT release-pin file required}"
: "${G3R_RELEASE_PIN_SHA256:?ROOT release-pin digest required}"
printf '%s  %s\n' "$G3R_RELEASE_PIN_SHA256" "$G3R_RELEASE_PIN_FILE" |
  sha256sum --check --status
recipe_head="$(awk -F '\t' '$1 == "recipe_head" {print $2}' "$G3R_RELEASE_PIN_FILE")"
recipe_tree="$(awk -F '\t' '$1 == "recipe_tree" {print $2}' "$G3R_RELEASE_PIN_FILE")"
git fetch origin codex/g3r-snapshot-adapter-gate-recipe-pin
git worktree add --detach "$HOME/g3r-gate-recipe" "$recipe_head"
test "$(git -C "$HOME/g3r-gate-recipe" rev-parse HEAD)" = "$recipe_head"
test "$(git -C "$HOME/g3r-gate-recipe" rev-parse HEAD^{tree})" = "$recipe_tree"
git fetch origin codex/g3r-snapshot-codec-adapter
git worktree add --detach "$HOME/g3r-adapter-6151142" \
  615114236150253d2fc04e0129a608ed1ab78bc6
mkdir -p "$HOME/.cache/g3r-adapter-gate-tmp"
export TMPDIR="$HOME/.cache/g3r-adapter-gate-tmp"
export CC=clang-21 CXX=clang++-21 LLVM_CONFIG_EXECUTABLE=llvm-config-21
# Put Python 3.14.6 first on PATH, as in the pinned hosted workflow.
"$HOME/g3r-gate-recipe/scripts/run-g3r-snapshot-codec-adapter-host-gate.sh" \
  "$HOME/g3r-adapter-6151142" "$HOME/g3r-adapter-evidence-6151142" \
  "$G3R_RELEASE_PIN_FILE" "$G3R_RELEASE_PIN_SHA256"
```

The wrapper checks the release-file digest and exact recipe HEAD/tree before
evidence creation or `make`, then checks the adapter HEAD, tree, merge base,
cleanliness, host settings, and unused evidence path. After evidence creation
and EXIT-trap installation, it copies the release file into the evidence and
seals `run.tsv`, host details, exit status and source status even if a later
step fails. Failures in the earlier preflight create no run.tsv or seal. The
accepted branch's `make toolchain` and aggregate runner perform Q0, source
closure, normal and repeated native/Eshkol execution, ASan+UBSan with leak
detection, independent wire oracle, stdout equality, empty stderr,
feature-off symbol absence, and input hashes. Keep all failure evidence; do
not rerun into the same evidence path.

The gate passes only if the wrapper exits zero, `run.tsv` records exit status
zero, `post.status` is empty, `SHA256SUMS` verifies, and the independent evidence
review accepts the exact pin and outputs. The orchestration root then decides
whether to proceed to integration. This gate does not prove a production C2
receipt, SAVE writer, file pair, restore, replay, or resumed generation.
