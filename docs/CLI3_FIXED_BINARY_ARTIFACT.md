# CLI3 fixed-profile binary artifact

Status: bounded fixed-profile artifact accepted. PR #233 merged at
`f0f56a8afaa38d28e9e9f4ecc123a3d779f8998e` (tree `3fe5e0e6`). Its
exact-head run `43952346894` and merged-main run `43965845021` each passed
25 required jobs with two intended diagnostic skips. The merged-main run
uploaded artifact `14278541470` (`cli3-fixed-binary-43965845021-1`).
Independent download verified the tar SHA-256
`20382a70354b905655e827c890ebe3e3612cb9e74b349ca0e34c3e866434d87c`,
all 24 inner file hashes, the tested-binary receipt and six source identity
files. The executable SHA-256 is
`2d8599fca65fa234e531c46addea5200d4679bb911ca501f925284ed01ac1b25`.
A fresh, network-disabled Ubuntu 22.04 container ran version, byte tokenizer,
six-byte corpus build, one training update and checkpoint, then greedy `c5`
and categorical `80` one-byte generation; all eight output hashes verified.

The required `cli3-commands` suite builds `eshkol-transformer` through the
production root, checks its package manifests, and exercises tokenizer, corpus,
one-update pretrain, one-byte generate, and cross-command behavior. Both gates
test the same freshly built executable in separate scratch directories. A
standalone cross-command run still builds its own executable. After both gates
pass, `package-cli3-binary.sh` copies **that tested executable**
from the production gate's disk-backed scratch directory into a new artifact
directory outside the checkout. Packaging does not compile another executable.

The production gate hashes its executable before the first command test and
checks that receipt again after its last test. The cross-command gate checks the
same receipt before running commands and before publishing its pass marker.
The packager rechecks the exact
path and digest before copying and compares the installed copy byte for byte.
The artifact carries a path-neutral `metadata/tested-binary.sha256` receipt
for that pre-test digest, both completed gate PASS receipts, and the production
pretrain and generation JSON outputs. It also requires the supported F0
toolchain, clean exact source tree, unchanged package manifests, a real
executable under the recorded production scratch path, and an unused output
path. It records the source commit/tree,
toolchain lock and build provenance, five package manifests, ELF header/dynamic
section and resolved shared libraries. In the new location it starts the copied
binary in fresh processes, checks exact `--version` output, creates a byte
tokenizer from the fixed config, and compares the JSON and tokenizer bytes with
the already passed production gate. It then writes and verifies `SHA256SUMS` for
every artifact file and archives the directory as a deterministic tarball, which
preserves the executable mode across artifact download. The CI suite uploads
the tarball and its SHA-256 receipt only if all required commands and packaging
succeed; upload failure fails the required suite.

On a downloaded artifact, run `sha256sum -c cli3-binary-artifact.tar.sha256`,
extract the tarball, then run `sha256sum -c SHA256SUMS` from the extracted
`cli3-binary-artifact` directory. Run
`sha256sum -c metadata/tested-binary.sha256` there and inspect the two gate
receipts and production outputs. The hosted CI run log remains the authoritative
source for full gate execution. Compare `metadata/identity.tsv`,
`metadata/elf.txt`, and `metadata/dependencies.txt` with the exact supported
run. The binary depends on the recorded Ubuntu 22.04 x86-64 shared-library
environment; the artifact
does not bundle those libraries or claim portability to another host.

This is a fixed CPU-f32, context-two, byte-tokenizer, one-byte diagnostic
binary. It is not a trained SAN release. Held-out exact resume, multi-token and
EOS persistence, evaluator/resource gates, a full trained-model walkthrough,
and CPU-v1 admission remain separate.
