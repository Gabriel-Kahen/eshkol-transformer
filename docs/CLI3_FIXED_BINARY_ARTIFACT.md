# CLI3 fixed-profile binary artifact

## Current accepted scope

The current exact-main tested-binary artifact is `15073929426`
(`cli3-fixed-binary-45723832757-1`, GitHub artifact digest
`sha256:60874332df23d738296cac360fae5e8e4daba4c8a82919801199e97dddc107ee`)
from [PR #247 run 45723832757](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45723832757).
Its source head is `b7a7c8ceb8bd23f17ad3086cece3d62f929b24fb`, tree
`cd85d8f2ba44776f416a23c164b5a3884499fddc`, and merge commit
`c96697efb4152dc0c70dd007cd31ef81c17612fc`. All 26 required hosted
jobs passed. This artifact is identified from the CI upload metadata; the
independent downloaded-binary walkthroughs below used earlier exact trees.

The latest independently downloaded DATA4 v2 walkthrough used the exact tested-tree
[PR #245 run 45718652932](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45718652932)
artifact `15071319148` (`cli3-fixed-binary-45718652932-1`, GitHub artifact
digest `sha256:c3252df4bba3520a69cef6a851cdf41348c9585de38af5fd9023307aa5dbc91c`).
Its source head is `6e1049a957103da743052980d347de0264c96075`, tree
`76e6128de9aedcee6593a1e964d224f56e842d49`, merged as `5744dc44`.
All 26 required jobs passed; [merged-main run 45729653136](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45729653136)
reused exact-tree evidence. Independent downloaded-binary validation in a
network-disabled Ubuntu 22.04 container passed DATA4 v2 binding to the exact
D1 manifest bytes, same-summary report-swap rejection, one training update,
evaluation and one-byte generation. Legacy DATA4 v1 reports require regeneration.

The prior [PR #243 run 45702533296](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45702533296)
uploaded artifact `15065156707` (`cli3-fixed-binary-45702533296-1`, GitHub
artifact digest
`sha256:1df8f17fd8aa97d674968dd0d813bf4fc23dec529e1ae974cca43b9a3483da36`).
Its exact head is `e63e4e3a835bc9ae31d2694186e61cf4aab5bdba`, tree
`4478b10ee9e84caa75d878c0cbd57622312b1fed`, merged as `417918d0`;
[merged-main run 45717837006](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/45717837006)
reused exact-tree evidence. Independent download verified the outer and inner
checksums, tested-binary receipt and source identity. The same binary repeated
the four-update witness in a fresh network-disabled Ubuntu 22.04 container:
uninterrupted four updates, one-plus-three resumed updates and chained
one-update resumes produced byte-identical C2 checkpoints. This is a fixed
CPU-f32, context-two, byte-tokenizer profile. Multi-token/EOS generation,
general trained-model resume and CPU-v1 admission remain open.

## Historical one-update artifact

The original bounded fixed-profile artifact from PR #233 merged at
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

At that historical gate, this was a fixed CPU-f32, context-two, byte-tokenizer,
one-byte diagnostic binary with a one-update walkthrough. It was not a trained
SAN release. The later PR #243 four-update witness supersedes that resume
limit for the same fixed profile; multi-token and EOS persistence, a full
trained-model walkthrough, and CPU-v1 admission remain separate.
