# CLI3 fixed-profile binary artifact candidate

Status: source and CI wiring candidate. No hosted artifact or installed-binary
acceptance is claimed until its exact head passes supported CI and the downloaded
artifact is independently checked and run.

The required `cli3-commands` suite builds `eshkol-transformer` through the
production root, checks its package manifests, and exercises tokenizer, corpus,
one-update pretrain, one-byte generate, and cross-command behavior. After both
existing gates pass, `package-cli3-binary.sh` copies **that tested executable**
from the production gate's disk-backed scratch directory into a new artifact
directory outside the checkout. Packaging does not compile another executable.

The production gate hashes its executable before the first command test and
checks that receipt again after its last test. The packager rechecks the exact
path and digest before copying and compares the installed copy byte for byte.
It also requires the supported F0 toolchain, clean exact source tree,
unchanged package manifests, a real executable under the recorded production
scratch path, and an unused output path. It records the source commit/tree,
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
`cli3-binary-artifact` directory. Check
`metadata/identity.tsv`, `metadata/elf.txt`, and
`metadata/dependencies.txt` against the exact supported run. The binary depends
on the recorded Ubuntu 22.04 x86-64 shared-library environment; the artifact
does not bundle those libraries or claim portability to another host.

This is a fixed CPU-f32, context-two, byte-tokenizer, one-byte diagnostic
binary. It is not a trained SAN release. Held-out exact resume, multi-token and
EOS persistence, evaluator/resource gates, complete walkthrough, combined
review, and merged-main CPU-v1 admission remain separate.
