# Run a downloaded fixed-profile CLI3 binary

Use the optional `test-cli3-downloaded-artifact` target on Ubuntu 22.04 x86-64.
Point it at a local directory containing only the downloaded `.tar` and
`.tar.sha256` files and a clean checkout of the exact source commit recorded
in the artifact. The command uses no compiler or network access:

```bash
make test-cli3-downloaded-artifact \
  CLI3_DOWNLOAD_DIR=/absolute/path/to/download \
  CLI3_EXPECTED_SOURCE=/absolute/path/to/exact/checkout
```

The verifier checks the outer SHA-256 receipt, the complete archive member
inventory and safe paths, all inner file hashes, the tested executable digest,
both gate receipts, and the source commit, tree, lock, and five package
manifests. It rejects a dirty or different source checkout before extraction.
It then unpacks only regular files and directories into a private temporary
directory and checks their hashes again before launching the binary.

The deterministic fixture checks version, byte tokenizer, DATA4 v2 report
validation and manifest binding, one CPU-f32 training update, checkpoint
inspection, one resumed update equal byte for byte to two uninterrupted
updates, held-out evaluation, and the fixed greedy `c5` one-byte generation.
The scratch directory is removed at exit. `--verify-only` on the Python script
checks seals and identity on other hosts without executing the binary.

This is a developer replay of the fixed context-two profile. The hosted CI log
remains the authority for the original production and cross-command gates.
The artifact does not bundle Ubuntu shared libraries or establish general
training, longer generation, GPU execution, or CPU-v1 admission.
