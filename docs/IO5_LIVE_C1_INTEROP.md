# Live compiled C1 and IO5 interoperability

The live gate starts with a P1 state backed by the production I2 dense
CPU F32 codec and writes it with the Eshkol C1 checkpoint writer linked to
`libeshkol_transformer_checkpoint_io.a`. The bridge generates an M3 fixture in
memory with 15 parameter paths, 14 logical tensors and one head/embedding
alias group. Payloads include negative zero, subnormals, both infinities,
and quiet and signaling NaN payloads. No fixture file is needed to start.

The gate builds the bridge twice with the pinned `eshkol-run --strict-types
--optimize 0 --no-stdlib` compiler and separate build and cache directories.
It compares executable bytes, generated checkpoints and CLI output.
The bridge also accepts `write <output.etcp>` and `verify <input.etcp>`
directly. The verifier obtains a live P1 state through the C1 file reader and
checks the provider identity, ordered roster, parameter kind, dtype, device,
layout, shape, alias topology and exact payload bytes.

The Python harness starts a fresh compiled bridge process to write the
original, invokes each IO5 CLI in both directions and checks that every
restored C1 file equals the original byte for byte. It starts the compiled
bridge again to load and verify each restored checkpoint.

| Route | Selection |
|---|---|
| Safetensors | IO5 synthetic names and manifest |
| GGUF | IO5 synthetic names and manifest |
| Mapped safetensors | `eshkol-m3-v1`, Hugging Face names |
| Mapped GGUF | `eshkol-m3-v1`, GGUF names |
| Sharded safetensors | `eshkol-m3-v1`, Hugging Face names, 4,200-byte shard ceiling |

Each route also rejects a flipped payload byte in the exchange file or a
referenced shard and leaves the rejected C1 destination absent. The bridge
rejects a C1 payload flip with stale checksums and separately rejects
well-formed C1 files with fresh checksums and a changed provider, alias group,
path, kind, shape or payload.

The proof covers C1 1.0 and the supported data-only little-endian F32
wire subsets, using a single M3 logical state. The fixture enters P1 through
owned I2 carriers. Forward and backward model execution, C2 training state,
other devices and dtypes, and native loading by external model runtimes are
outside this gate. The sibling [model mapping tests](IO5_MODEL_MAPPING.md)
cover `eshkol-mod5-v1` names, ranks and metadata independently. All IO5
Python invocations are development and interoperability steps.

## Running the gate

On the supported Ubuntu host with the pinned toolchain available:

```sh
bash scripts/test-io5-live-c1.sh
make test-io5
make test-io5-gguf
make test-python-isolation
bash scripts/check-ci-topology.sh
```

The live script verifies the toolchain and builds the C1 archive under
`build/c1/` by default. It removes temporary bridge and fixture files at
exit. The format test targets discover the live harness and skip its class
when no compiled bridge is configured. The live script runs the harness
directly and requires all live tests to pass.

The supported-container invocation for the pinned toolchain cache is:

```sh
toolchain_cache=/absolute/path/to/eshkol-fe9dfd52-ubuntu22
repo_git=$(git rev-parse --path-format=absolute --git-common-dir)
docker run --rm -v "$PWD:/repo" -w /repo \
  -v "$toolchain_cache:/tc:ro" -v "$repo_git:$repo_git:ro" \
  -e ESHKOL_SOURCE_DIR=/tc/src -e ESHKOL_BUILD_DIR=/tc/build \
  eshkol-r0-debug:20260928 bash scripts/test-io5-live-c1.sh
```

## Validation

The live gate passed on Ubuntu 22.04.5 LTS x86-64 with LLVM and Clang
21.1.8 and Eshkol `fe9dfd5241a1f4c4f58dee8442f44e4ff95e55b9`. The two AOT
executables, original checkpoints and CLI output matched byte for byte.
Each of the five IO5 routes returned the same 10,941-byte checkpoint with
SHA-256 `7c9808cc7daeda1982f1388685a7311f10ab5e6030af3446cd914e0983218461`.
All restored files passed compiled C1 reload. All 12 corruption and
fixture mismatch cases were rejected.

Both format targets passed with safetensors 0.6.2, GGUF 0.19.0 and NumPy
2.5.4. Their separate native GGML witness and compiled C1 class skipped in
format discovery. The dedicated live gate ran the compiled C1 class in full.
Python isolation and CI topology also passed.
