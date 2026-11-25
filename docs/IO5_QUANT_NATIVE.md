# IO5 native narrow and Q8_0 codec candidate

This bounded candidate copies the C11 ABI, implementation, and native tests
unchanged from `0da7a13e`. Its only test-script change removes the Python
interchange suite, which imports model mapping from a separate unmerged IO5
workstream. It adds no C1, safetensors, GGUF, P1, model, or trainer entry point.

`include/eshkol_transformer/io5_quant_abi.h` defines ABI 1.0. The codec accepts
little-endian byte buffers, including unaligned addresses, for finite F32 to/from
F16 and BF16 and for GGML-shaped Q8_0 blocks (32 values, 34 bytes). Narrowing is
round-to-nearest-even; out-of-range and non-finite values reject. Q8_0 uses an
F16 scale and signed 8-bit values. All bulk calls check lengths, overlap, and
numeric validity before writing output. The implementation allocates no heap
memory. It does not interpret tensor shapes: a later wire converter must verify
GGML row alignment (`ne[0] % 32 == 0`) and its own format and metadata contracts.

Run `CC=clang CXX=clang++ bash scripts/test-io5-quant-native.sh`. The script
checks deterministic `.a`/`.so` rebuilds, symbol imports/exports, C++17 linkage,
C11 numerical and negative cases, allocation guards, and ASan/UBSan/LSan. These
tests do not establish an interchange format, live mixed-precision training,
GPU execution, or MP4 provider compatibility. The full quantized interchange
source at `29ae61cd` still imports `model_mapping`, `gguf_interop`, and
`safetensors_interop`. The reviewed mapping head `e63e4e3a` exposes M3 and
decoder profiles, while that quantized source and its Python tests require a
MOD5 profile from its divergent ancestor. Rebase and profile reconciliation,
then format and native-reader validation, are separate admission gates.
