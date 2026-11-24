# Constrained Decoding / Grammar-Guided Logit Masking Engine (DFA + JSON-Object Schema FSA)

## 1. Overview and Scope

This module delivers a standalone, Eshkol-native constrained decoding engine built around a deterministic byte-level finite automaton (DFA), reverse-BFS co-accessibility pruning, precomputed vocabulary bitset indexing, checksummed binary grammar serialization (`ETCD1000` v1.0), and an Eshkol AOT bridge and driver.

Core files:

- Public C11 / C++17 ABI: `include/eshkol_transformer/constrained_decode_abi.h`
- Core C11 implementation: `native/constrained_decode.c` -> `build/constrained_decode/libeshkol_transformer_constrained_decode.a`
- Eshkol AOT bridge: `native/constrained_decode_eshkol_bridge.h`, `native/constrained_decode_eshkol_bridge.c` -> `build/constrained_decode/libeshkol_transformer_constrained_decode_aot.a`
- Tests:
  - `tests/constrained_decode/header_cd_cpp.cpp` (C++17 header / standard-layout / linkage check)
  - `tests/constrained_decode/test_constrained_decode_native.c` (C11 unit, negative, artifact, and `--wrap=malloc/calloc/realloc/free` zero-allocation hot-path tests)
  - `tests/constrained_decode/test_constrained_decode_oracle.py` (independent Python reverse-BFS, exact bitset mask, and 64-trajectory `json.loads()` oracle)
  - `tests/constrained_decode/test_constrained_decode_aot.esk` (compiled Eshkol AOT driver)
- Build and test scripts: `scripts/build-constrained-decode.sh`, `scripts/test-constrained-decode.sh`

## 2. Automaton and Co-Accessibility Invariants

### 2.1 Byte-Level DFA Representation

A compiled `et_cd_dfa` satisfies the following invariants:

1. **Bounded State Space**: `1 <= num_states <= ET_CD_MAX_STATES (4096)`, `start_state < num_states`.
2. **Dense Byte Transition Table**: `next_state[state * 256 + byte]` is either `ET_CD_DEAD_STATE (0xFFFF)` or a valid state index `< num_states`.
3. **Co-Accessibility (Liveness) Pruning**:
   - At construction (`et_cd_dfa_build_raw`, `et_cd_dfa_compile_literal_choices`, `et_cd_dfa_compile_flat_json_schema`) and deserialization (`et_cd_dfa_parse`), the engine builds a deduplicated reverse adjacency list in CSR form and runs reverse BFS starting from all accepting states (`accept_bitset`).
   - A state `s` is **live** (`live_bitset[s] == 1`) if and only if there exists a finite byte path from `s` to at least one accepting state.
   - Every transition entering a non-live state and every outgoing transition of a non-live state is pruned to `ET_CD_DEAD_STATE`.
   - If `start_state` is not live (empty language / dead-end automaton), construction fails closed with `ET_CD_ERR_EMPTY_LANGUAGE (-3)`.
   - Consequently, at the byte level, no valid transition can ever trap generation in a dead-end state.

### 2.2 Vocabulary Index Precomputation (`et_cd_vocab_index_build`)

Given a co-accessible `et_cd_dfa`, a vocabulary of `V` token byte strings (`1 <= V <= ET_CD_MAX_VOCAB_SIZE = 8192`, each length `<= ET_CD_MAX_TOKEN_BYTES = 256`), and `eos_token_id < V`:

- `mask_words = ceil(V / 64)`.
- For every live state `s` and every token `t in [0, V)`:
  - If `t == eos_token_id`: `t` is allowed if and only if `s` is accepting (`accept_bitset[s] == 1`).
  - Else if `token_lengths[t] == 0`: `t` is disallowed in every state (prevents zero-progress infinite loops from empty special tokens).
  - Else: starting from `cur = s`, walk every byte `b` of `token_bytes[t]` through `dfa->next_state[cur][b]`. Token `t` is allowed if and only if every byte transition lands in a live state (never `ET_CD_DEAD_STATE`), and `dest_state[s][t]` stores the resulting live destination state.
- Results are packed into `allowed_mask[s * mask_words + (t >> 6)]` and `allowed_count[s]`.

### 2.3 Zero-Allocation Hot-Path Contracts

Both runtime hot-path entry points perform **zero heap allocations** (enforced under `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free` in both `-O2` and ASan/UBSan/LSan builds):

- `et_cd_apply_logit_mask_v1(index, state, logits, masked_logits)`:
  - Complexity: `O(V)`.
  - Validates IEEE-754 default rounding/MXCSR environment (`ET_CD_ERR_FP_ENV`), non-null aligned buffers, no partial aliasing (`ET_CD_ERR_ALIASING`; disjoint or exact in-place `logits == masked_logits` are allowed), and valid live `state` (`ET_CD_ERR_STATE`).
  - Validates **all** `V` input logits are finite (`isfinite`) before writing any output (`ET_CD_ERR_NONFINITE`).
  - Fails closed with `ET_CD_ERR_NO_VALID_TOKENS` if `allowed_count[state] == 0`.
  - On any error, `masked_logits` is preserved untouched.
  - On success, allowed logits are copied bit-for-bit (preserving `-0.0f` sign bits and subnormals) and disallowed logits are set to binary32 `-INFINITY` (`0xFF800000`).
- `et_cd_advance_token_v1(index, state, token_id, out_next_state, out_is_finished)`:
  - Complexity: `O(1)`.
  - Validates live `state`, `token_id < V`, and `allowed_mask[state][token_id] == 1` (rejects illegal tokens with `ET_CD_ERR_ILLEGAL_TOKEN` without modifying outputs).
  - On `eos_token_id` from an accepting state: sets `*out_is_finished = 1` and `*out_next_state = ET_CD_DEAD_STATE` (so any subsequent mask/advance call fails closed with `ET_CD_ERR_STATE`).
  - On any non-EOS allowed token: sets `*out_is_finished = 0` and `*out_next_state = dest_state[state][token_id]`.

## 3. Built-In Deterministic Grammar Compilers

1. **`et_cd_dfa_compile_literal_choices`**:
   - Compiles an exact finite union of `1..256` distinct non-empty byte strings (each `<= 256` bytes) into a deterministic prefix trie DFA. Handles shared prefixes (e.g., `"no"` and `"none"`) without nondeterminism.
2. **`et_cd_dfa_compile_flat_json_schema`**:
   - Compiles an ordered compact RFC 8259 JSON object schema of `1..64` fields:
     `{"<k0>":<v0>,"<k1>":<v1>,...,"<k{n-1}>":<v{n-1}>}`
   - Keys must be unique, non-empty printable ASCII (`0x20..0x7E`, excluding `"` and `\`, length `<= 128`).
   - Supported field types:
     - `ET_CD_JSON_STRING`: RFC 8259 JSON string with valid escapes (`\"`, `\\`, `\/`, `\b`, `\f`, `\n`, `\r`, `\t`, `\u[0-9a-fA-F]{4}`), rejecting unescaped control characters (`< 0x20`) and enforcing strict RFC 3629 UTF-8 multi-byte sequences (rejecting lone continuations, overlong encodings, UTF-16 surrogates U+D800..U+DFFF, and code points > U+10FFFF).
     - `ET_CD_JSON_INTEGER`: exact regex `-?(0|[1-9][0-9]*)` (forbids leading zeros such as `01`, `-01`, bare `-`, `+1`, floats).
     - `ET_CD_JSON_BOOLEAN`: `true|false`.
     - `ET_CD_JSON_ENUM`: JSON-escaped deterministic trie over `1..256` unique UTF-8 string choices, closed cleanly by `"`.

## 4. Checksummed Binary Grammar Artifact (`ETCD1000` v1.0)

`et_cd_dfa_serialize`, `et_cd_dfa_parse`, `et_cd_dfa_save`, and `et_cd_dfa_load` implement a deterministic, non-executable, versioned binary format:

- **Header (32 bytes, little-endian)**:
  - `[0..7]`: magic `"ETCD1000"`
  - `[8..9]`: `version_major = 1`, `[10..11]`: `version_minor = 0`
  - `[12..13]`: `num_states`, `[14..15]`: `start_state`, `[16..17]`: `live_state_count`, `[18..19]`: `flags = 0`
  - `[20..23]`: `accept_words = ceil(num_states / 64)`
  - `[24..31]`: `payload_bytes = accept_words * 8 + num_states * 256 * 2`
- **Payload**: `accept_bitset[accept_words]` (`uint64_t` LE, zero padding bits enforced) followed by `next_state[num_states * 256]` (`uint16_t` LE).
- **Trailer (32 bytes)**: SHA-256 digest over Header + Payload.
- **File I/O safety**: `et_cd_dfa_save` writes via `mkstemp` + `fsync` + atomic `rename`; `et_cd_dfa_load` opens with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC` (rejecting symlinks with `ET_CD_ERR_SYMLINK`), verifies `S_ISREG`, verifies SHA-256, and verifies canonical co-accessible pruning on load.

## 5. Verification Evidence

Verified inside container `eshkol-r0-debug:20260928` (`clang-21`, `clang++-21`, pinned Eshkol toolchain):

- **Deterministic archives & symbol audit** (`scripts/build-constrained-decode.sh`): PASS
  - `libeshkol_transformer_constrained_decode.a` SHA-256: `34c3a28426cf73f9174c6012c67384e63dafbf98353d22842791390a55c2b6b6`
  - `libeshkol_transformer_constrained_decode_aot.a` SHA-256: `e298a3aec1f87349061b7572098dc21fd951a44e76e7f5fa24b458e77db8c4dc`
- **C++17 header & standard-layout linkage** (`header_cd_cpp.cpp`): PASS (normal `-O2` and ASan/UBSan/LSan)
- **Native C11 tests with zero-allocation trap** (`test_constrained_decode_native.c` with `-Wl,--wrap=malloc,--wrap=calloc,--wrap=realloc,--wrap=free`): PASS (normal `-O2` and `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`)
- **Independent Python oracle** (`test_constrained_decode_oracle.py`): PASS
  - Verified exact reverse-BFS co-accessibility pruning on raw DFA with dead cycles and branches.
  - Verified exact 64-bit `allowed_mask` bitset parity across all 72 states x 114 tokens (8,208 state-token pairs).
  - Verified 64/64 randomized adversarial-logit trajectories (58 distinct outputs, 2,743 adversarial token blocks) parse via `json.loads()` with exact key order and field types.
- **Compiled Eshkol AOT driver** (`tests/constrained_decode/test_constrained_decode_aot.esk`, `--strict-types --optimize 0 --no-stdlib`): PASS
  - Dual fresh compilations produced byte-identical ELF binaries and byte-identical stdout:
  - AOT ELF SHA-256: `2e50581973ac9cc159b34cf48fcf8108be6ac1505abac6ec3b56dc431bb89fa3`
  - Verified greedy JSON (`{"answer":"yes","score":42,"ok":true}`), adversarial multi-byte JSON (`{"answer":"maybe","score":-42,"ok":false}`, 7/7 steps blocked adversarial spikes), adversarial byte-only JSON (`{"answer":"no","score":0,"ok":true}`, 36/36 steps blocked adversarial spikes), exact enum grammar (`maybe`), `ETCD1000` roundtrip SHA-256, and all negative rejections (`EMPTY_LANGUAGE`, `ILLEGAL_TOKEN`, `NONFINITE` NaN/+Inf with output preservation, `STATE`, `CHECKSUM`).
- **Repository gates**: `tests/q0/test_python_isolation.py` (5/5 OK) and `scripts/check-ci-topology.sh` (PASS).

## 6. Measured Limitations and Unsupported Cases

- **Regular / finite-state languages only**: Nesting of arbitrary depth (general context-free grammars, recursive nested JSON objects/arrays) is outside the DFA model and is not supported by the flat JSON object compiler.
- **Compact canonical JSON**: `et_cd_dfa_compile_flat_json_schema` emits compact JSON with fixed key ordering and no optional inter-token whitespace loops (preventing unbounded whitespace generation under adversarial logits).
- **Hard capacity bounds**: `num_states <= 4096`, `vocab_size <= 8192`, `token_length <= 256`, `num_fields <= 64`, `num_choices <= 256`, `choice_length <= 256`, `key_length <= 128`. Inputs exceeding these bounds fail explicitly with `ET_CD_ERR_LIMIT`.
- **Token-level dead ends with incomplete vocabularies**: Co-accessibility pruning guarantees every live state has a byte-level path to acceptance; if a custom vocabulary omits tokens needed to traverse outgoing bytes from a live state `s`, `et_cd_apply_logit_mask_v1` explicitly fails closed at `s` with `ET_CD_ERR_NO_VALID_TOKENS` rather than silently unmasking invalid tokens.
- **CPU binary32 execution only**: Logit masking operates on IEEE-754 `float` arrays on CPU with default round-to-nearest-even MXCSR/x87 state; non-default FP environments fail explicitly with `ET_CD_ERR_FP_ENV`.
