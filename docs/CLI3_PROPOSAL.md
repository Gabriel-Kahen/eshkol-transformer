# CLI3 bounded command contract

Status: **bounded integration candidate**. This document selects a separable
first CLI3 slice from the already merged D1, T1, T2, C1, C2, and X1 contracts.
It does not define trainer, evaluator, or generator APIs.

## Decision and executable

Build one AOT hosted-native executable at
`build/cli3/eshkol-transformer`. The first slice contains exactly:

```text
eshkol-transformer tokenizer byte
eshkol-transformer tokenizer train-bpe
eshkol-transformer tokenizer inspect
eshkol-transformer corpus build
eshkol-transformer corpus inspect
eshkol-transformer checkpoint inspect
```

This is useful now: it can create and validate byte/BPE tokenizer artifacts,
turn a bounded set of raw documents into an exact D1 corpus, validate a corpus,
and inspect a complete C2 checkpoint without loading tensors. `pretrain`,
`evaluate`, and `generate` are reserved command words but are not present in the
binary or help until their owners merge. Invoking one in this slice is a usage
error; it must not print a speculative option list or call a diagnostic model.

The executable is Eshkol-native. Python remains permitted only for reference
fixture generation in tests. No command may select a shell, Python, scalar,
alternate checksum, device, dtype, or approximate implementation.

## Command grammar

All options are long options and use `--name value`; booleans are bare flags.
Options may appear in any order after the complete `GROUP COMMAND` prefix.
There are no abbreviations, short-option clusters, environment-derived options,
configuration includes, or implicit current-directory inputs. Unknown options,
missing values, duplicate singleton options, unexpected positionals, and a value
after a boolean flag are usage errors. Each occurrence of a repeatable option is
one option/value pair: `--document PATH --document PATH`, never
`--document PATH...` as one argv value. Occurrence order is preserved.

`--help` must be the sole option at one of three complete parse stages: top
level, after a known group, or after a complete known command. `--version` must
be the sole top-level option. Either exits zero without touching files. A help
flag mixed with any other token is a usage error. Paths are nonempty UTF-8
strings of at most 4,096 bytes with no NUL. `-` has no stdin/stdout meaning in
this slice.

Unsigned integers use shortest ASCII decimal spelling: no sign and no leading
zero except `0`. The parser checks signed-i64 range before applying the narrower
command limit. Repeated options are rejected except the options explicitly
marked repeatable below. The exact version output, including its LF, is:

```text
eshkol-transformer 0.0.1
```

The exact top-level help output, including its final LF, is:

```text
Usage: eshkol-transformer GROUP COMMAND [OPTIONS]

Groups:
  tokenizer   Create and inspect tokenizer artifacts
  corpus      Build and inspect D1 token corpora
  checkpoint  Inspect C2 checkpoints
```

### Tokenizers

```text
eshkol-transformer tokenizer byte
  --config PATH --output PATH [--force]

eshkol-transformer tokenizer train-bpe
  --document PATH [--document PATH]... --maximum-merges N --minimum-frequency N
  --utf8-policy raw|strict [--special NAME=omit|error]...
  [--prefix NAME]... [--suffix NAME]... --output PATH [--force]

eshkol-transformer tokenizer inspect
  --input PATH [--special NAME]...
  [--max-file-bytes N] [--max-metadata-bytes N]
```

The exact tokenizer group help is:

```text
Usage: eshkol-transformer tokenizer COMMAND [OPTIONS]

Commands:
  byte       Create a canonical byte tokenizer
  train-bpe  Train a bounded deterministic BPE tokenizer
  inspect    Validate and inspect a tokenizer artifact
```

The exact command help lines are:

```text
Usage: eshkol-transformer tokenizer byte --config PATH --output PATH [--force]
Usage: eshkol-transformer tokenizer train-bpe --document PATH [--document PATH]... --maximum-merges N --minimum-frequency N --utf8-policy raw|strict [--special NAME=omit|error]... [--prefix NAME]... [--suffix NAME]... --output PATH [--force]
Usage: eshkol-transformer tokenizer inspect --input PATH [--special NAME]... [--max-file-bytes N] [--max-metadata-bytes N]
```

Each command help response is its one applicable line plus LF.

`tokenizer byte` reads at most 16,384 bytes, calls X1 parse and resolution with
an empty override list, validates the result, and requires the T1 rule
`model.vocabulary-size = 256`. It writes the canonical byte-tokenizer artifact.
The full X1 source schema is required; this command does not synthesize a model
configuration or reinterpret a resolved-run manifest.

Each `train-bpe --document` file is one document and one chunk. There must be
1..4,096 documents, and their aggregate bytes must be at most 65,536. Empty
files are valid documents. `maximum-merges` is 0..256 and
`minimum-frequency` is a positive i64. Special names use T2's
`[a-z][a-z0-9._-]{0,63}` grammar. The command rejects duplicate special names,
sorts special specifications by ascending ASCII name for the T2 constructor,
and preserves `prefix` and `suffix` occurrence order. Prefix/suffix names must
name an `omit` special. It neither inserts nor guesses BOS, EOS, or padding.

The output path must not exist unless `--force` is present. Publication uses
C1's same-directory checked atomic writer: no-replace without `--force`, atomic
replacement with it. Rename is the commit point. A reported post-rename sync or
close failure remains an `io` error with `published? = true`; the diagnostic
must say that the artifact is visible with unknown crash durability.

`tokenizer inspect` fully loads and validates the artifact, then reports its
family from the exact fingerprint prefix, fingerprint, and vocabulary size.
Every repeated `--special NAME` is resolved through
`tokenizer-special-token-id`; duplicate requested names are usage errors, two
distinct names resolving to the same integer ID are rejected as corrupt data,
and an absent name is an error rather than a null ID. The output contains each
requested name exactly once in ascending ASCII order. Default file and metadata
limits are 1,048,576 bytes and may only be lowered.

### Corpus

```text
eshkol-transformer corpus build
  --tokenizer PATH --document PATH [--document PATH]... --output-directory DIRECTORY
  --shard-token-limit N

eshkol-transformer corpus inspect
  --input-directory DIRECTORY
  [--max-manifest-bytes N] [--max-shard-bytes N]
  [--max-total-tokens N]
```

The exact corpus group help is:

```text
Usage: eshkol-transformer corpus COMMAND [OPTIONS]

Commands:
  build    Encode documents into a D1 token corpus
  inspect  Validate and inspect a D1 token corpus
```

The exact command help lines are:

```text
Usage: eshkol-transformer corpus build --tokenizer PATH --document PATH [--document PATH]... --output-directory DIRECTORY --shard-token-limit N
Usage: eshkol-transformer corpus inspect --input-directory DIRECTORY [--max-manifest-bytes N] [--max-shard-bytes N] [--max-total-tokens N]
```

Each command help response is its one applicable line plus LF.

`corpus build` loads the tokenizer under the exact 1,048,576-byte T1/T2 policy.
It reads 1..4,096 regular document files with at most 65,536 aggregate bytes,
encodes each document separately through T2's localized i64-le staging path,
and concatenates the token sequences in option order. Thus BPE pairs never cross
documents, and a tokenizer's prefix/suffix applies once per document. D1 stores
only the resulting flat order; it does not preserve document boundaries. The
aggregate output limit is 65,536 token IDs. Staging conversion is the bounded
CPU control-plane list required by D1, not a tensor conversion or numerical
fallback.

`output-directory` must already exist. D1's `.d1-writer-lock`, no-preexisting-
canonical-target rule, shard publication, and manifest-last commit remain
authoritative. There is no `--force`: retrying after a crash requires the D1
operator recovery procedure, not automatic deletion. `shard-token-limit` is a
positive i64 no greater than 65,536.

`corpus inspect` calls the complete D1 validator and reports only its summary.
Defaults are 1,048,576 manifest bytes, 1,048,576 shard bytes, and 65,536 total
tokens; each may be lowered, and total zero retains D1's exact empty-only
meaning. Unreferenced directory entries retain D1's documented limitation and
are not presented as validated artifacts.

### Checkpoint inspection

```text
eshkol-transformer checkpoint inspect
  --input PATH [--max-file-bytes N] [--max-metadata-bytes N]
  [--max-tensor-bytes N] [--max-tensors N]
```

The exact checkpoint group and command help are:

```text
Usage: eshkol-transformer checkpoint COMMAND [OPTIONS]

Commands:
  inspect  Validate and inspect a C2 checkpoint
```

```text
Usage: eshkol-transformer checkpoint inspect --input PATH [--max-file-bytes N] [--max-metadata-bytes N] [--max-tensor-bytes N] [--max-tensors N]
```

The command help response is that one line plus LF.

Defaults are CLI3's measured bounded caller tuple: 16,777,216 file bytes,
524,288 metadata bytes, 8,388,608 bytes per tensor, 64 tensors, and device
`cpu`. Options may lower these limits only. The command constructs the ordinary
public five-argument `persistence-policy` in authenticated C2 `wire` mode; it
never calls the private unverified operational-target constructor. It performs
full C2 envelope, semantic, size, version, and checksum validation via
`checkpoint-inspect`, preserving C2's `corrupt-data` versus `unsupported`
classifications. It does not call `checkpoint-load`, construct tensors,
discover a provider, or claim that the checkpoint can be resumed by a trainer.

## Output and diagnostics

Successful non-help commands write exactly one compact UTF-8 JSON object plus
LF to stdout and nothing to stderr. Keys are in the order shown below; strings
are JSON-escaped, integers are decimal, symbols are emitted as strings, and
versions are two-integer arrays. The executable buffers and completes this
object before its first stdout write.

```json
{"artifact":"tokenizer","family":"byte","fingerprint":"...","vocabulary_size":256}
{"artifact":"token-corpus","shard_count":3,"shard_token_limit":3,"tokenizer_fingerprint":"...","total_shard_bytes":...,"total_tokens":7,"vocabulary_size":263}
{"api_version":"0.1.0-draft","artifact":"checkpoint","checksum_algorithm":"sha256","config_fingerprint":"...","config_schema_version":[1,0],"format_id":"eshkol-training-state","format_version":[1,0],"payload_bytes":24,"required_features":[],"tensor_count":3,"tokenizer_fingerprint":"..."}
```

Tokenizer inspection with requested specials adds a final `specials` object;
names are ascending ASCII regardless of query order. Creation prints the same
tokenizer object as inspection. Corpus creation prints the same summary as
inspection. Parse, domain, publication, and rendering failures occur before the
first stdout write and therefore leave stdout empty. A stdout short write is
retried. A terminal stdout write error is an `io` failure with status 13 and a
diagnostic on working stderr; stdout may then contain the already written
prefix because a process cannot retract it. That reporting failure never
changes a successfully published artifact into an unsuccessful publication:
its diagnostic retains the known publication and durability details.

The diagnostic guarantee begins after the executable has installed its outer
dispatch handler and assumes a working stderr. Runtime startup failure, host
process termination, and stderr-device failure are outside this guarantee.
Within it, every failure writes exactly one ASCII stderr line of at most 4,096
bytes including LF. Stdout is empty except for the explicitly bounded reporting
write-failure case above. General successful output and ordinary diagnostic
rendering may allocate.

Usage errors have the exact form
`eshkol-transformer: usage message=JSON-STRING`. A caught E1 error has the exact
field order `eshkol-transformer: category=JSON-STRING operation=JSON-STRING
message=JSON-STRING details=JSON-OBJECT`. JSON strings escape quote, backslash,
and control characters; every other non-ASCII Unicode scalar is written with
lowercase `\u` escapes, using a surrogate pair when necessary. Invalid UTF-8 is
never copied into a diagnostic. Encoded category and operation strings are
limited to 256 bytes each, message strings to 1,024 bytes (3,072 for a usage
message), and each detail value to 256 bytes. A longer string retains the
longest complete escaped-scalar prefix that permits an ASCII `...` and closing
quote. Detail entries whose encoded key exceeds 256 bytes are omitted.
Truncation therefore never divides an escape or creates duplicate keys.

Detail keys are sorted by ASCII spelling. The scalar allowlist is boolean,
signed-i64 exact integer, character, symbol, and string; characters and symbols
render as JSON strings. Lists, vectors, nested values, non-integral numbers,
and exact integers outside signed-i64 are omitted. Rendering stops before 64
entries or 2,048 encoded detail bytes. Paths and artifact contents are never
inferred from a message.

If normal formatting fails, a private checked nonallocating stderr seam writes
one of these fixed lines from static bytes:

```text
eshkol-transformer: usage: invalid command line
eshkol-transformer: io: publication state unknown; inspect the output path
eshkol-transformer: internal: diagnostic unavailable
```

The first retains status 2. When the already captured failure is in the known
I/O/publication category, the second conservatively reports publication as
unknown and retains status 13. All other formatting failures and foreign
conditions use the third and status 70. Private status capture is permitted for
this fallback, but it adds no public policy or launcher. Stable exit statuses
are:

| Status | Meaning |
|---:|---|
| 0 | success/help/version |
| 2 | command grammar or lexical option error |
| 10 | `invalid-argument`, `shape-mismatch`, `dtype-mismatch`, `device-mismatch`, or `noncontiguous` |
| 11 | `unsupported`, `version-mismatch`, or `determinism-unavailable` |
| 12 | `corrupt-data` |
| 13 | `io` |
| 14 | `invalid-state` |
| 70 | `internal` or a foreign condition |

## Exact API mapping

| Command stage | Merged operation used |
|---|---|
| Bounded config/document reads | pinned `file-size` for the bounded allocation followed by `et_checkpoint_io_read_exact_v1`, which rechecks a no-follow regular file, exact length, EOF, identity, and close |
| Byte configuration | `config-parse`, `config-resolve`, `config-validate`, then `tokenizer-byte` |
| BPE construction | localized `t2-bpe-train-core documents maximum-merges minimum-frequency utf8-policy special-specs prefix-names suffix-names`, followed in the same registry-owning aggregate by T2 core registration and canonical serialization |
| Tokenizer admission/report | `persistence-policy`, `tokenizer-load`, `tokenizer-vocab-size`, `tokenizer-fingerprint`, optional `tokenizer-special-token-id` |
| Tokenizer publication | localized T1/T2 canonical serializer plus `et_checkpoint_io_atomic_write_v1`; this preserves `tokenizer-save!` bytes while adding C1's existing no-replace mode |
| Document encoding | localized `t2-private-tokenizer-encode-staging`; its i64-le values are range-checked before forming D1's bounded list |
| Corpus publication/report | `token-corpus-write!`, then its returned summary and all six `token-corpus-summary-*` accessors |
| Corpus admission/report | `token-corpus-validate` and all six summary accessors |
| Checkpoint admission/report | ordinary five-argument `persistence-policy` in authenticated C2 `wire` mode, `checkpoint-inspect`, and the ten required `checkpoint-metadata-ref` keys |

T2 deliberately has no installed BPE-training or streaming procedure. These calls
therefore do not become library API. The CLI must be a new reviewed
source-composed successor of the C2 root so E1/P1/D1/X1/C1/T1/T2/D2/O2/K2/C2 and
the CLI share one identity universe. It must not link the already localized T2
and C2 archives together. Only the executable entry/dispatch symbols remain
global; all BPE/core/serialization/file seams and constructors are localized.
Every `persistence-policy` value is the current authenticated C2 shell. C1 and
T1/T2 consumers receive only the existing `c2-policy-c1-subpolicy-internal` and
`c2-policy-t2-entry-internal` projections installed by the C2 root; CLI3 adds no
second same-name policy or mutable vector representation.

Pinned Eshkol `command-line`, `display-error`, bytevector ports, and `exit` are
adequate for dispatch, diagnostics, bounded file transfer, and status. Its
`core.argparse` is not used: at commit
`90cbd7130f47b8184bcc77b8d5c1b0026da980de` it treats unknown flags as
positionals, silently drops a nonboolean flag with no value, accepts
`--no-<string-option>`, and converts malformed integers through an unchecked
`string->number`. CLI3 needs the stricter grammar above.

The general `open-input-file`/`read-bytevector` surface is likewise not treated
as checked artifact I/O: reads may be partial and the port API does not supply
C1's same-file size/identity/EOF checks. CLI3 uses the existing checked C1 seam
and maps its stage/status word to E1 `io`; it does not retry through a different
reader.

## End-to-end acceptance fixtures

The focused implementation gate must build the executable twice from fresh
caches, compare its object/executable closure evidence, and run at least these
deterministic cases without Python in the delivered process:

1. Run `tokenizer byte` on X1's minimal 256-vocabulary source. The reported
   fingerprint is
   `sha256:eshkol-byte-tokenizer-v1:aabb31f49216963582383a00fd2e85d7c20b658d5bef0e7945344ed6bfe50704`;
   `tokenizer inspect` reports 256 and the same bytes survive save/load.
2. Train from three files containing `banana banana`, `bandana`, and `banana`,
   with 8 merges, minimum frequency 2, raw policy, specials
   `bos=omit,eos=omit,invalid=error`, prefix `bos`, and suffix `eos`.
   The output exactly equals `tests/t2/fixtures/bpe_tokenizer_v1.tsv` and reports
   fingerprint
   `sha256:eshkol-bpe-tokenizer-v1:1866ebedd76bf7d0e8e111ab25603a99aee927ea341e1305e4bb2c145648eb72`.
   Reversed document option order produces identical tokenizer bytes.
3. Build a corpus from one `banana bandana` document with that tokenizer and
   shard limit 3. Inspection reports 7 tokens and 3 shards, and the existing T2
   D1 reader observes IDs `(260 259 32 257 100 258 261)`.
4. Inspect the deterministic C2 three-tensor fixture. Compare all ten metadata
   fields with the existing C2 public-runtime expectations and prove that no
   tensor/provider callback is reached.
5. Cover empty and exact-limit inputs plus one-over documents, bytes, IDs,
   files, metadata, shards, and tensor counts. Cover malformed argv, invalid
   UTF-8 under `strict`, tokenizer/corpus identity mismatch, corrupt checksum,
   unsupported version/feature/algorithm, stale writer lock, existing output,
   short read/write, close/sync/rename failures, and `published?` diagnostics.
   Every failure has the specified exit status, empty stdout, one bounded stderr
   diagnostic, and no partially published artifact, except that the stdout
   reporting-error fixture may retain its written prefix. Inject an actual
   stdout short write followed by success and an actual stdout write error: the
   former must retry to exact output and status 0; the latter must return status
   13, emit its stderr diagnostic, preserve the published artifact bytes and
   report their known publication state.

## Dependency split and blockers

The six commands above are implemented as one independent CLI3-A workstream. It
changes no public library name or artifact format. `pretrain` remains blocked on
the accepted TR3 trainer/restore API and full trajectory evidence; `evaluate`
remains blocked on E3's accepted public evaluation interface; `generate` remains
blocked on G3's accepted generator interface and sampling/cache evidence. CLI3
must consume those exact merged interfaces later rather than reserve guessed flags
now.
For later pretraining integration, the accepted TR3 stopping limits are deltas for
one invocation; resumed global O2 schedule and checkpoint counters persist. This
proposal intentionally assigns no CLI spelling until the TR3 interface merges.

The isolated runtime-union integration starts at transformer commit `559e318`,
which pins Eshkol `81298b4a` and admits the handler-reserve ABI. The reviewed CLI
series is preserved byte-for-byte except for commit `47d06f9`, which adds that one
measured dependency to the undefined-symbol manifest. Its final package boundary
is 8 global definitions, 2 public exports, 83 public-name strings, 34 Eshkol
sources, 36 formatter-fixture sources, 54 native sources, and 157 undefined
symbols.

The supported Ubuntu 22.04/LLVM 21 gate builds the package from two fresh caches,
compares every output byte, and runs `scripts/test-cli3.sh`. The focused
`scripts/test-cli3-allocation-fallback.sh` gate links the unchanged production CLI
entry and aggregate to the exact production runtime archive. A test-only linker
seam fails the real private-dispatch handler allocation at active depth one; the
outer production guard receives canonical runtime condition 5 and the real static
fallback emits its fixed internal diagnostic with status 70. A second seam denies
vector allocation persistently only after D1 has published shard zero under its
writer lock. The rollback removes the shard, lock, temporaries, and manifest, and
the same production fallback handles the formatting allocation failure. The seam
adds no production hook or alternate launcher and audits its exact four GNU
wrappers and final-runtime link-map bindings.

There is no missing D1/T1/T2/C1/C2/X1 semantic API for this bounded slice. Full
CLI3 remains incomplete until the real trainer, evaluator, and generator APIs and
their end-to-end evidence are accepted.
