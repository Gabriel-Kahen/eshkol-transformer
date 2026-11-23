# DATA4 installed report

The DATA4-C manifest binding passed the local installed gate from clean
commit `a5f9b38f135fee36028ff633933850334c5a359b` in the pinned Ubuntu
22.04/LLVM 21.1.8 container. The gate took 13:20.06 with 5,791,748 KiB peak
RSS. All 42 source checks and all 13 installed tests passed. The source
closure, native source closure, public strings, package exports, and
undefined symbol manifests matched byte-for-byte. The package's defined
symbol check passed as well. Independent review and hosted validation of
DATA4-C are pending.

The earlier DATA4-B command, reader, and focused diagnostic source/tests
were independently reviewed. [Supported installed diagnostic 43935284566](https://github.com/Gabriel-Kahen/eshkol-transformer/actions/runs/43935284566)
passed at `2e656717`/tree `c99df006`: 32 DATA4 source checks, the exact CLI3
package/manifest gate, and all 9 installed command tests. The downloaded outer
and inner SHA256SUMS verify. The temporary binary was hashed on the runner but
not uploaded as a deliverable. The successor-only CLI3 builder repair and this
installed command slice were integrated through PR #233 at `f0f56a8a`.
Exact-head run `43952346894` and merged-main run `43965845021` passed the
required DATA4 corpus, model-composition and CLI3 command suites. The latter
uploaded fixed-profile binary artifact `14278541470`, whose outer/inner seals
and source identity were independently checked. Complete DATA4 admission
beyond this bounded installed report slice remains pending.

## Commands

```text
eshkol-transformer corpus prepare --tokenizer PATH --source SOURCE_ID=PATH [--source SOURCE_ID=PATH]... --minimum-bytes N --maximum-bytes N --output-directory DIRECTORY --shard-token-limit N
eshkol-transformer corpus report-validate --input-directory DIRECTORY
```

`prepare` accepts 1..128 explicit sources in option order. An ID is 1..64
characters from `[A-Za-z0-9._-]`, unique within the request; `PATH` is the
nonempty remainder after its first `=`. Paths may contain further `=`. Source
files are checked regular files with at most 65,536 aggregate raw bytes. Bounds
are required canonical unsigned decimal integers with
`0 <= minimum-bytes <= maximum-bytes <= 65536`; the shard-token limit is
1..65,536. The output directory must already exist. The command calls the
existing build-only DATA4 prepare/write seam in CLI3's trusted T2/D1 aggregate.
It leaves `corpus build` and the installed public library API unchanged.

On success, `prepare` writes a [version 2.0 report](DATA4_CORPUS_SLICE.md)
to `DIRECTORY/data4-report.tsv` and prints the existing D1 corpus summary JSON.
After D1 commits, CLI3 reads the complete `DIRECTORY/manifest.etm` through the
checked no-follow reader with a 1 MiB limit. It places
`manifest-sha256<TAB><64 lowercase hex>` immediately after `total-tokens`.
The report checksum covers that field. The existing C1 checked no-replace
atomic writer publishes the report. A sidecar observed by `file-exists?` is
rejected before D1 publication. That Eshkol predicate may not observe a dangling
symlink; C1's no-replace publication still rejects the symlink but may do so
after D1 commits. A checked manifest read failure, including a manifest over
1 MiB, or a pre-rename report write failure returns CLI3 I/O status 13 with
empty stdout and says `corpus published; report not published by this invocation`.
A conflicting preexisting sidecar may remain. A post-rename C1
directory sync/close failure returns status 13 with empty stdout and says
`corpus published; report visible with uncertain durability`. The D1 directory
and sidecar are separate publication units. D1 has no fsync guarantee; no joint
atomicity or crash durability is claimed. A crash between D1 commit and sidecar
publication can leave a valid corpus without its report.

`report-validate` fully validates D1 under CLI3's 1 MiB manifest/shard and
65,536 token limits, then reads the manifest and at most 32,768 report bytes
through the checked no-follow file reader. It accepts canonical printable
ASCII, tab and LF, with exact field order/count, decimal spellings, source-ID grammar and
uniqueness, ordinal sequence, filter/duplicate status consistency, bounded
counts and byte totals, exact SHA-256 checksum, and tokenizer fingerprint,
vocabulary, shard limit/count and token count equal to the D1 summary. It
requires the report's manifest digest to match SHA-256 of every byte in the
manifest and returns the ordinary D1 summary JSON. The trusted directory must
stay stable across validation and the subsequent reads. Validation uses the
D1 artifacts and report only. Raw source revalidation and document boundary
reconstruction are outside this contract.

A missing, malformed, or mismatched manifest digest produces `corrupt-data`
with exit 12, including a validly checksummed report swapped between corpora
whose D1 summaries are identical. The reader accepts version 2.0. Legacy
version 1.0 and other unsupported versions produce `version-mismatch` with
exit 11. For a v1.0 report, rerun `corpus prepare` with the original inputs
and options in a fresh directory to produce the bound report. There is no
in-place upgrade.

The required CLI3 production gate exercises installed positive and malformed,
tampered, same-summary swap, legacy-version, usage, post-commit manifest-read,
and pre/post-rename C1 failure cases. The
DATA4 source gate exercises the pure report reader with generated report bytes.
The CLI3 package source closure adds the two DATA4 private Eshkol files and a
successor-only DATA4 command extension. The predecessor CLI3 source and formatter
fixture retain their original command surface. The required package checks in
PR #233 confirmed the existing public exports, defined and undefined symbol
lists, native source closure, and public-symbol string inventory.
The DATA4 successor uses a separate builder so the accepted M3 predecessor
builder retains its exact pinned bytes. Full run `43941165585` reached the
M3 source-pin check and rejected the prior shared-builder edit. Exact-head run
`43952346894` and merged-main run `43965845021` passed with the separated builder.
An opt-in `ci.yml` manual mode calls branch-local
`data4-installed-diagnostic.yml` as a non-admission route requiring exact
branch/head/tree inputs. Ordinary PR, push and manual full-suite selection
remain on their existing path; the diagnostic mode skips the ordinary final
gate. The called workflow uses supported Ubuntu 22.04, LLVM 21.1.8 and the
pinned Eshkol toolchain, invokes only the existing CLI3 package builder and
`test-cli3-data4-installed.sh`, and uploads checksummed logs on success or
failure. It does not replace ordinary full CI or its acceptance gate.
