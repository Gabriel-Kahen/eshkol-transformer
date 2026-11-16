# DATA4-B installed report candidate

Status: command, reader, and focused diagnostic source/tests independently
reviewed. Supported compiled package and command gates, and exact-head ordinary
CI remain pending. This does not admit DATA4 as a complete workstream.

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

On success, `prepare` writes DATA4-A's exact version-1 report bytes to
`DIRECTORY/data4-report.tsv` and prints the existing D1 corpus summary JSON.
The report is written after D1's manifest commit through the already linked C1
checked no-replace atomic writer. A sidecar observed by `file-exists?` is
rejected before D1 publication. That Eshkol predicate may not observe a dangling
symlink; C1's no-replace publication still rejects the symlink but may do so
after D1 commits. A pre-rename report write failure returns CLI3 I/O status 13
with empty stdout and says `corpus published; report not published by this
invocation`. For an ordinary write failure with no preexisting path, the sidecar
is absent; a conflicting preexisting path may remain. A post-rename C1
directory sync/close failure returns status 13 with empty stdout and says
`corpus published; report visible with uncertain durability`. The D1 directory
and sidecar are separate publication units. D1 has no fsync guarantee; no joint
atomicity or crash durability is claimed. A crash between D1 commit and sidecar
publication can leave a valid corpus without its report.

`report-validate` fully validates D1 under CLI3's 1 MiB manifest/shard and
65,536 token limits, then reads at most 32,768 report bytes through the checked
no-follow file reader. It accepts only canonical printable ASCII, tab and LF,
with exact field order/count, decimal spellings, source-ID grammar and
uniqueness, ordinal sequence, filter/duplicate status consistency, bounded
counts and byte totals, exact SHA-256 checksum, and tokenizer fingerprint,
vocabulary, shard limit/count and token count equal to the D1 summary. It
returns the ordinary D1 summary JSON. It does not re-read raw source files,
reconstruct document boundaries, or bind report bytes to a particular D1
manifest digest. A report swapped between different corpora with the same
aggregate D1 summary may validate. The negative gate requires a mismatched
summary swap to fail; manifest-byte binding remains a later reviewed contract.

The required CLI3 production gate exercises installed positive and malformed,
tampered, mismatched-summary, usage, and pre/post-rename C1 failure cases. The
DATA4 source gate exercises the pure report reader with generated report bytes.
The CLI3 package source closure adds only the two DATA4 private Eshkol files;
public exports, defined/undefined-symbol lists, native source closure, and
public-symbol string inventory remain unchanged pending compiled confirmation.
An opt-in `ci.yml` manual mode calls branch-local
`data4-installed-diagnostic.yml` as a non-admission route requiring exact
branch/head/tree inputs. Ordinary PR, push and manual full-suite selection
remain on their existing path; the diagnostic mode skips the ordinary final
gate. The called workflow uses supported Ubuntu 22.04, LLVM 21.1.8 and the
pinned Eshkol toolchain, invokes only the existing CLI3 package builder and
`test-cli3-data4-installed.sh`, and uploads checksummed logs on success or
failure. It does not replace ordinary full CI or its acceptance gate.
