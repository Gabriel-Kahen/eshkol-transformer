# T2 private merge-path performance follow-up

## First pass at `38b023de` (historical baseline for the live pair index)

The public tokenizer API, limits, streamed encoder/decoder, artifact format, and
training tie break were unchanged. Whole encode and training compacted only
newly owned token vectors in place. The scan invariant is `written <= input`:
both operands are read before a match is written, so no unread token is
overwritten, including when the operands are equal. Each training document
retains its own active length; stale vector suffixes never contribute pairs.
Whole encode also counts active token IDs and skips a rank when an operand is
absent. A replacement decreases each operand count by one and creates one
fresh result ID. Equal operands therefore decrease their count twice.
When both operands are present without adjacent matches, whole encode still
scans each rank, so its worst case remains O(input bytes × merge ranks).
The token-vector space is O(input bytes) rather than a new vector on every
rank; training still recounts every document's pairs at each rank.

The development-only workload is `tests/t2/performance_runtime.esk`, measured
by `tests/t2/run_performance.py`. The runner compares every output with the
independent Python T2 reference before recording time/RSS. It records the
actual core, workload, compiler and binary hashes. Compile the workload with
the pinned Eshkol AOT using `--strict-types --no-stdlib -I /workspace`, with
an isolated `XDG_CACHE_HOME`; invoke the runner inside the same container with
`--core-source`, `--source-sha256`, `--workload-source`, `--compiler`,
`--compiler-sha256`, `--image-digest`, `--source-commit`, `--binary`, and
`--output`. The baseline core is the exact `7291ef8` file (SHA-256
`3e3ea26ca39f2449a6a488e9d57d1ed6cdf784434f6272577cbdab619e5b1268`);
the candidate core SHA-256 is
`85ea8edd19460d4225f98ec5e9b9c8e291c95dfa631daf16a5a8a3c59f9e2c3a`.
The baseline workload changes only its load path to the preserved baseline
core; its SHA-256 is `3fcf6c6e4eb73534f3c6d6527ae434cdc6e479163f73ce45e58f43e7f8751cd7`
versus `1ad713563cef9df456b8636f1e668bc6adf15b6f36363ec793aa93039b2fabca`
for the candidate workload. The compiler is pinned `fe9dfd5`, SHA-256
`7dd254bab761fe41142a0e3777338f41b3b9f03a5ce4c0bba419f1e2b22a99aa`.
The container image digest is
`f568a6c2f1f11decd3bdf89034bb95f5a313185851d51c80764c79f12c56394d`.
Runs used no network, a two-CPU quota and a 6 GiB memory limit on a shared
desktop host (Ryzen 7 3700X, 8 cores/16 logical CPUs, Linux
7.2.7-1-cachyos) with Ubuntu 22.04 container userspace.

Medians of three separate AOT process runs, in seconds and KiB:

| Workload | Baseline time/RSS | Candidate time/RSS |
|---|---:|---:|
| 65,536 repeated bytes, 256 ranks, absent operand | 4.00 / 282,252 | 0.90 / 16,204 |
| 65,536 overlapping bytes, 256 ranks | 2.08 / 150,556 | 0.51 / 17,548 |
| 65,536 mixed bytes, all operands present, no adjacency | 4.03 / 280,672 | 4.00 / 18,496 |
| 65,536 bytes in 4,096 documents, 64 training merges | 3.59 / 128,248 | 1.98 / 28,056 |
| 65,536 bytes in one document, 64 training merges | 1.81 / 78,208 | 1.86 / 21,116 |
| 65,536 bytes, zero merges control | 0.91 / 17,480 | 0.87 / 15,588 |
| Three bytes, 256-rank control | 0.12 / 14,920 | 0.13 / 12,732 |

These are whole-process measurements: model construction, input preparation,
and output SHA-256 are included. GNU time provided RSS; a monotonic clock
measured elapsed time around each process. Mixed bytes show no measured speed
gain when all operands remain present. Small controls and the slight one-doc
timing difference are too short or small to support a speed claim. This is a
bounded 65,536-byte implementation; these measurements do not establish
large-corpus throughput or isolated kernel latency. The source and tests keep
the accepted streaming path and exact operational ceilings.

Validation on this source passed 25 Python reference checks and 96 compiled
training/stream checks. The complete T2 invocation passed its runtime,
public, parser, D1 and T1 successor checks, then its boundary section first
stopped at a missing D2 build prerequisite. After building D2 from this exact
tree, the boundary rerun passed both fresh production/D1 aggregate repeats,
byte comparisons, localization, negative binding and public-caller checks,
then hit the nested Wave 1 builder's 300-second default timeout. The same
Wave 1 build passed standalone with the supported 600-second override in
405.12 seconds; the exact Wave 1 and candidate Wave 2 archives then failed
their combined link for the expected duplicate E1 owner symbols. The boundary
script now forwards its configured timeout to that nested builder. These
separate checks cover the remaining assertion, but no uninterrupted full
boundary invocation returned success on this worktree.

## Live pair-index follow-up

Commits `bfd72199` and `006f7145` build a private, bounded index of live adjacent token pairs.
Each node is an original byte position in one document; a merge keeps its left
node, removes its right node, and updates only the affected neighboring edges.
The pair buckets unlink old occurrences before adding new ones, so they do not
retain stale history. A sparse bucket uses a reusable position heap to apply
matches in original left-to-right order; a dense bucket scans original positions
in that order. Both reject candidates invalidated by an overlapping earlier
match. Training maintains eligible pair counts on those same edge updates and
scans only pair slots seen during this invocation for the highest frequency,
breaking ties by ascending numeric `(left ID, right ID)`. No pair crosses a
document tail. The workspace is private; caller chunks and tokenizer inputs
remain unchanged.

The pair-table width is `256 + possible merges`, at most 512, and all node and
candidate vectors are bounded by the admitted 65,536 bytes. Whole encode still
uses the prior exact scan for at most 128 bytes and for uniform-byte inputs;
training uses a small-vector recount for at most 128 bytes. When no merge rule
has both operand byte IDs initially present, whole encode returns its byte IDs
before allocating the index. This is exact because the first successful merge,
if any, must consume
two initial byte IDs. The small-training path can count at most 127 edges, so
its counts cannot approach the i64 limit. The indexed training path still
searches its once-seen pair slots at each rank. A rank taking the dense route
scans all original positions, while sparse ranks visit indexed candidates.
Memory is O(input bytes + possible-token-IDs squared), never an
unbounded occurrence history. The public API, streamed path, limits, rank
order, artifact bytes, and error admission are unchanged.

The new development comparison uses exact prior optimized `38b023de` core
SHA-256 `85ea8edd19460d4225f98ec5e9b9c8e291c95dfa631daf16a5a8a3c59f9e2c3a`
against `006f7145` core SHA-256
`53511fbed8b7c8137f4180e2c1489b875103c6a241d3c744ca1de0cda23901e5`.
Both AOT workloads contain the same nine scenarios; the baseline differs only
in its `load` path. Candidate workload SHA-256 is
`7048f68a4566e43190dcb3deb40462f7d28aaaae860986d25e12bb70bc311a84`;
baseline workload SHA-256 is
`82b5fa66052d7a6386617a7bd1af2ebd0a01ed3b01a6ca5ae1be78860f7c293b`.
The same pinned compiler and container above ran with no network, two CPUs,
6 GiB, isolated caches, and a shared host. Exact compile commands, output
hashes, and raw timings are in `.deps/t2-pair-index` and the preserved local
evidence snapshot. Every one of 54 AOT process outputs matched the independent
Python reference before recording.

Medians of three whole-process runs, seconds / peak RSS KiB:

| Workload | `38b023de` | `006f7145` |
|---|---:|---:|
| 65,536 mixed bytes, both operands present but no adjacency, 256 ranks | 4.017 / 18,120 | 0.942 / 33,724 |
| 65,536 bytes with one 255-rank sparse chain | 3.949 / 17,960 | 0.987 / 31,524 |
| 65,536 bytes in 4,096 documents, 64 training merges | 1.922 / 28,048 | 0.226 / 36,060 |
| 65,536 bytes in one document, 64 training merges | 1.874 / 21,360 | 0.283 / 27,492 |
| 65,536 bytes in two documents, 256 training merges | 3.992 / 19,752 | 0.414 / 38,140 |
| 65,536 uniform bytes, no applicable rank | 0.904 / 18,200 | 0.943 / 17,940 |
| 65,536 overlapping bytes, 256-rank model | 0.568 / 16,936 | 0.505 / 17,216 |
| 65,536 bytes, zero-merge control | 0.907 / 17,440 | 0.934 / 17,952 |
| Three bytes, 256-rank control | 0.129 / 14,940 | 0.119 / 13,176 |

Small control differences are sensitive to shared-host variation. The reported gains are scoped to the
adversarial sparse and measured training workloads; these are not isolated
kernel timings or evidence of throughput beyond the accepted input limits.

The development differential generator compares 125 deterministic training
artifact and encoded-output cases against the independent Python oracle,
including both sides of the 128-byte route, overlapping equal-ID pairs,
newly adjacent later ranks, empty and reordered documents, chunk partitions,
duplicate valid spellings, the 256-byte spelling cutoff, and dense/sparse
index routes. One pinned AOT process passed all 125 checks at 94,152 KiB peak
RSS without a heap warning. The existing 25 Python reference tests also pass.

The first exact-tree D2 aggregate attempt exposed one new pinned runtime
tail-transfer import at its strict undefined-symbol boundary. A private
non-tail return in `006f7145` removed that import without changing the accepted
T1/T2 manifest or public interface; focused AOT disassembly found only the
test workload's own tail-transfer calls, as in the baseline. The canonical T2
Wave 2 object and archive then passed their unchanged exact import, export,
source-closure, and T1/T2 equality checks at `006f7145` in 434.90 seconds,
peaking at 5,754,788 KiB compiler RSS. The T2 object SHA-256 is
`9c6475d545c788ba997bfb4dbebc6c5d95aa9732f228816a6fd75d9a391877a8`;
the archive SHA-256 is
`f5b35809fef3000cd97ab9ad116638d5f680f4826b228020ff6dda8219e85241`.
The exact-tree D2 aggregate also passed its unchanged strict undefined-symbol
manifest in 491.36 seconds (6,350,296 KiB peak compiler RSS), publishing object
SHA-256 `5fd4626a29311e8fd0f10e4fce8efcf0ed0f574001da0a6e0663c51640860157`
and archive SHA-256
`efe5c32eda0a12f8588f7e04c86982ad36703ab09cbca82d012fd0e9865b39fd`.
The uninterrupted full T2 gate then passed at `006f7145` with
`T2_COMPILER_TIMEOUT_SECONDS=900 /usr/bin/bash scripts/test-t2.sh` in the
pinned container. It finished in 3,524.77 seconds with exit status zero and
5,758,172 KiB peak process RSS. Both runtime repetitions, 125-case pair-index
differential, compiled parser and D1 negatives, public T1 compatibility, and
the exact boundary suite completed. The boundary printed its exact 47/41
production and 48/42 D1-test surface pass, deterministic isolated builds,
hostile-path checks, and Wave 1 plus Wave 2 collision pass. Raw stdout,
stderr, timing, and the exact launcher are retained under
`.deps/t2-pair-index/full-t2-no-tail.*` and
`.deps/t2-pair-index/run-full-t2-no-tail.sh`.
