# E3 N1 exact OCI and capture-input plan

Status: **inert input specification; no reviewed plan SHA, scan, capture, N1/M1
admission, or Numerical F rerun**. This document starts at accepted gzip-scanner
C `3796c7bbca22d9a934cbabdb42b3682f7897d35a`/tree
`f94a9863dbc4663263548fb0be1a6529b51b58e0`. It is a conditional plan,
not a `schema: 1` preflight input: the image-selected guest bytes and final
capture recipe are unknown. `REVIEWED_PLAN_SHA256` in
`tests/e3_private/e3-n1-raw-capture.py` stays `None`; `--capture` remains disabled.

## Immutable identity chain

The read-only seal
`/home/gabe/.codex/evidence/eshkol-transformer/e3-section9-n1-local-index-host-pins-20261011-a/SHA256SUMS-ACQUISITION`
(`feec372c6bd36787be0f30aea2f7ce9ad4ae969bf5bc26c42238349930969ff3`)
authenticates these three copied small JSON blobs. Its `INVENTORY.json`
records the local layer **descriptors and sizes**, not their verified bytes.
Before any layer read, seal a **metadata-only acquisition input**: the exact
three small JSON bytes, the two ordered expected descriptors, each local
content-store candidate's absolute/resolved path, `lstat` type/mode, size and
file identity, the fresh export destination and measured free space. A path or
matching size is a candidate, not proof of a layer digest. Independently and
ROOT-review a separate bounded export-acquisition procedure and those pins
before it opens either gzip blob. The procedure must stream each source once
into a private temporary regular file, count and hash bytes against its ordered
descriptor, reject source identity/size drift and symlinks, enforce per-blob
512 MiB and combined 512 MiB byte caps plus positive time/RSS/disk caps,
and publish the export only after both exact hashes and sizes pass. The
[export custody contract candidate](E3_N1_EXPORT_CUSTODY_CONTRACT.md) requires
a separate-UID protected parent, a durable broker-owned pre-stage HOLD intent,
and a stage marker synced before any layer read. On ordinary failure, preserve
and mark partial bytes when storage permits. If marker write/sync, cleanup, or
post-rename retraction fails, keep the attempt INDETERMINATE and broker-owned;
a visible target is quarantined, never admitted. Only an authenticated
post-check/post-sync VERIFIED receipt bound to the exact target identity,
together with an independently verified read-only export seal, can release an
export. Directory presence or readable VERIFIED bytes cannot release it. This
corrects the previous literal promise of a marker beside every partial file
and no visible target after arbitrary storage failure, which cannot be
guaranteed. The broker protocol, source stability, receipt, custody, and
publication must be independently and ROOT-reviewed; this document supplies
no executor or permission to start it.
It must not invoke Docker, extract a layer, execute a guest file or use the
historical F `/out` route.

The two currently reported source candidates are
`/home/gabe/.local/share/docker/containerd/daemon/io.containerd.content.v1.content/blobs/sha256/<layer digest>`
using the ordered digests below. The small inventory reports presence and
descriptor-matching sizes; it does not record verified layer hashes or the
full pre-read file identity required above.

Only **after that separately reviewed acquisition** can a full OCI export be
sealed and reviewed. It must have a regular, non-symlink `oci-layout` containing
`{"imageLayoutVersion":"1.0.0"}`, a regular `index.json`, and regular blobs
at `blobs/sha256/<digest>` on a read-only, separately sealed path. Freeze
absolute export path, resolved path, type, mode, size, SHA-256, creation method
and evidence seal; do not use the containerd content store as an implicit OCI
layout. Compare exact raw bytes, not a reserialized JSON equivalent.

| Ordered object | Required digest | Descriptor size | State |
| --- | --- | ---: | --- |
| `index.json`, F image ID | `f31d1db76958339e6ebd2a2f667052cdb85aeb5229914ffb10ac4fcdc6db22e6` | 855 | small bytes sealed |
| sole `linux/amd64` manifest | `3649921cfd39ef9b2c4a54d0c6ca01cd6fc49768806297fe61e05a70648804fb` | 676 | small bytes sealed |
| manifest config | `584fc09fcf7a84e381d8ef1f2903143b7b1debbbd5661468ed31d56863087ee2` | 2,326 | small bytes sealed |
| layer 0, OCI gzip tar | `20c3783cc497b5b0df1fc5f92bd64c3d2fbb24057c88692c8fda205b6ea8a2f2` | 29,750,251 | descriptor only |
| layer 1, OCI gzip tar | `399066089356a53b7c7046be9a8a2e938647e6a4076724bf42034912ea6ace98` | 449,869,772 | descriptor only |

Require the index's unique exact `linux/amd64` descriptor, then its manifest's
exact config and ordered layer descriptors. The config must say `linux/amd64`
and carry ordered diff IDs
`sha256:ea16cace89338c84eb6bcb91a7efdfcae6838fff359efe951858227436486c34`
and
`sha256:5ef3a0f7c37c1e91eaf25f50ed9c318abd80e417aef95d9352435bb165eeb458`.
These diff IDs are **claims in sealed config bytes**, pending verification by
decompression. The second index entry is an `unknown/unknown` attestation
descriptor; never substitute it for the amd64 manifest. Reject any changed
identity, media type, descriptor order, platform, size, digest, missing blob,
symlinked blob or extra gzip member/trailing bytes. Never infer F's image ID
from the config digest.

## Selected image bytes and boundaries

Before a real scan, a reviewer must derive a finite **path candidate** table
from the exact failed F commands, retained objects, source/depfile routes and
image metadata. The final pinned selection requires every row's `role`,
absolute guest `path`, final layer number,
`kind` (`file` or `symlink`), mode, and either file SHA-256 plus size or exact
symlink target. All intermediate symlink targets must be separately selected.
The table must include:

* guest `readelf`, `objdump` and `addr2line` or their exact chosen equivalents
  for relocations, final lowering and debug locations; each command's `tool`
  and `argv[0]` must name a pinned selected guest path;
* the exact guest interpreter, chosen loader path and every provider candidate
  that could resolve `bcmp` or `puts`, including their symlink chains, ELF
  dependencies and loader configuration/search inputs;
* compiler, linker and resource-directory tools actually needed for any
  parity replay, plus complete direct/transitive system-header candidates;
  F's four host Eshkol header copies are mount-derived candidates only;
* files needed to prove `private.o`, `package-native-5.o` and
  `package-native-6.o` producer/ABI declarations and all 39 call sites.

The historical image config supplies `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
and `DEBIAN_FRONTEND=noninteractive`; it does not identify resolved tool or
provider files. The sealed host inventory's `/usr/bin/readelf`,
`/usr/bin/objdump`, `/usr/bin/addr2line`, `/usr/bin/python3.14`,
`/usr/lib/ld-linux-x86-64.so.2` and `/usr/lib/libc.so.6` are **host-only
candidates**, never guest pins. Do not copy their hashes into guest rows.
Image-selected guest paths/hashes must come from the reviewed export through
a reviewed bounded read, and a runtime loader map requires a separately
reviewed actual observation.
The existing C scanner compares against already pinned selected-entry hashes;
it has no discovery mode. None of those hashes is in the small seal. If no
independent selected-entry inventory exists, first review a separate bounded,
read-only discovery procedure and its resource limits with the independent
reviewer and root. That discovery **is a real layer read**, and cannot start
under this plan alone. Seal its output, then review and freeze the exact
selection before using C as a verifier. The production preflight also has no
reviewed plan pin and requires a raw receipt; a scan-only invocation needs its
own reviewed source/procedure rather than bypassing those gates informally.

The C scanner reads gzip layers in order, checks each compressed SHA-256/size
and expanded diff ID, and compares final selected entries. It rejects selected
whiteouts, hardlinks/unsupported selected entry kinds, escaping or unselected
symlink targets, mode/hash/size drift, and resource-limit crossing. This is
only a selected-path check: it does not reconstruct arbitrary overlay state,
prove parent-directory symlink semantics, or prove runtime loader resolution.
Reject an affected path if an opaque directory, deletion, nonselected parent
link, hardlink, duplicate/ambiguous route, or mount shadowing prevents exact
selection; expand the table and review the scanner before proceeding. Preserve
raw tar path spelling and layer order in the seal. Do not execute extracted
files during selection.

## Host routes, recipe and bounded resources

Retain F's six exact read-only mount routes in `validate_routes`, with root
`lstat` kind/mode, symlink target if any, and resolved absolute path. The
first two Python mount routes share one host runtime root; there are five
unique input roots. Replace F's historical seventh `/out` host with one fresh,
private, empty absolute output directory, mounted read-write at `/out` only.
Pin its parent and require at least twice the output cap free. Record selected
host tool/interpreter/loader file mode, resolved path, size and SHA-256
separately from guest rows; changes to either side HOLD. Do not treat mounted
headers or a host libc as image-selected guest files.

Proposed upper limits for a later reviewed `schema: 1` plan are: total
preflight 1,800 seconds; each compressed blob 512 MiB (above the 449,869,772
byte largest descriptor); each expanded layer 8 GiB; all expanded layers
16 GiB; selected file bytes read across layers 256 MiB; each raw record 2 GiB; total
output 2 GiB; peak process RSS 2 GiB. The source enforces ceilings at or
above these proposed values, but no real-layer workload has measured them.
If the finite selection or measured resources exceed a cap, HOLD and revise
through source review; never silently raise it or accept truncated output.
The future runner also needs separately enforced per-command time/output/RSS,
network isolation, no target execution, and a sealed partial-failure record.

Freeze exact direct `argv` arrays, environment maps, output paths and positive
per-command caps for all eight preflight categories: call relocations, final
producer, ABI/declarations, provider ELF, provider symlinks, loader observation,
system headers, and argv/environment. The current source validates their
shape, not their semantics or execution. F's `launch.sh` proves only the outer
Docker invocation and selected environment; it does not supply per-object
compiler argv/env. A replay claimed to be F must reproduce the three retained
F object hashes and relevant depfile hashes; otherwise classify it as a new
candidate. NUL-delimited argv/env, complete system-header trace, provider
closure, actual loader map and all 39 per-call producer/ABI rows require raw
records and independent semantic review.

## Admission sequence and stop rules

1. Scanner C has independent and ROOT source-only acceptance on its exact
   head/tree. Seal the metadata-only source-path/descriptor candidates,
   destination and resource measurements. Independently and ROOT-review the
   broker custody contract, deployed separate-UID authority, exact bounded
   export-acquisition source and pins **before the first real gzip read**.
   This review is still pending; exporter D remains HOLD.
2. Only under that reviewed procedure, stream/copy/hash the two layers, reject
   drift and cap breaches, then require the authenticated VERIFIED receipt and
   independent read-only seal before export admission. Review the actual
   path/mode/size/hash chain. Freeze finite guest path candidates,
   host routes and a proposed capture recipe. The full export's verified
   layer hashes are post-acquisition evidence, not a pre-read prerequisite.
3. Independently and ROOT-review bounded guest-entry discovery if hashes are
   still absent. After that review, acquire and seal selected guest-entry
   hashes; review and freeze the complete selection and a scan-only invocation.
   Then use C to verify the
   index→manifest→config→ordered gzip chain, compressed/expanded identities
   and selected entries. Review overlay/symlink disposition. Any mismatch,
   unsupported format, cap hit or ambiguous selection is HOLD; no partial
   scan is admission.
4. Independently review a capture runner and its actual guest tool/loader
   binding before any Docker, loader or provider use. Capture the raw 39-call,
   provider, header and argv records under that runner; verify F object parity
   and seal every raw path/hash/size. A loader trace from host tools alone is
   insufficient.
5. Independently review semantic producer, ABI, declaration, provider,
   version, header and loader claims and pin their immutable claims hash in a
   separate verifier source change. Only then consider a tuple-scoped import
   policy change and a fresh full Numerical F gate. N1/M1 require their own
   acceptance verdicts.

At this document's head, steps 1–5 remain conditional. There is no accepted
broker custody contract, deployed broker, reviewed export-acquisition
procedure, authenticated receipt, approved OCI export path, reviewed plan hash,
complete guest selection, real-layer scan, reviewed discovery/scan-only
procedure, capture runner or raw image-specific receipt. Stop at the first absent or
changed pin, ambiguous file/overlay/loader path, unproven F parity or resource
breach. Leave F evidence and scanner C's worktree immutable.
