This is a direct, uninstalled CPU4 E3 consumer fixture. It uses the actual C4
constructor, typed binding/lease, P1 fixed set and modes, D2 dataset/capabilities,
CPU4 E3 adapter/frame and reviewed native providers. It adds no production hook.

The recipe deliberately builds every native TU with candidate mode OFF (64-byte
parameter control), including the exact owner/pin/training/release features.
It does not establish compatibility with the eventual candidate-ON aggregate.
The source list contains one F32, I1 and D2 registry; the helper is appended once.

The canonical raw/empty byte tokenizer is V256. Development resources derive its
identity independently with the existing tokenizer reference; the old tokenizer
file with three specials is V259 and is not used here. One batch is skipped from
13 tokens, leaving two T4 batches/eight physical tokens. Runtime observations
serialize all four metric uint32 words and both counts and compare exact bytes
across repeats and normal/generated-ASan execution. The accepted native frame
oracle remains the numerical authority; this fixture adds consumer transaction
coverage and does not claim a new gradient or numerical oracle.

The fixture checks all 17 modes, exact encoded cursor/metadata identities,
nonzero logical restoration with increasing generations/stale old capability,
model values, all gradient metadata, and varied finite nonzero present-gradient
payloads. A deliberately zeroed expected gradient word must fail the actual
snapshot comparison before the original witness is restored. Absent
internal gradient backing is not exposed by this candidate-OFF tuple. It tests
malformed admission, actual second-bind rejection in idle/prepared/eval states,
same-frame prepared contention and native READ contention, controlled
partial prepare/enter/borrow cleanup, strict binding checks while READ remains,
late binding denial and fourth-destination publication denial, prior report
retention, and same-bound retries. Manual partial stages call existing internal
operations; the ordinary success/retry path calls e3-evaluate-into!. The supported
P1 boundary is one enrolled E3 frame per model. After genuine unbind/destruction,
a sequential frame rebind uses the same still-live binding and produces the same
report. The defensive source registry guard does not establish support for two
simultaneously bound frames.

Source-only checks:

    python3 -B -m unittest -v tests.e3_cpu4_consumer.test_source
    bash -n scripts/test-e3-cpu4-consumer.sh

The full host source suite requires genuine Guile. Since the supported f31 image
has no Guile, the primary generates an immutable source-bound reader receipt on
the host after the final clean commit:

    python3 -B -m tests.e3_cpu4_consumer.host_reader create /absolute/evidence/host-reader.json

The receipt binds exact HEAD/tree, the five files actually read, test and receipt
code hashes, host reader binary/version, exact arguments and successful empty
output. Keep the receipt beside the supervisor package, leaving its `out/` absent.
Mount the receipt read-only at `/host-reader.json` and explicitly set guest
`CPU4_CONSUMER_HOST_READER_RECEIPT=/host-reader.json` in the invocation. Before native compilation the guest validates it and
explicitly reports `HOST GUILE VERIFIED / GUEST GUILE NOT RUN`. Only that reader
obligation uses the receipt; all other source checks still execute in the guest.
Head/tree/file/code mismatches, failed reader results and malformed receipts are
rejected. No silent skip, image installation or changed compiler is permitted.

Only the primary orchestrator launches the supported-container gate under the
global heavy lock, with one compilation job and a 2400-second supervisor budget:

    bash scripts/test-e3-cpu4-consumer.sh /out/consumer

The recipe retains raw IR/bitcode/depfile, applies the pinned streaming ASan helper
to that same IR, proves generated object address-check imports, instruments all
native TUs, checks runtime diagnostics and seals success/failure outputs. A
source-test pass is not native acceptance. Native execution is NOT RUN until an
actual compiled receipt and independent result review say otherwise.
