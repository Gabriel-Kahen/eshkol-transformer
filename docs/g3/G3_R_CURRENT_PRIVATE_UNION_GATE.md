# G3-R current-main private provider union gate

This is a bounded test recipe for the reviewed source head
`57a8e80ac6f862696d7c1d0a451c8ce9fc95f320` (tree
`8c5fcf94cf57124931bc9f89c03eec3e8b84e4b8`) above main
`86f4eb36f2fcf079953049d3cdb11377cea5981c`. It is a prerequisite
gate, not a G3-R SAVE writer or continuation admission. The recipe pins the
reviewed provider/toolchain source and refuses to run if it changes.

`TMPDIR=/dev/shm bash scripts/check-g3r-current-private-union.sh --check`
compiles one native archive from the real M3/F32, G3, A2, tokenizer,
sampling, and test-bridge sources. It checks that the M3 fixed14 roster,
F32 witness, and G3 17-word hold APIs are defined together and remain absent
from feature-off objects. This source-only mode does **not** compile or run
the Eshkol caller. It can run on an unsupported host as a compatibility
check; it is not admission evidence.

The compiled Eshkol caller is
`tests/g3r/current_history_private_union_test.esk`, with test bridge
`tests/g3r/current_history_private_union_probe.c`. It constructs an actual
diagnostic C2 M3 model, tokenizer, and G3 generator, commits P1-G1, then
supplies the native owners to the bridge. Under a live G3/A2 hold, the bridge
obtains the sealed fixed14 roster and checks all 14 F32 shape/bit witnesses.
It checks the disjoint 136-byte copy, short destination rejection, alias to
the live M3 owner rejection, wrong F32 and G3 identity rejection, and checks
and ends every acquired witness and hold. A separate genuine post-P1 G1
failure checks pending-history rejection. The existing focused M3 and G3
tests separately cover graph/logits and workspace roster aliases; this
aggregate does not replace them.

The heavy `--run` mode requires an explicit one-use ROOT release and
`G3R_PRIVATE_UNION_RUNTIME_RELEASE=1`. It applies the supported-host, pinned
Eshkol source/binary, and compiler-provenance gates before compiling and
running the Eshkol caller. The released C executor ran this exact gate on
source `c0f747dc29a19890f96afee78c2cea675f5e27f3` with the independently
accepted C7b fe9/f31 toolchain. Its sealed guest stdout is
`G3-R current private union PASS: checks=8`, stderr is empty, and the host
and guest exits are zero. The host evidence is
`g3r-current-private-union-run-20261013-c/host/SHA256SUMS-HOST`
(`a7932101227dda19b6f1c2f58f82c6174149563a93690c0b10e564250a2f1c60`).
The independent raw review is
`g3r-current-private-union-c-receipt-independent-review-20261013-a/SHA256SUMS-REVIEW`
(`bb7905f60311a3e3e14912ea40712f083ef46806c7c0b997f8e0c7c17a8345d4`).
This accepts only the eight-check private union gate. The one-second writable
monitor is a sampling guard, and the evidence does not admit a SAVE writer,
public generation, a continuation coordinator, or replay. Those remain
separate integration gates.
