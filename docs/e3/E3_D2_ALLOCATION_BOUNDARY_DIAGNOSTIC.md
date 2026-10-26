# D2 read-fault allocation boundary diagnostic

The sealed `2b77481` compiled repair attempt left a +2,288-byte shared-root
delta in direct and mapped D2 faults with flat controls. Its full-E3 pair still
had +2,360 bytes and aborted at the known public-I2 observer mismatch; the
canonical call aborted at `failure-arena-bytes`. These numbers do not identify
an object or equate the 72-byte difference with a hosted exception header.

This noninstalled source successor leaves production D2, E1 and fe9 bytes
unchanged. The private package alone generates fixed-byte overlays for the D2
fault macro and E1 `core-raise`, authenticated from the pinned source hashes.
Both call the same fixed, test-only native hook; its static array samples the
true shared-root arena without allocation. The direct D2 driver arms it only
after 512 completed E3 calls and the same injected exact-read failure. A fresh
control takes the ordinary successful D2 read. Both check typed outcome,
cursor, live batch/borrow, empty dataset escape slot and genuine E3 retry.

The ten ordered samples are: 0 before D2 call; 1 before E1 error construction;
2 after E1 construction/before initial raise; 3 caught by the D2 shard guard
before `region-close`; 4 after `region-close` promotion; 5 after the fault-only
envelope; 6 before re-raise; 7 at the outer E1 catch; 8 after its region closes;
9 after checks and genuine retry. Counts require all ten once in fault mode,
and only 0, 8 and 9 in control mode. Recorded bytes are read and printed after
disarming, so output cannot perturb the measured interval.

Stage deltas locate the first root-arena growth. They do not by themselves
identify an allocation: E1 construction, hosted raise machinery and region
promotion may each allocate. The generated E1 overlay is a distinct source
closure member; its digest is separate from D2's. First gate is exact compiler
import discovery and retained rejected-stage IR/objects. Only after measured
imports and independent review should a new committed manifest successor
admit a focused AOT fault/control run. No full retention acceptance follows
from this diagnostic.

The sealed `b465081` attempt1 authenticated the normal rejected-stage IR and
181-symbol `nm -u` inventory. Relative to the tracked 179-symbol policy,
only `et_e3_retention_boundary_mark_v1` and
`et_e3_retention_boundary_read_v1` were added; no expected symbol was absent.
The normal manifest now matches those measured bytes. The subsequent private
unresolved-authority check admits only these two declared hooks when the
boundary diagnostic selector is active; selector-off, other tuples and any
additional private symbol still reject. Sanitized
noninstrumentation and instrumentation manifests remain unchanged and must be
measured in a separate fail-closed discovery before any AOT or runtime.
