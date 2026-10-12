# E3 N1 synthetic broker candidate

Status: **source-only HOLD**. `scripts/e3_n1_broker.py` is a noninstalled
library with no CLI, service, production pin, or real F layer route. It
implements the separate-UID broker's proposed attempt protocol against tiny
generated source files. It does not establish an actual UID, ACL, ancestor,
mount, `/proc`, inherited-fd, or immutable target boundary.

The broker constructor takes a frozen `Config`: protected parent, reviewed pin
digest, target basename, ordered `Source` descriptors, positive byte/time/RSS
caps, a private receipt key, each source's reviewed resolved path, and the
device/inode/mode chain from filesystem root through the source. The tiny test
fixture freezes these values locally; **no production source identities or
ancestor policy have been reviewed**. `attempt(pin, target)` accepts only that
exact pin and basename. It creates a unique checksummed `HOLD` intent, fsyncs record
and parent before stage creation, opens and compares the stage inode, then
fsyncs an initial stage `HOLD.json` before the first source read. Each source
must be a nonsymlink regular file whose pinned path chain, pre-open `lstat`,
opened `fstat`, and post-read path/fd identity agree; copied bytes must meet
its exact size and SHA-256. A same-byte inode replacement is rejected. The broker retains
the stage fd through no-replace publication, target readback and final sync.

An ordinary copy or no-replace failure keeps the broker-owned partial stage
and checksummed `HOLD.json` with observed sizes and SHA-256 for every present
copied file. Marker/sync/cleanup/retraction faults produce an
`INDETERMINATE` result without a receipt. A failed retraction may leave a
visible broker-owned target. A verified target receives a checksummed
`VERIFIED.json`; only after that and its parent sync does the broker return a
versioned HMAC receipt bound to attempt, pin, target device/inode and exact
file list. The receipt is returned out of band and is **not** stored in the
export; a crash before return has no consumer authority. The protected intent
stays HOLD for recovery and cannot itself promote a target.

`verify_receipt(config, receipt, seal, seal_key)` requires the HMAC receipt and
a separately keyed seal over the same body, then checks the live target inode,
checksummed VERIFIED record and every file's hash/size through nofollow fd
walks. This is a synthetic consumer check. A deployment must separately prove
the seal's read-only custody and key separation; HMAC bytes alone do not make
a writable target immutable. There is no automated recovery promotion.

Before any real acquisition, independently review this source and tests;
freeze the exact release config, source device/inode, resolved path, ancestor
policy and bounds; deploy a
broker under a distinct UID; prove parent and all writable ancestors reject
caller rename/unlink/write, including ACL and mount behavior; prove `/proc`
and inherited fd isolation; and review receipt key custody plus an independent
read-only export seal. The currently available login has UID 1000 and no
passwordless elevation, so the cross-UID gate is untested. Exporter D, real
f31 layers, capture, N1/M1 and the numerical gate remain HOLD.
