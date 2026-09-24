# TR3-C private snapshot lease authority

Status: **bounded private prerequisite; snapshot composition remains absent**.

The accepted 22-slot trainer lease now gives a future private `trainer-state`
composer one exact exclusion path.  The composer must construct this ten-slot
self-bound parent before admission:

```text
0 tag              5 O2 authority cell
1 self             6 C2 authority cell
2 trainer shell    7 copied controls[5]
3 phase            8 cleanup record[2]
4 P1 authority cell 9 caller result cell
```

All three authority cells and the result cell are distinct one-slot vectors
initialized to `#f`; the copied-control and cleanup vectors are also distinct
and empty.  `tr3-lease-snapshot-enter-internal!` authenticates the trainer,
requires `idle/#f`, rechecks fixed model, D2, gradient, optimizer, and counter
admission, then stores the complete parent in trainer slot 4 before publishing
`snapshotting`.  That first store is the sole promotion root.

`tr3-lease-snapshot-recheck-internal` accepts only the exact canonical parent
and repeats live admission.  Recoverable abort minimally authenticates the
trainer, marks the parent dead, clears slot 4, and restores `idle`.  The success
tail requires an already-dead exact parent and fail-stops on impossible
authority divergence before the same assignment-only unlink.

This leaf adds no second registry, public operation, P1/O2 snapshot, C2 compose,
checkpoint byte, trainer implementation, or exact-resume claim.  Restore parent
identity, phases, and tails are unchanged.
