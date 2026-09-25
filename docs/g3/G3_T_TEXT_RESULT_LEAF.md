# G3-T private detached text result

The accepted private `g3t-generation-output-text/1` accessor authenticates
an exact independently live output and copies its rooted raw bytes into a new
one-element list of a fresh bytevector. The result has length G, zero for
P1/G0 and P2/G0 and one for P1/G1, within the N=1/C2 scope. The caller may
change the returned list or bytevector without changing the output or another
accessor result. A detached copy survives output release and generator close.

The source bytevector was allocated at output reservation, filled by the
accepted same-aggregate T1 raw decoder, and retained at final publication.
The accessor allocates and roots its bytevector and list before copying at
most one byte. It enrolls no new Eshkol or native owner and calls no native
clone. An allocation or copy failure cannot expose a partial result or change
the source output or registry. The focused runner does not inject Eshkol
allocator faults; source order and registry invariance provide the bounded
no-orphan check.

The P2 aggregate witness checks P1/G0, P2/G0 and P1/G1 exact bytes, fresh
list/bytevector identity, caller mutation, survival after parent releases,
and forged/wrong-kind/pending/dead output rejection. This is source-private:
no public G3-G facade, package, N>1 behavior, native API or CLI is added.
