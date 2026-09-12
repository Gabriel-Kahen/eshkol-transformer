# Private C2 checkpoint save

`c2-checkpoint-save-internal!` is a source-composed C2-private operation. It is
not an A0/K2 API and provides no load, reconstruction, or capability surface.

The operation admits options, required features, path, policy, and the exact C2
owner before filesystem work. It borrows that owner synchronously, revalidates
and detaches its inert controls, uses the committed claimed-owner C1 model and
O2 staging seams, then assembles with `et_c2_checkpoint_encode_v1`. The native
bridge recomputes outer and moment digests and requires a complete
`et_c2_checkpoint_parse_v1` load-mode parse before returning the image. The C2
borrow ends before the existing C1 atomic writer publishes it. Atomic writer
errors retain their `published?` and `durability` fields.

Authenticated P1/O2 work runs in the enclosing arena because its exact access
identities must not cross a regional write barrier. Only inert outer checkpoint
measure, destination allocation, encode, and full-parser scratch run in the
64 KiB-hinted lexical region. The final image is promoted on region escape.
Thus peak memory can include detached controls, C1 and O2 staging images, the
regional destination, its promoted image, and the atomic writer's native
snapshot. Repeated-save tests prove the relevant I2/O2/P1 native controls stay
flat; they do not establish flat heap/RSS or the unverified 16 MiB / 512 KiB /
8 MiB / 64 operational target.
