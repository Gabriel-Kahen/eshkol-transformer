# Private history/T1 one-root runner: source preflight HOLD

Status: **source-only HOLD**. `scripts/preflight-g3c4-public-t1-private-union.py`
checks the clean descendant of reviewed `4ab9b5c0`, byte-identical fixed
package inputs and repository source-closure members against that parent, the six ordered native
provider objects, one archive member, and exactly one package-root load in
each private fixture. It checks the T1 native companion's four compile gates
and absence of the two release-only test gates in the public package recipe.
It always exits 2. It cannot invoke a compiler, loader, Docker, or native
binary. The source negative tests mutate root count/identity, mixed package
use, closure entries, fixed manifest membership/order, and filesystem type.

The fixed installed tuple and ordered include roots are those of
`scripts/build-g3g-manual.sh` and `scripts/g3g-manual-package-policy.sh`.
The existing private strict-AOT command shape is in
`scripts/test-g3g-manual-private.sh`, but its native archive omits source
members required by the G3-G package and it targets `private_fault_runtime.esk`.
It cannot safely run the history or T1 fixture by changing only its caller
path. Each eventual private history/T1 binary needs its own one-root build,
fresh cache and one native closure; neither may link the installed archive.
The native companion requires `ET_T1_SEALED_RELEASE_TEST_ONLY`,
`ET_T1_I64_SHELL_TESTING`, `ET_G3C4_T1_I1_EXACT_PAIR_PRIVATE`, and
`ET_I64_TENSOR_STORAGE_QUERY_PRIVATE` in its private build only.

No full independently admitted external history/T1 seals, selected-image
loader/generated-header/runtime/provider pin inventory, or package-phase resource bounds
are bound to this source candidate. The historical F loader result and the
source-plan seal are not a current launch decision. An independent source/tests
review, immutable full pin decision, negative drift preflight before every
child, and measured resource bounds are required before implementing a launch
path. The combined compiled gate and current-main integration remain OPEN.

## Decision-gated AOT route candidate

`scripts/run-g3c4-public-t1-private-union.py` defines the exact private
history/T1 AOT argv shape from the existing private fault runner: strict types,
optimization level zero, no standard library, the four ordered trusted roots,
`lib` and `native`, and one private archive per fixture and mode. The archive
must be independently built and pinned; the installed package archive is
excluded. A canonical launch decision must bind the exact source commit/tree,
accepted image, root review, every selected file's path/bytes/mode/owner,
symbol and closure inventories, owned slot receipt, and nine positive resource
bounds. The gate rejects missing or duplicate rows, dirty source, root
duplication, and pin drift before constructing a command.

The embedded approved decision digest is intentionally empty. No decision
file or private archive is supplied here, and this revision has no child
launch path. A reviewed successor must bind the actual full decision, build
the separate native closures without duplicate registries, stage owned inputs
against source-to-use races, enforce limits, recheck immediately before every
child, and inspect emitted symbols and receipts. The AOT argv candidate alone
does not establish any compiled behavior.
