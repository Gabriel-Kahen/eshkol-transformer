# E3 §9 N4 signed-zero and fenv reachability

Read-only source and host audit of baseline `44c0418d2d2d8f3c1354c922d3eeee39b0d7e6ef`
(tree `70706c0a25cf2055815de363eec6539823a8fca8`). The N3K, L3S,
E3-METRICS and E3 traversal sources on this branch are byte-identical to that
baseline. The only `e3_frame.c` successor addition is the gated N4 K1 observer;
its role dispatch and metric call are unchanged. No compiled N4 case is admitted
by this audit.

## Signed zero

Authentic E3 role 20 is `n3k.linear.forward-no-bias` with shape
`[1,2,4,256]` and the tied embedding/head owner
(`src/eshkol_transformer/e3_frame.c:142,1225-1228`). Its output is the same
retained `forward[20]` view used as the K1 correct operand
(`e3_frame.c:1323-1327,1343-1352`). N3K initializes **each** logit sum to
`+0.0f`, multiplies four input/weight pairs, adds each binary32 product to
the sum, and has no bias in this role
(`native/n3k_primitives_provider.c:248-266`). N3K rejects nonfinite products
and partial sums there. Supported controls require round-to-nearest, masked
exceptions, and clear FTZ/DAZ (`n3k_primitives_provider.c:421-428,609-612`).

Under those controls, adding `-0` to the initial `+0` yields `+0`; exact
cancellation also yields `+0`. Negative underflow of a product to `-0` is
erased by that addition. Subsequent sums of binary32 values cannot round a
nonzero negative result to `-0`: binary32 operands lie on the integer lattice
of minimum-subnormal units, so a nonzero exact sum has magnitude at least one
such unit. This covers any finite C1-loaded parameter state and admitted D2
row that reaches the final role, not just the bounded host sample below.

The independent host check rounds after every product and addition. Its eight
input/weight pairs cover both signed zeros, both minimum-subnormal signs,
normal signs and cancellation. It enumerates `8^4 = 4096` four-term sequences;
656 yield `+0`, zero yield `-0`, and 3440 are nonzero. This is a check of the
source argument, not an M3 runtime observation:

```sh
python3 -B - <<'PY'
from itertools import product
from struct import pack, unpack
f32 = lambda value: unpack('<f', pack('<f', value))[0]
word = lambda value: unpack('<I', pack('<f', value))[0]
value = lambda bits: unpack('<f', pack('<I', bits))[0]
negzero, minsub = value(0x80000000), value(1)
pairs = [(negzero, 1.), (0., -1.), (minsub, -1.), (-minsub, 1.),
         (1., -1.), (-1., 1.), (1., 1.), (-1., -1.)]
counts = {0: 0, 0x80000000: 0, 'other': 0}
for terms in product(pairs, repeat=4):
    total = f32(0.)
    for x, weight in terms:
        total = f32(total + f32(x * weight))
    bits = word(total)
    counts[bits if bits in (0, 0x80000000) else 'other'] += 1
assert counts == {0: 656, 0x80000000: 0, 'other': 3440}, counts
print('cases=4096 positive_zero=656 negative_zero=0 nonzero=3440')
PY
```

L3S canonicalizes zero CE and reduction inputs/outputs with `positive_zero`
(`native/l3s_masked_objective_provider.c:120,129-169`). E3 correct computes
its hit count from `+0`; E3 accumulation canonicalizes every zero float input
and output, and the finalizer canonicalizes its three zero float inputs
(`native/e3_evaluation_metrics_provider.c:205-220,226-287`). Thus authentic
nonempty D2 traversal supplies no `-0` logit to K1 correct or `-0` operand to
the metric finalizer. Direct-provider signed-zero behavior remains a separate
provider case. The private signed-zero row remains **structural/provider-only,
not compiled-admitted**.

## External floating environment

The E3 traversal stages D2 batch admission at 6, all 21 M3 roles at 7, CE at
8, L3S reduction at 9, K1 correct at 10, accumulation at 11 and finalization
at 12 (`native/e3_private_extension.esk:335-366`). The first role is genuine
`n3k.embedding.forward` (`e3_frame.c:1158-1161`). N3K checks x87 exception
masks and rounding plus MXCSR masks, rounding, DAZ and FTZ **before floating
inputs** (`n3k_primitives_provider.c:421-428,609-612`). Therefore an
unsupported control set externally before traversal first rejects at stage 7,
role 0, before K1 correct or finalization. A C1 parameter or D2 row cannot
time a thread-local fenv mutation between roles; this source path has no
reviewed callback or preinvoke fenv setter. A timed test seam would require
separate ownership/restoration and first-error review.

Sticky exception flags are distinct: N3K, L3S and E3-METRICS test control
bits, not x87 status or MXCSR bits 0-5
(`n3k_primitives_provider.c:421-428`,
`native/l3s_masked_objective_provider.c:66-72`,
`native/e3_evaluation_metrics_provider.c:68-74`). Their validation saves and
restores `fenv_t` (`n3k_primitives_provider.c:660-668`,
`e3_evaluation_metrics_provider.c:378-389`). Source permits previously raised
sticky flags to reach K1; exact preservation across the compiled traversal is
**open**. Unsupported timed-fenv private reachability remains
**structural/provider-only**, and neither case closes its provider or private
compiled gate.
