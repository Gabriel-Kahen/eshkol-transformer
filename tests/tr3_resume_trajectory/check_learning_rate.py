"""Compare O2's next-update effective f32 learning rate across fresh processes."""

from pathlib import Path
import re
import struct
import sys


evidence = Path(sys.argv[1])
pattern = re.compile(r"^TR3-C NEXT LR (uninterrupted|producer|receiver) ([123]) ([1-4]) ([0-9]+)$")
base_bits = 0x3DCCCCCD  # The single optimizer group in runtime_smoke.esk.
expected_factors = (0x3F000000, 0x3F800000, 0x3F466666, 0x3F0CCCCD)


def f32(bits: int) -> float:
    return struct.unpack(">f", bits.to_bytes(4, "big"))[0]


def bits(value: float) -> int:
    return int.from_bytes(struct.pack(">f", value), "big")


def trace(mode: str, accumulation: int) -> dict[int, int]:
    path = evidence / f"{mode}-{accumulation}.stdout"
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith("TR3-C NEXT LR "):
            continue
        match = pattern.fullmatch(line)
        assert match is not None, (path, line)
        found_mode, found_accumulation, update, factor = match.groups()
        assert (found_mode, int(found_accumulation)) == (mode, accumulation)
        update = int(update)
        assert update not in result, (path, update)
        result[update] = int(factor)
    return result


rows = ["accumulation\tupdate\tuninterrupted_factor\tcontinued_factor\teffective_lr_bits"]
for accumulation in (1, 2, 3):
    uninterrupted = trace("uninterrupted", accumulation)
    producer = trace("producer", accumulation)
    receiver = trace("receiver", accumulation)
    assert set(uninterrupted) == {1, 2, 3, 4}
    assert set(producer) == {1}
    assert set(receiver) == {2, 3, 4}
    for update, expected in enumerate(expected_factors, 1):
        continuation = producer if update == 1 else receiver
        source_factor = uninterrupted[update]
        continued_factor = continuation[update]
        assert source_factor == continued_factor == expected, (
            accumulation, update, source_factor, continued_factor, expected
        )
        # O2 uses binary32 multiplication of this fixed group LR and its native
        # schedule factor. A product of two f32 values is exact in binary64,
        # so packing once to f32 reproduces the product's single rounding.
        source_lr = bits(f32(base_bits) * f32(source_factor))
        continued_lr = bits(f32(base_bits) * f32(continued_factor))
        assert source_lr == continued_lr
        rows.append(
            f"{accumulation}\t{update}\t{source_factor:08x}\t"
            f"{continued_factor:08x}\t{source_lr:08x}"
        )

(evidence / "effective-learning-rate.tsv").write_text("\n".join(rows) + "\n")
print("TR3-C fresh-process effective LR PASS: 12 exact binary32 next-update comparisons")
