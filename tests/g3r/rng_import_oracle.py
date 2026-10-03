"""Independent development-only G3-S oracle for actual imported-owner runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.g3s.reference import MASK64, bits, f32, literal, philox, word


def successor(low: int, high: int, mode: str) -> tuple[int, int]:
    if mode == "greedy":
        return low, high
    counter = ((high & MASK64) << 64) | (low & MASK64)
    assert counter < (1 << 128) - 1
    counter += 1
    def signed(value: int) -> int:
        return value if value < (1 << 63) else value - (1 << 64)
    return signed(counter & MASK64), signed(counter >> 64)


def check(path: Path) -> None:
    lines = [line for line in path.read_text().splitlines()
             if line.startswith("G3R_ORACLE ")]
    assert len(lines) == 3, len(lines)
    assert [(line.split()[1], int(line.split()[3]), int(line.split()[4]))
            for line in lines] == [
                ("categorical", -1, 0),
                ("categorical", -2, -1),
                ("greedy", -1, -1),
            ]
    for line in lines:
        fields = line.split()
        assert len(fields) == 14 + 256, len(fields)
        _, mode, seed, low, high, actual_id, *tail = fields
        seed, low, high = int(seed), int(low), int(high)
        native_state = tuple(map(int, tail[:4]))
        clone_state = tuple(map(int, tail[4:8]))
        raw = tail[8:]
        assert seed == 1729
        expected_state = (1, seed, *successor(low, high, mode))
        assert native_state == expected_state, (mode, native_state, expected_state)
        assert clone_state == expected_state, (mode, clone_state, expected_state)
        logit_bits = [int(value) for value in raw]
        assert all(0 <= value <= 0xFFFFFFFF for value in logit_bits)
        logits = [word(value) for value in logit_bits]
        assert all(bits(value) == encoded for value, encoded in zip(logits, logit_bits))
        if mode == "categorical":
            lanes = philox(seed, low, high)
            uniform = f32((lanes[0] >> 8) * 2**-24)
            expected_id = literal(logits, 1.0, 256, 1.0, uniform)["token"]
        else:
            expected_id = min(range(256), key=lambda index: (-logits[index], index))
        assert int(actual_id) == expected_id, (mode, actual_id, expected_id)
    print("G3-R imported RNG G3-S/Philox oracle: PASS (3 actual runs)")


if __name__ == "__main__":
    check(Path(sys.argv[1]))
