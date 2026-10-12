"""Independent bounded G3RCV1 pair fixture generator for the private test."""
from hashlib import sha256
from pathlib import Path
import struct
import sys


def record(digest: bytes, fingerprint_digit: bytes = b"0") -> bytes:
    data = bytearray(288)
    data[:6] = b"G3RCV1"
    struct.pack_into("<HHIIIIII", data, 8, 1, 0, 288, 1, 1, 2, 2, 0)
    struct.pack_into("<I", data, 36, 0)
    struct.pack_into("<6q", data, 40, 0, 1065353216, 256,
                     1065353216, 1, -1)
    struct.pack_into("<4q", data, 88, 1, 7, 0, 0)
    data[120:122] = bytes((7, 11))
    data[128:224] = (b"sha256:eshkol-byte-tokenizer-v1:"
                     + fingerprint_digit * 64)
    data[224:256] = digest
    data[256:288] = sha256(data[:256]).digest()
    return bytes(data)


def main() -> None:
    c2_path, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=False)
    good = record(sha256(c2_path.read_bytes()).digest())
    (output / "good.g3r").write_bytes(good)
    (output / "short.g3r").write_bytes(good[:-1])
    (output / "long.g3r").write_bytes(good + b"\0")
    torn = bytearray(good)
    torn[120] ^= 1
    (output / "torn.g3r").write_bytes(torn)
    (output / "mismatched.g3r").write_bytes(record(bytes(32)))
    (output / "wrong-tokenizer.g3r").write_bytes(
        record(sha256(c2_path.read_bytes()).digest(), b"a"))


if __name__ == "__main__":
    main()
