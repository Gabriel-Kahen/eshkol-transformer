"""Independent binary inspector for the GGUF v3 IO5 test fixtures."""

import struct


def unpack_quant_gguf(raw: bytes):
    header = struct.Struct("<4sIQQ")
    magic, version, tensor_count, meta_count = header.unpack_from(raw)
    assert magic == b"GGUF" and version == 3
    at = header.size

    def number(fmt):
        nonlocal at
        value, = struct.unpack_from("<" + fmt, raw, at)
        at += struct.calcsize("<" + fmt)
        return value

    def text():
        nonlocal at
        length = number("Q")
        start = at
        at += length
        assert at <= len(raw)
        return raw[start:at].decode("utf-8")

    metadata = {}
    for _ in range(meta_count):
        key = text()
        kind = number("I")
        assert key and key not in metadata and kind in (4, 8)
        metadata[key] = (kind, text() if kind == 8 else number("I"))
    infos = []
    for _ in range(tensor_count):
        name = text()
        rank = number("I")
        dims = tuple(number("Q") for _ in range(rank))
        ggml_type = number("I")
        offset = number("Q")
        infos.append((name, dims, ggml_type, offset))
    data_at = at + (-at % 32)
    rows = []
    for name, dims, ggml_type, offset in infos:
        count = 1
        for dim in dims:
            count *= dim
        if ggml_type == 0:
            size = count * 4
        elif ggml_type == 1:
            size = count * 2
        elif ggml_type == 8:
            size = (count // 32) * 34
        else:
            raise AssertionError(f"unexpected type {ggml_type}")
        start = data_at + offset
        rows.append((name, dims, ggml_type, offset, raw[start:start + size]))
    return metadata, rows, at, data_at
