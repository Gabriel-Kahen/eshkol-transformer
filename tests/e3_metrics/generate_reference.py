"""Pinned development-only BOOL multi-batch PyTorch reference (Q0 format)."""
from __future__ import annotations
import hashlib
import json
import os
import platform
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.q0.oracle_format import encode_fixture

PIN = "90cbd7130f47b8184bcc77b8d5c1b0026da980de"
if os.environ.get("ATEN_CPU_CAPABILITY") != "default" or os.environ.get("MKL_CBWR") != "COMPATIBLE":
    raise RuntimeError("set ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE before generation")
import torch
import torch.nn.functional as functional


def tensor(name, role, shape, values, dtype="float32"):
    encoding = {"float32": "ieee754-hex-be", "float64": "ieee754-hex-be",
                "int64": "twos-complement-hex-be", "bool": "bool01"}[dtype]
    def word(x):
        if dtype == "float32": return struct.pack(">f", x).hex()
        if dtype == "float64": return struct.pack(">d", x).hex()
        if dtype == "int64": return int(x).to_bytes(8, "big", signed=True).hex()
        return str(int(x))
    return dict(name=name, role=role, shape=list(shape), data=[word(x) for x in values],
                dtype=dtype, device="cpu", layout="row_major", encoding=encoding)


def batch(index):
    values = [((i % 256) % 17 - 8) * 0.125 + index * 0.0625 + (0.03125 if i >= 256 else 0)
              for i in range(512)]
    if index == 0:
        values[3] = values[7] = values[256] = values[511] = 4
        return values, [3, 254], [1, 1]
    if index == 1:
        values[:256] = [0.0 if i % 2 else -0.0 for i in range(256)]
        values[511] = 4
        return values, [0, 255], [0, 1]
    values[0] = values[258] = 4
    return values, [0, 2], [1, 0]  # second position is padded, yet finite/valid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload():
    if platform.python_version() != "3.14.6" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("requires pinned Python 3.14.6 and torch 2.13.0+cpu")
    if f"eshkol_commit\t{PIN}\n" not in (ROOT / "toolchain/eshkol.lock").read_text():
        raise RuntimeError("canonical Eshkol source pin changed")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(107)
    tensors, cases, ns, ws, cs = [], [], [], [], []
    mathematical_numerator = torch.tensor(0, dtype=torch.float64)
    for index in range(3):
        name = f"batch{index}"
        values, ids, masks = batch(index)
        logits = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 256)
        targets = torch.tensor(ids, dtype=torch.int64)
        mask = torch.tensor(masks, dtype=torch.float32)
        ce = functional.cross_entropy(logits.reshape(2, 256), targets, reduction="none")
        correct = (logits.reshape(2, 256).argmax(-1) == targets).to(torch.float32)
        n, w, c = (ce * mask).sum(), mask.sum(), (correct * mask).sum()
        ns.append(n); ws.append(w); cs.append(c)
        mathematical_ce = functional.cross_entropy(logits.double().reshape(2, 256), targets, reduction="none")
        mathematical_numerator += (mathematical_ce * mask.double()).sum()
        tensors.extend((tensor(name + ".logits", "input", (1, 2, 256), values),
                        tensor(name + ".targets", "input", (1, 2), ids, "int64"),
                        tensor(name + ".mask", "input", (1, 2), masks, "bool"),
                        tensor(name + ".ce", "expected", (1, 2), ce.tolist()),
                        tensor(name + ".reduction", "expected", (3,), [n.item(), w.item(), (n / w).item()]),
                        tensor(name + ".correct", "expected", (), [c.item()]),
                        tensor(name + ".mathematical_ce", "expected", (1, 2), mathematical_ce.tolist(), "float64")))
        cases.append(dict(name=name, kind="parity", operation="e3.evaluation-metrics.composition",
                          inputs=[name + suffix for suffix in (".logits", ".targets", ".mask")],
                          expectation=dict(error=None, outputs=sorted(name + suffix for suffix in (".ce", ".reduction", ".correct", ".mathematical_ce"))),
                          tolerance=dict(absolute=struct.pack(">d", 2e-6).hex(), relative=struct.pack(">d", 2e-5).hex(), equal_nan=False)))
    n = torch.tensor(0, dtype=torch.float32)
    for item in ns: n = n + item
    w = sum(ws); c = sum(cs); loss = n / w
    tensors.append(tensor("global.metrics", "expected", (4,), [loss.item(), w.item(), loss.exp().item(), (c / w).item()]))
    tensors.append(tensor("global.mathematical_loss", "expected", (), [(mathematical_numerator / w.double()).item()], "float64"))
    cases.append(dict(name="global", kind="parity", operation="e3.evaluation-metrics.finalize",
                      inputs=sorted(f"batch{i}.{suffix}" for i in range(3) for suffix in ("logits", "targets", "mask")),
                      expectation=dict(error=None, outputs=["global.mathematical_loss", "global.metrics"]),
                      tolerance=dict(absolute=struct.pack(">d", 2e-6).hex(), relative=struct.pack(">d", 2e-5).hex(), equal_nan=False)))
    return dict(tensors=sorted(tensors, key=lambda t:t["name"]), cases=sorted(cases, key=lambda c:c["name"]),
                generator=dict(name="e3.pytorch.bool_metrics", version=1, seed=107,
                               framework=dict(name="pytorch", version="2.13.0+cpu"),
                               source_sha256=digest(Path(__file__)),
                               dependency_lock_sha256=digest(ROOT / "tests/q0/requirements-oracle.lock")))


if __name__ == "__main__":
    output = Path(sys.argv[1])
    output.write_bytes(encode_fixture(build_payload()))
    provenance = dict(schema="e3-metrics-oracle-provenance-v1", python=platform.python_version(),
                      torch=torch.__version__, eshkol_source_commit=PIN,
                      toolchain_lock_sha256=digest(ROOT / "toolchain/eshkol.lock"),
                      generator_sha256=digest(Path(__file__)), fixture_sha256=digest(output))
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, sort_keys=True, separators=(",", ":")) + "\n")
