"""Pinned development-only PyTorch masked objective reference; no native import."""
from __future__ import annotations
import hashlib
import os
import platform
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.q0.oracle_format import encode_fixture
if os.environ.get("ATEN_CPU_CAPABILITY") != "default" or os.environ.get("MKL_CBWR") != "COMPATIBLE":
    raise RuntimeError("set ATEN_CPU_CAPABILITY=default MKL_CBWR=COMPATIBLE before generation")
import torch
import torch.nn.functional as functional


def tensor(name, role, shape, values, dtype="float32"):
    encoding = "ieee754-hex-be" if dtype == "float32" else "twos-complement-hex-be"
    data = [struct.pack(">f", v).hex() if dtype == "float32" else int(v).to_bytes(8, "big", signed=True).hex() for v in values]
    return dict(name=name, role=role, shape=list(shape), data=data, dtype=dtype,
                device="cpu", layout="row_major", encoding=encoding)


def build_payload():
    if platform.python_version() != "3.14.6" or torch.__version__ != "2.13.0+cpu":
        raise RuntimeError("requires pinned Python 3.14.6 and torch 2.13.0+cpu")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(8701)
    values = [((i % 256) % 17 - 8) * 0.125 + (0.03125 if i >= 256 else 0) for i in range(512)]
    values[3], values[457] = 1.75, -1.5
    logits = torch.tensor(values, dtype=torch.float32, requires_grad=True).reshape(1, 2, 256)
    targets = torch.tensor([[3, 201]], dtype=torch.int64)
    losses = functional.cross_entropy(logits.reshape(2,256), targets.reshape(2), reduction="none").reshape(1,2)
    tensors = [tensor("input.logits", "input", (1,2,256), values),
               tensor("input.targets", "input", (1,2), [3,201], "int64")]
    cases = []
    for name, weights in (("ones",[1,1]),("weighted",[0.5,2]),("zero",[0,1]),("scaled",[1,4])):
        mask = torch.tensor([weights],dtype=torch.float32)
        weight = mask.sum()
        numerator = (mask * losses).sum()
        mean = numerator / weight
        ng, = torch.autograd.grad(numerator, logits, retain_graph=True)
        mg, = torch.autograd.grad(mean, logits, retain_graph=True)
        tensors.append(tensor(name+".mask", "input", (1,2), weights))
        results = {"loss": ((1,2), losses), "reduction": ((3,),torch.stack((numerator,weight,mean))),
                   "numerator_seed": ((1,2),mask), "mean_seed": ((1,2),mask/weight),
                   "numerator_gradient": ((1,2,256),ng), "mean_gradient": ((1,2,256),mg)}
        outputs = []
        for field,(shape,result) in results.items():
            output = name+"."+field
            outputs.append(output)
            tensors.append(tensor(output,"analytic_gradient" if "gradient" in field else "expected",
                                  shape,result.detach().reshape(-1).tolist()))
        for kind in ("parity","gradient"):
            selected=[o for o in outputs if ("gradient" in o)==(kind=="gradient")]
            cases.append(dict(name=name+"."+kind,kind=kind,operation="l3s.masked-objective.composition",
                              inputs=["input.logits","input.targets",name+".mask"],
                              expectation=dict(error=None,outputs=sorted(selected)),
                              tolerance=dict(absolute=struct.pack(">d",3e-6).hex(),relative=struct.pack(">d",3e-5).hex(),equal_nan=False)))
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    return dict(tensors=sorted(tensors,key=lambda t:t["name"]),cases=sorted(cases,key=lambda c:c["name"]),
                generator=dict(name="l3s.pytorch.masked_objective",version=1,seed=8701,
                               framework=dict(name="pytorch",version="2.13.0+cpu"),
                               source_sha256=digest(Path(__file__)),
                               dependency_lock_sha256=digest(ROOT/"tests/q0/requirements-oracle.lock")))


if __name__ == "__main__":
    Path(sys.argv[1]).write_bytes(encode_fixture(build_payload()))
