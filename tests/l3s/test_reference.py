"""Frozen PyTorch, independent analytic and double finite-difference evidence."""
from __future__ import annotations
import hashlib
import math
import struct
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tests.q0.oracle_format import decode_tensor,load_fixture,tensor_by_name
from tests.q0.numerics import ComparisonPolicy,TensorMetadata,compare_values

RUNNER = Path(sys.argv.pop())


class Reference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=load_fixture(Path(__file__).with_name("masked_objective_v1.json"))
        first=subprocess.run([str(RUNNER),"--emit"],check=True,capture_output=True,timeout=60)
        second=subprocess.run([str(RUNNER),"--emit"],check=True,capture_output=True,timeout=60)
        if first.stdout != second.stdout or first.stderr or second.stderr:
            raise AssertionError("native output is not deterministic and silent")
        cls.native={}
        for line in first.stdout.decode().splitlines():
            name,*values=line.split()
            if name in cls.native: raise AssertionError("duplicate native field")
            cls.native[name]=tuple(struct.unpack(">f",bytes.fromhex(v))[0] for v in values)

    def test_pinned_identity_and_exact_schema(self):
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        g=self.payload["generator"]
        self.assertEqual(g["source_sha256"],digest(Path(__file__).with_name("generate_reference.py")))
        self.assertEqual(g["dependency_lock_sha256"],digest(ROOT/"tests/q0/requirements-oracle.lock"))
        self.assertEqual(g["framework"],dict(name="pytorch",version="2.13.0+cpu"))
        expected={t["name"] for t in self.payload["tensors"] if t["role"] != "input"}
        self.assertEqual(set(self.native),expected)
        self.assertEqual(tensor_by_name(self.payload,"input.logits")["shape"],[1,2,256])

    def test_frozen_pytorch_parity(self):
        for name,actual in self.native.items():
            t=tensor_by_name(self.payload,name)
            policy=ComparisonPolicy(3e-6,3e-5)
            self.assertTrue(compare_values(actual,decode_tensor(t),
                actual=TensorMetadata("native",tuple(t["shape"]),"float32","cpu"),
                expected=TensorMetadata("torch",tuple(t["shape"]),"float32","cpu"),policy=policy).passed,name)

    def test_independent_analytic_and_finite_difference(self):
        logits=list(decode_tensor(tensor_by_name(self.payload,"input.logits")))
        targets=(3,201)
        def ce(x,t):
            maximum=max(x)
            return math.log(math.fsum(math.exp(a-maximum) for a in x))+maximum-x[t]
        losses=[ce(logits[256*t:256*(t+1)],targets[t]) for t in range(2)]
        maximum_error=0.0
        for case in ("ones","weighted","zero","scaled"):
            mask=decode_tensor(tensor_by_name(self.payload,case+".mask"));weight=sum(mask)
            for mean in (False,True):
                suffix="mean_gradient" if mean else "numerator_gradient"
                actual=self.native[case+"."+suffix]
                def objective(x):
                    n=math.fsum(mask[t]*ce(x[256*t:256*(t+1)],targets[t]) for t in range(2))
                    return n/weight if mean else n
                for i in range(512):
                    t,j=divmod(i,256);row=logits[256*t:256*(t+1)]
                    m=max(row);prob=math.exp(row[j]-m)/math.fsum(math.exp(a-m) for a in row)
                    expected=mask[t]*(prob-(j==targets[t]))/(weight if mean else 1)
                    maximum_error=max(maximum_error,abs(actual[i]-expected))
                    self.assertLessEqual(abs(actual[i]-expected),3e-6+3e-5*abs(expected))
                for i in (0,3,127,255,256,457,400,511):
                    high,low=logits.copy(),logits.copy();high[i]+=1e-4;low[i]-=1e-4
                    fd=(objective(high)-objective(low))/2e-4
                    self.assertLessEqual(abs(actual[i]-fd),3e-6+3e-5*abs(fd))
            reduction=self.native[case+".reduction"]
            self.assertEqual(reduction[1],weight)
            expected=math.fsum(m*l for m,l in zip(mask,losses))/weight
            self.assertLessEqual(abs(reduction[2]-expected),3e-6+3e-5*abs(expected))
        print(f"L3S independent analytic maximum gradient absolute error: {maximum_error:.9g}")

    def test_normalization_and_scaling_distinctions(self):
        weighted=self.native["weighted.mean_gradient"]
        self.assertEqual(weighted,self.native["scaled.mean_gradient"])
        self.assertEqual(tuple(2*x for x in self.native["weighted.numerator_gradient"]),self.native["scaled.numerator_gradient"])
        self.assertNotEqual(weighted,self.native["weighted.numerator_gradient"])
        self.assertTrue(all(x==0 for x in self.native["zero.mean_gradient"][:256]))


if __name__ == "__main__": unittest.main()
