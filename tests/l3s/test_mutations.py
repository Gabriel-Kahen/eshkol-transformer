"""Compile genuine L3S arithmetic mutants and execute independent native checks."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def replace_once(source, old, new):
    if source.count(old)!=1:
        raise AssertionError(f"mutation anchor must occur exactly once: {old!r}")
    return source.replace(old,new)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cc",required=True)
    parser.add_argument("--build-dir",type=Path,required=True)
    args=parser.parse_args()
    args.build_dir.mkdir(parents=True,exist_ok=True)
    artifacts=Path(os.environ.get("BUILD_DIR",str(ROOT/"build")))
    if not artifacts.is_absolute():artifacts=ROOT/artifacts
    source=(ROOT/"native/l3s_masked_objective_provider.c").read_text()
    seed="op == MEAN_BOOL || op == MEAN_F32 ? m[i] / w : m[i]"
    mutations={
        "pairing": [("const float p0 = m[0] * ce[0];","const float p0 = m[0] * ce[1];"),
                    ("const float p1 = m[1] * ce[1];","const float p1 = m[1] * ce[0];")],
        "seed-position": [("result[i] = positive_zero(seed);","result[1u-i] = positive_zero(seed);")],
        "mean-missing-normalization": [("const float mean = n / w;","const float mean = n;")],
        "mean-double-normalization": [("const float mean = n / w;","const float mean = (n / w) / w;")],
        "mean-count-normalization": [("const float mean = n / w;","const float mean = n / 2.0f;")],
        "seed-missing-normalization": [(seed,"op == MEAN_BOOL || op == MEAN_F32 ? m[i] : m[i]")],
        "seed-double-normalization": [(seed,"op == MEAN_BOOL || op == MEAN_F32 ? (m[i] / w) / w : m[i]")],
        "seed-count-normalization": [(seed,"op == MEAN_BOOL || op == MEAN_F32 ? m[i] / 2.0f : m[i]")],
        "numerator-mean-seed-swap": [(seed,"op == MEAN_BOOL || op == MEAN_F32 ? m[i] : m[i] / w")],
        "reciprocal-substitution": [(seed,"op == MEAN_BOOL || op == MEAN_F32 ? m[i] * (1.0f / w) : m[i]")],
        "contraction": [("n = n + p1;","n = fmaf(m[1], ce[1], n);")],
    }
    reversal=[("w = w + m[0];","w = w + m[1];"),
              ("w = w + m[1];","w = w + m[0];"),
              ("n = n + p0;","n = n + p1;"),
              ("n = n + p1;","n = n + p0;")]
    # Simultaneous placeholders avoid undoing the first half of each swap.
    reversed_source=source
    for i,(old,_new) in enumerate(reversal):
        reversed_source=replace_once(reversed_source,old,f"/* reversal-{i} */")
    for i,(_old,new) in enumerate(reversal):
        reversed_source=reversed_source.replace(f"/* reversal-{i} */",new)
    variants={"baseline":source,"addition-reversal-equivalent":reversed_source}
    for name,changes in mutations.items():
        changed=source
        for old,new in changes:changed=replace_once(changed,old,new)
        variants[name]=changed
    flags=["-std=c11","-Wall","-Wextra","-Werror","-Wpedantic","-O2",
           "-ffp-contract=off","-fexcess-precision=standard","-fno-fast-math","-frounding-math",
           "-I",str(ROOT/"include"),"-I",str(ROOT/"tests/l3s")]
    baseline=None
    adversarial_baseline=None
    for name,changed in variants.items():
        path=args.build_dir/(name+".c");binary=args.build_dir/name
        path.write_text(changed)
        command=[args.cc,*flags,str(path),str(ROOT/"tests/l3s/oracle_runner.c"),
                 str(artifacts/"l2/libeshkol_transformer_l2.a"),str(artifacts/"k1/libeshkol_transformer_k1.a"),
                 "-lm","-o",str(binary)]
        compiled=subprocess.run(command,capture_output=True,text=True,timeout=60)
        if compiled.returncode!=0 or compiled.stdout or compiled.stderr:
            raise AssertionError(f"{name} failed clean compilation: {compiled.stdout}{compiled.stderr}")
        result=subprocess.run([str(binary),"--emit"],capture_output=True,timeout=60)
        if name=="baseline":
            if result.returncode!=0 or result.stderr:raise AssertionError(result.stderr.decode())
            baseline=result.stdout
        elif name=="addition-reversal-equivalent":
            if result.returncode!=0 or result.stderr or result.stdout!=baseline:
                raise AssertionError("pure two-term addition reversal must be observationally equivalent")
        elif result.returncode!=1 or b"FAIL " not in result.stderr or b"dispatch " in result.stderr:
            raise AssertionError(f"{name}: expected independent numerical assertion, got {result.returncode}: {result.stderr!r}")
        if name in ("baseline","addition-reversal-equivalent"):
            # The admitted-domain control also executes real signed-zero,
            # subnormal/extreme, rejection, environment and atomicity cases.
            atomic_binary=args.build_dir/(name+"-atomicity")
            atomic_command=[args.cc,*flags,str(path),str(ROOT/"tests/l3s/test_l3s.c"),
                            str(artifacts/"k1/libeshkol_transformer_k1.a"),"-lm","-o",str(atomic_binary)]
            compiled=subprocess.run(atomic_command,capture_output=True,text=True,timeout=60)
            if compiled.returncode!=0 or compiled.stdout or compiled.stderr:
                raise AssertionError(f"{name} atomicity control failed clean compilation: {compiled.stdout}{compiled.stderr}")
            atomic=subprocess.run([str(atomic_binary)],capture_output=True,timeout=60)
            if atomic.returncode!=0 or atomic.stderr:
                raise AssertionError(f"{name} atomicity control failed: {atomic.stderr!r}")
            if name=="baseline":adversarial_baseline=atomic.stdout
            elif atomic.stdout!=adversarial_baseline:
                raise AssertionError("addition reversal changed admitted-domain atomicity output")
        print(f"L3S compiled mutation {name}: {'equivalent' if name.endswith('equivalent') else 'passes' if name=='baseline' else 'killed'}")
    print(f"L3S MUTATION PASS: {len(mutations)} compiled mutants killed; addition reversal equivalent in numeric and adversarial domains")


if __name__=="__main__":main()
