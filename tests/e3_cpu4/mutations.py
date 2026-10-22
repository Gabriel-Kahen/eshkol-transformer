"""Compile provider-only T4 arithmetic/admission mutants after a passing baseline."""

import argparse
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def variants(source):
    first="""    for (size_t i = 0; i < 4; ++i) {
      const float product = weights[i] * losses[i];"""
    edits={
        "reordered-four-term-sum":(first,"""    for (size_t j = 4; j > 0; --j) {
      const size_t i = j-1;
      const float product = weights[i] * losses[i];"""),
        "wider-four-term-sum":("float numerator = 0.0f;","double numerator = 0.0;"),
        "physical-capacity-denominator":("const float average = numerator / weight;",
                                           "const float average = numerator / 4.0f;"),
        "truncated-four-word-seed":("memcpy(output(c,0)->data, result, 4u*sizeof(float));",
                                      "memcpy(output(c,0)->data, result, 2u*sizeof(float));"),
        "masked-last-invalid-bypass":("""      losses[i] = positive_zero(value);""",
                                       """      losses[i] = positive_zero(value);"""),
    }
    for name,(old,new) in edits.items():
        if name=="masked-last-invalid-bypass":
            old="""      if (!nonnegative(value)) return 0;
      losses[i] = positive_zero(value);"""
            new="""      if (weights[i] != 0.0f && !nonnegative(value)) return 0;
      losses[i] = weights[i] == 0.0f ? 0.0f : positive_zero(value);"""
        if source.count(old)!=1:raise AssertionError(f"mutation anchor differs: {name}")
        yield name,source.replace(old,new,1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--objects",type=Path,required=True)
    parser.add_argument("--reference-dir",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir()
    flags=["-std=c11","-Wall","-Wextra","-Werror","-Wpedantic","-O2",
           "-ffp-contract=off","-fexcess-precision=standard","-fno-fast-math","-frounding-math",
           "-fstack-protector-all","-I",str(ROOT/"include"),"-I",str(args.reference_dir)]
    source=(ROOT/"native/l3s_masked_objective_provider.c").read_text()
    objects=[args.objects/name for name in ("kernel_abi.o","indexed_cross_entropy.o","e3_evaluation_metrics_provider.o")]
    for name,changed in variants(source):
        path=args.output/(name+".c");binary=args.output/name;path.write_text(changed)
        command=[args.cc,*flags,str(path),str(ROOT/"tests/e3_cpu4/native.c"),*map(str,objects),"-lm","-o",str(binary)]
        compiled=subprocess.run(command,capture_output=True,timeout=60)
        (args.output/(name+".compile.stdout")).write_bytes(compiled.stdout)
        (args.output/(name+".compile.stderr")).write_bytes(compiled.stderr)
        if compiled.returncode or compiled.stdout or compiled.stderr:
            raise AssertionError(f"mutant did not compile cleanly: {name}")
        result=subprocess.run([str(binary)],capture_output=True,timeout=60)
        (args.output/(name+".stdout")).write_bytes(result.stdout)
        (args.output/(name+".stderr")).write_bytes(result.stderr)
        (args.output/(name+".exit")).write_text(f"{result.returncode}\n")
        if result.returncode!=1 or not result.stderr.startswith(b"FAIL ") or b"unexpected dispatch" in result.stderr:
            raise AssertionError(f"mutant failed outside its value/admission assertion: {name}")
        print(f"CPU4 provider mutant {name}: compiled assertion kill")


if __name__=="__main__":main()
