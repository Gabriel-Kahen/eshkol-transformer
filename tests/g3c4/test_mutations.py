#!/usr/bin/env python3
"""Compile semantic provider mutants; every mutant must trip an independent oracle."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f"mutation anchor count {text.count(old)}: {old!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--k1", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / "native/g3c4_primitives_provider.c").read_text()
    mutations = {
        "attention-row-order": (
            "{6,dims_0_0},{6,dims_0_1},{6,dims_0_2},{6,dims_0_3},{6,dims_0_4}",
            "{6,dims_0_1},{6,dims_0_0},{6,dims_0_2},{6,dims_0_3},{6,dims_0_4}"),
        "dimension-reserved": ("dims_0_0[] = {D(1),D(2)", "dims_0_0[] = {{1,1,0,{1}},D(2)"),
        "provider-name": ("\"g3c4.cpu-f32.serial\",\"1.0\",\"G3-C4:cpu-f32-c4-forward-v1\",COUNT(capabilities)", "\"g3c4.wrong\",\"1.0\",\"G3-C4:cpu-f32-c4-forward-v1\",COUNT(capabilities)"),
        "transposed-weight": ("weight[o * s->a + i]", "weight[i * s->b + o]"),
        "reverse-linear-order": ("for (size_t i = 0; i < s->a; i++) {", "for (size_t rev=s->a;rev>0u;--rev) { size_t i=rev-1u;"),
        "linear-fma": ("sum = sum + product;\n          if (!isfinite(sum))", "sum = fmaf(x[index_linear_x(s,n,t,i)],weight[o*s->a+i],sum);\n          if (!isfinite(sum))"),
        "unbiased-layernorm": ("*rstd = variance_sum / dimension;", "*rstd = variance_sum / (dimension - 1.0f);"),
        "reverse-layernorm-sum": ("sum = sum + x[index3(s, n, t, d)];", "sum = sum + x[index3(s,n,t,s->d-1u-d)];"),
        "layernorm-fma": ("const float value = scaled + beta[d];", "const float value = fmaf(normalized,gamma[d],beta[d]);"),
        "fixed-epsilon": ("*(const float *)in[3],out)", "1e-5f,out)"),
        "gelu-scale": ("const float inv_sqrt_2 = 0x1.6a09e6p-1f;", "const float inv_sqrt_2 = 1.0f;"),
        "layout-identity": ("out+(k==SPLIT_F ? head:token),\n          (const float *)in[0]+(k==SPLIT_F ? token:head)", "out+token+(head-head),\n          (const float *)in[0]+token"),
        "attention-ignore-mask": ("return mask[mask_index(s, n, tq, tk)] != 0u &&", "return (mask[mask_index(s,n,tq,tk)] != 0u || 1) &&"),
        "attention-reverse-causal": ("key_positions[n * s->tk + tk] <=", "key_positions[n * s->tk + tk] >="),
        "attention-collapse-kv-head": ("const size_t hkv = hq / group;", "const size_t hkv = 0u; (void)group;"),
        "both-shift-checks-removed": ("BOTH_SHIFT_CHECKS", ""),
        "denominator-shift-check": ("if (!isfinite(shifted)) return -1;", "if (isfinite(shifted)) return -1;"),
        # On stable admitted inputs recomputation yields the same finite score.
        # Inverting the second check proves the probability pass executes it;
        # source/provenance checks separately require the literal rejection.
        "probability-shift-check": ("if (!isfinite(shifted)) return 0;", "if (isfinite(shifted)) return 0;"),
        "attention-reduction-check": ("if (!isfinite(output)) {\n              return 0;\n            }", "if (0) {\n              return 0;\n            }"),
        "residual-negative-zero": ("if (write) out[i]=value;", "if (write) out[i]=value==0.0f ? 0.0f:value;"),
    }
    with tempfile.TemporaryDirectory(prefix="g3c4-mutations-") as temporary:
        directory = Path(temporary)
        for name, (old, new) in mutations.items():
            mutant = directory / f"{name}.c"
            if old == "BOTH_SHIFT_CHECKS":
                text = replace_once(source, "if (!isfinite(shifted)) return -1;", "if (0) return -1;")
                text = replace_once(text, "if (!isfinite(shifted)) return 0;", "if (0) return 0;")
            else:
                text = replace_once(source, old, new)
            mutant.write_text(text)
            binary = directory / name
            command = [args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                       "-ffp-contract=off", "-fexcess-precision=standard", "-fno-fast-math",
                       "-I" + str(root / "include"), str(root / "tests/g3c4/test_numerical.c"),
                       str(mutant), str(Path(args.k1).resolve()), "-lm", "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=os.environ.copy())
            if result.returncode == 0 or "FAIL " not in result.stderr:
                raise AssertionError(f"mutant {name} not cleanly killed: rc={result.returncode}\n{result.stderr}")
            print(f"G3-C4-N mutation {name}: detected")
    print(f"G3-C4-N mutations: {len(mutations)} compiled semantic mutants detected PASS")


if __name__ == "__main__":
    main()
