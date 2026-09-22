#!/usr/bin/env python3
"""Development-only mutation gate: each compiled numerical mutant must fail tests."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f"mutation anchor is not unique: {old!r}")
    return text.replace(old, new, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", required=True)
    parser.add_argument("--k1", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = (root / "native/g3n_primitives_provider.c").read_text()
    mutations = {
        "raw-embedding-row-order": ("rows_1[] = {{4,dims_1_0},{4,dims_1_1}};", "rows_1[] = {{4,dims_1_1},{4,dims_1_0}};"),
        "raw-dimension-reserved": ("dims_0_0[] = {{1,1,0,{0}},", "dims_0_0[] = {{1,1,0,{1}},"),
        "raw-provider-name": ("sizeof(et_kernel_provider_v1),ET_KERNEL_ABI_MAJOR,ET_KERNEL_ABI_MINOR,0,\n  \"g3n.cpu-f32.serial\"", "sizeof(et_kernel_provider_v1),ET_KERNEL_ABI_MAJOR,ET_KERNEL_ABI_MINOR,0,\n  \"g3n.wrong\""),
        "transposed-weight": ("weight[o * s->a + i]", "weight[i * s->b + o]"),
        "reverse-linear-order": ("for (size_t i = 0; i < s->a; i++) {", "for (size_t rev = s->a; rev > 0u; --rev) { size_t i = rev - 1u;"),
        "contract-linear-fma": ("sum = sum + product;\n          if (!isfinite(sum))", "sum = fmaf(x[index_linear_x(s, n, t, i)], weight[o * s->a + i], sum);\n          if (!isfinite(sum))"),
        "contract-layernorm-fma": ("const float value = scaled + beta[d];", "const float value = fmaf(normalized, gamma[d], beta[d]); (void)scaled;"),
        "reverse-layernorm-sum": ("sum = sum + x[index3(s, n, t, d)];", "sum = sum + x[index3(s, n, t, s->d - 1u - d)];"),
        "unbiased-layernorm": ("*rstd = variance_sum / dimension;", "*rstd = variance_sum / (dimension - 1.0f);"),
        "fixed-epsilon": ("*(const float *)in[3],out)", "1e-5f,out)"),
        "skip-probability-one-score": ("const size_t group = s->hq / s->hkv;", "*result = 0.0f; return 1;\n  const size_t group = s->hq / s->hkv;"),
        "ignore-keep": ("return mask[mask_index(s, n, tq, tk)] != 0u &&", "return (mask[mask_index(s, n, tq, tk)] != 0u || 1) &&"),
        "reverse-causal-position": ("key_positions[n * s->tk + tk] <=", "key_positions[n * s->tk + tk] >="),
        "ignore-unselected-nonfinite": ("if (exact_text(s.in[i].dtype,\"f32\") &&", "if (k!=EMB_F && exact_text(s.in[i].dtype,\"f32\") &&"),
        "erase-residual-negative-zero": ("if (write) out[i]=value;", "if (write) out[i]=value == 0.0f ? 0.0f : value;"),
    }
    with tempfile.TemporaryDirectory(prefix="g3n-mutations-") as temporary:
        directory = Path(temporary)
        for name, (old, new) in mutations.items():
            path = directory / f"{name}.c"
            path.write_text(replace_once(source, old, new))
            binary = directory / name
            command = [args.cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
                       "-ffp-contract=off", "-fexcess-precision=standard", "-fno-fast-math",
                       "-I" + str(root / "include"), str(root / "tests/g3n/test_numerical.c"),
                       str(path), str(Path(args.k1).resolve()), "-lm", "-o", str(binary)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True, env=os.environ.copy())
            if result.returncode == 0 or "FAIL " not in result.stderr:
                raise AssertionError(f"mutant {name} not cleanly killed: rc={result.returncode}\n{result.stderr}")
            print(f"G3N mutation {name}: detected")
    print(f"G3N mutations: {len(mutations)} compiled semantic mutants detected PASS")


if __name__ == "__main__":
    main()
