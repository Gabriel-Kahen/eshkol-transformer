"""Compiled provider and orchestration mutations, with explicit equivalent controls.

Every kill executes a successfully compiled binary. Numeric mutations must differ
in a value assertion/transcript, never merely crash or lose capability dispatch.
Admission/atomicity mutations are reported separately against the native guard gate.
"""
from __future__ import annotations
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tests.q0.oracle_format import load_fixture
from test_reference import check_transcript


def replace(source, old, new):
    if source.count(old) != 1:
        raise AssertionError(f"mutation anchor count {source.count(old)}: {old!r}")
    return source.replace(old, new)


def changed(source, edits):
    for old, new in edits: source = replace(source, old, new)
    return source


def command(argv):
    result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=60)
    return result


class Gate:
    def __init__(self, cc, directory, runner):
        self.cc, self.directory, self.runner = cc, Path(directory), Path(runner)
        self.directory.mkdir(parents=True, exist_ok=True)
        artifacts = Path(os.environ.get("BUILD_DIR", ROOT / "build"))
        if not artifacts.is_absolute(): artifacts = ROOT / artifacts
        self.k1 = artifacts / "k1/libeshkol_transformer_k1.a"
        self.libraries = [artifacts / p for p in ("l3s/libeshkol_transformer_l3s.a", "l2/libeshkol_transformer_l2.a",
                          "i2/libeshkol_transformer_f32.a", "i1/libeshkol_transformer_i64.a")] + [self.k1]
        self.flags = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-O2", "-ffp-contract=off",
                      "-fexcess-precision=standard", "-fno-fast-math", "-frounding-math", "-I", ROOT / "include"]
        self.probes = [("--emit",), ("--witness",), ("--mean-witness",),
                       ("--finalize", "42b17217", "3f800000", "00000000"),
                       ("--finalize", "42b17218", "3f800000", "00000000")]
        self.baseline = []
        for args in self.probes:
            result = command([runner, *args])
            if result.returncode or result.stderr: raise AssertionError("baseline numerical runner failed: " + result.stderr)
            self.baseline.append(result.stdout)
        check_transcript(load_fixture(TESTS / "bool_metrics_v1.json"), runner)

    def compile(self, name, provider, driver, native=False):
        source = self.directory / (name + ".c")
        source.write_text(provider)
        binary = self.directory / name
        if native:
            files = [source, TESTS / "native_test.c", self.k1]
            defines = []
        else:
            composed = self.directory / (name + "-driver.c")
            composed.write_text(driver)
            files = [source, composed, *self.libraries]
            defines = ["-DET_E3_COMPOSITION_STANDALONE"]
        result = command([self.cc, *self.flags, *defines, *files, "-lm", "-o", binary])
        if result.returncode or result.stdout or result.stderr:
            raise AssertionError(f"{name} failed clean compilation: {result.stdout}{result.stderr}")
        return binary

    def numeric(self, name, provider, driver, equivalent=False):
        binary = self.compile(name, provider, driver)
        differences = []
        for args, expected in zip(self.probes, self.baseline):
            result = command([binary, *args])
            if result.returncode:
                if result.returncode != 1 or "NUMERIC ASSERTION " not in result.stderr or "UNEXPECTED DISPATCH " in result.stderr:
                    raise AssertionError(f"{name}: nonnumeric failure {result.returncode}: {result.stderr}")
                differences.append(args[0] + ": numerical assertion")
            elif result.stderr:
                raise AssertionError(f"{name}: unexpected diagnostics: {result.stderr}")
            elif result.stdout != expected:
                differences.append(args[0] + ": exact numeric transcript")
            if differences and not equivalent:
                break  # A concrete value assertion is already a meaningful kill.
        if equivalent == bool(differences):
            raise AssertionError(f"{name}: expected {'equivalence' if equivalent else 'numeric kill'}, got {differences}")
        return differences

    def native(self, name, provider, driver, equivalent=False):
        binary = self.compile(name + "-native", provider, driver, native=True)
        result = command([binary])
        if equivalent:
            if result.returncode or result.stderr: raise AssertionError(name + ": equivalent native control failed: " + result.stderr)
        elif result.returncode != 1 or "native_test.c:" not in result.stderr:
            raise AssertionError(f"{name}: expected admission/atomicity assertion, got {result.returncode}: {result.stderr}")


def main():
    cc, directory, runner = sys.argv[1:4]
    gate = Gate(cc, directory, runner)
    provider = (ROOT / "native/e3_evaluation_metrics_provider.c").read_text()
    driver = (TESTS / "composition.c").read_text()
    gate.numeric("baseline", provider, driver, equivalent=True)
    gate.native("baseline", provider, driver, equivalent=True)
    numeric = {
        "highest-index-tie": [("if (row[j] > maximum)", "if (row[j] >= maximum)")],
        "ignore-mask": [("const float weight0 = mask[0] == 0u ? 0.0f : 1.0f;", "const float weight0 = 1.0f;"),
                        ("const float weight1 = mask[1] == 0u ? 0.0f : 1.0f;", "const float weight1 = 1.0f;")],
        "flip-mask": [("const float weight0 = mask[0] == 0u ? 0.0f : 1.0f;", "const float weight0 = mask[0] == 0u ? 1.0f : 0.0f;"),
                      ("const float weight1 = mask[1] == 0u ? 0.0f : 1.0f;", "const float weight1 = mask[1] == 0u ? 1.0f : 0.0f;")],
        "target-offset": [("winner0 == targets[0]", "winner0 == (targets[0] + 1) % 256")],
        "wrong-axis": [("winner1 = argmax(logits + 256)", "winner1 = argmax(logits)")],
        "padding-count-two": [("result->correct.active = (int64_t)mask[0] + (int64_t)mask[1];", "result->correct.active = 2;")],
        "omit-batch-numerator": [("const float next_n = n + nb;", "const float next_n = n + 0.0f * nb;")],
        "double-batch-numerator": [("const float next_n = n + nb;", "const float next_n = n + 2.0f * nb;")],
        "wrong-next-counter": [("const int64_t next_tokens = tokens + active;", "const int64_t next_tokens = tokens + active + 1;")],
        "extra-denominator": [("const float loss = n / w;", "const float loss = (n / w) / w;")],
        "wrong-output-slot": [("*(float *)output(c,2)->data = result.finalize.perplexity;", "*(float *)output(c,2)->data = result.finalize.accuracy;"),
                              ("*(float *)output(c,3)->data = result.finalize.accuracy;", "*(float *)output(c,3)->data = result.finalize.perplexity;")],
        "clamp-exp-overflow": [("const float perplexity = expf(loss);", "const float perplexity = fminf(expf(loss), FLT_MAX);")],
    }
    for name, edits in numeric.items():
        evidence = gate.numeric(name, changed(provider, edits), driver)
        print(f"E3-METRICS numeric mutant {name}: killed ({evidence[0]})")
    admission = {
        "skip-masked-target-validation": [("if (targets[i] < 0 || targets[i] >= 256)", "if (mask[i] && (targets[i] < 0 || targets[i] >= 256))")],
        "skip-masked-logit-validation": [("if (!isfinite(logits[i*256+j]))", "if (mask[i] && !isfinite(logits[i*256+j]))")],
        "omit-logit-finite-check": [("if (!isfinite(logits[i*256+j])) return domain(e,op);", "(void)logits[i*256+j];")],
        "omit-sum-finite-check": [("if (!isfinite(next_n)) return domain(e,op);", "/* removed next_N finite admission */")],
        "early-output-before-exp": [("const float perplexity = expf(loss);", "*(float *)output(c,0)->data = loss;\n  const float perplexity = expf(loss);")],
    }
    for name, edits in admission.items():
        gate.native(name, changed(provider, edits), driver)
        print(f"E3-METRICS admission/byte-atomicity mutant {name}: killed (native assertion)")
    equivalent = {
        "weight-from-token-counter-equivalent": [("result->accumulate.w = positive_zero(next_w);", "result->accumulate.w = (float)next_tokens;")],
        "two-term-reversal-equivalent": [("correct = correct + p0;\n  correct = correct + p1;", "correct = correct + p1;\n  correct = correct + p0;")],
        "canonical-bool-fma-equivalent": [("const float p0 = weight0 * hit0;", "const float p0 = fmaf(weight0, hit0, 0.0f);"),
                                         ("const float p1 = weight1 * hit1;", "const float p1 = fmaf(weight1, hit1, 0.0f);")],
        "local-f64-add-cast-equivalent": [("const float next_n = n + nb;", "const float next_n = (float)((double)n + (double)nb);")],
        "identical-error-f64-retry-equivalent": [("if (!isfinite(perplexity)) return domain(e,op);", "if (!isfinite(perplexity)) { volatile double retry = exp((double)loss); (void)retry; return domain(e,op); }")],
    }
    for name, edits in equivalent.items():
        mutant = changed(provider, edits)
        gate.numeric(name, mutant, driver, equivalent=True)
        gate.native(name, mutant, driver, equivalent=True)
        print(f"E3-METRICS compiled control {name}: equivalent in numerical/admission/atomicity gates")
    orchestration = {
        "batch-call-reorder": [("const float numerators[]={64.0f,0x1p-18f,0x1p-18f};", "const float numerators[]={0x1p-18f,0x1p-18f,64.0f};")],
        "batch-call-count": [("b<(mean_case?2u:3u)", "b<(mean_case?2u:2u)")],
        "mean-of-means": [("view(&counters[current][1],8,\"i64\",0,NULL),n.view,w.view,correct.view,ao[1]", "view(&counters[current][1],8,\"i64\",0,NULL),mean.view,w.view,correct.view,ao[1]")],
        "retained-f64-accumulator": [("const float numerators[]={64.0f,0x1p-18f,0x1p-18f};", "double wider=0.0;\n  const float numerators[]={64.0f,0x1p-18f,0x1p-18f};"),
            ("dispatch(r,CAP,OP(\"accumulate\"),3,ls,in,9,out,5);current=next;", "dispatch(r,CAP,OP(\"accumulate\"),3,ls,in,9,out,5);wider+=(double)bn;state[next][0]=(float)wider;current=next;")],
    }
    # The mutant deliberately finalizes each real batch before global reduction,
    # then averages those provider perplexities in its reporting orchestration.
    orchestration["exp-before-global-loss"] = [
        ("size_t current=0;\n  for(size_t b=0;b<3;b++)", "size_t current=0;float premature_ppl=0.0f;\n  for(size_t b=0;b<3;b++)"),
        ("REQUIRE(scalar(&correct)==1.0f && active==(b==0?2:1));", "REQUIRE(scalar(&correct)==1.0f && active==(b==0?2:1));\n    float local[4];et_kernel_tensor_view_v1 pi[]={n.view,w.view,correct.view},po[4];\n    for(size_t i=0;i<4;i++)po[i]=view(&local[i],4,\"f32\",0,NULL);\n    dispatch(e3,CAP,OP(\"finalize\"),3,ls,pi,3,po,4);premature_ppl+=local[2];"),
        ("values[i]=scalar(&metrics[i]);print_values(\"global.metrics\",values,4);", "values[i]=scalar(&metrics[i]);values[2]=premature_ppl/3.0f;print_values(\"global.metrics\",values,4);")]
    for name, edits in orchestration.items():
        evidence = gate.numeric(name, provider, changed(driver, edits))
        print(f"E3-METRICS orchestration mutant {name}: killed ({evidence[0]}; actual unmodified providers)")
    print(f"E3-METRICS MUTATION PASS: {len(numeric)} numeric, {len(admission)} admission/atomicity, {len(orchestration)} orchestration mutants; {len(equivalent)} equivalent controls")


if __name__ == "__main__":
    main()
