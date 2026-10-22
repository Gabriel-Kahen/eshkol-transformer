"""CPU4 source/oracle checks; these are not native execution evidence."""

import ast
from fractions import Fraction
import importlib.util
import runpy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cpu4_reference", Path(__file__).with_name("reference.py"))
ref = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ref)


class Cpu4SourceTests(unittest.TestCase):
    def test_rational_rounding_and_provider_only_order_witness(self):
        self.assertEqual(ref.rounded(Fraction(1,2**150)),0)
        self.assertEqual(ref.rounded(Fraction(3,2**150)),2)
        one = ref.bits(1)
        losses = tuple(map(ref.bits,(64,2**-18,2**-18,0)))
        result,_,_ = ref.objective(losses,(one,)*4)
        self.assertEqual(result[0],0x42800000)
        self.assertEqual(ref.rounded(sum(map(ref.value,losses))),0x42800001)
        self.assertEqual(ref.objective(tuple(reversed(losses)),(one,)*4)[0][0],0x42800001)

    def test_all_masks_and_four_seed_words(self):
        cases = list(ref.cases())
        self.assertEqual(len(cases),22)
        for mask,(boolean,losses,weights) in enumerate(cases[:15],1):
            self.assertEqual(boolean,1)
            self.assertEqual(weights,tuple(ref.bits((mask>>i)&1) for i in range(4)))
            reduced,num,mean=ref.objective(losses,weights)
            self.assertEqual(ref.value(reduced[1]),mask.bit_count())
            self.assertEqual(num,weights)
            self.assertEqual(len(mean),4)
        with self.assertRaisesRegex(ValueError,"positive weight"):
            ref.objective((0,)*4,(0,)*4)

    def test_fixed_mask_objective_derivatives(self):
        # Independent rational objective derivatives; masks are fixed, never
        # differentiated. Native CE/backward checks live in native.c.
        losses = (Fraction(2),Fraction(10),Fraction(3),Fraction(7))
        for _,_,words in ref.cases():
            weights=tuple(map(ref.value,words));total=sum(weights)
            delta=Fraction(1,1024)
            for i in range(4):
                high=list(losses);low=list(losses);high[i]+=delta;low[i]-=delta
                n_high=sum(a*b for a,b in zip(high,weights))
                n_low=sum(a*b for a,b in zip(low,weights))
                self.assertEqual((n_high-n_low)/(2*delta),weights[i])
                self.assertEqual((n_high/total-n_low/total)/(2*delta),weights[i]/total)

    def test_repeated_independent_header_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            files=[Path(directory)/f"reference-{i}.h" for i in range(2)]
            for file in files:
                subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("reference.py")),str(file)],check=True)
            self.assertEqual(files[0].read_bytes(),files[1].read_bytes())
        source=Path(__file__).with_name("reference.py").read_text()
        self.assertNotIn("native/",source)
        self.assertNotIn("import torch",source)

    def test_t2_arithmetic_and_mutation_anchors_preserved(self):
        l3s=(ROOT/"native/l3s_masked_objective_provider.c").read_text()
        before=subprocess.check_output(["git","-C",str(ROOT),"show",
            "b34268a88cc5be80e98c5484f49b94c46c6cc590:native/l3s_masked_objective_provider.c"],text=True)
        body=lambda source:source[source.index("static int evaluate("):source.index("/* Direct callbacks")]
        old=body(before).replace("float result[3]","float result[4]")
        new=body(l3s).replace("  if (c->request->shape[1] == 4u) return evaluate_four(c,op,result);\n","")
        self.assertEqual(new,old)
        for name,source in (("l3s/test_mutations.py",l3s),
                            ("e3_metrics/test_mutations.py",(ROOT/"native/e3_evaluation_metrics_provider.c").read_text())):
            tree=ast.parse((ROOT/"tests"/name).read_text())
            for node in ast.walk(tree):
                if (isinstance(node,ast.Assign) and isinstance(node.value,ast.Dict)
                    and any(isinstance(target,ast.Name) and target.id in
                            ("mutations","numeric","admission","equivalent") for target in node.targets)):
                    for value in node.value.values:
                        if not isinstance(value,ast.List):continue
                        for edit in value.elts:
                            if isinstance(edit,ast.Tuple) and isinstance(edit.elts[0],ast.Constant):
                                anchor=edit.elts[0].value
                                if isinstance(anchor,str):
                                    self.assertEqual(source.count(anchor),1,anchor)

    def test_exact_rows_scratch_and_bounded_scalar_checks(self):
        l3s=(ROOT/"native/l3s_masked_objective_provider.c").read_text()
        metrics=(ROOT/"native/e3_evaluation_metrics_provider.c").read_text()
        self.assertIn("float result[4]",l3s)
        self.assertIn("(r->shape[1] != 2u && r->shape[1] != 4u)",l3s)
        self.assertIn("memcpy(output(c,0)->data, result, 4u*sizeof(float));",l3s)
        self.assertIn("if (active < 1 || active > width)",metrics)
        self.assertLess(metrics.index("if (batches > 8192 || tokens > token_cap)"),
                        metrics.index("(width-1)*batches"))
        for source in (l3s,metrics):
            self.assertNotRegex(source,r"\b(double|malloc|calloc|realloc|free)\b")
            self.assertIn("{4,4,0,{0}}",source)
            self.assertNotRegex(source,r"\{2,4,0,\{0\}\}")
        # The established T2 metric arithmetic statements remain unique.
        for statement in ("const float next_n = n + nb;","const float next_w = w + wb;",
                          "const float next_c = correct + cb;","const float loss = n / w;",
                          "const float accuracy = correct / w;","const float perplexity = expf(loss);"):
            self.assertEqual(metrics.count(statement),1)

    def test_each_t4_mutant_has_one_concrete_source_change(self):
        source=(ROOT/"native/l3s_masked_objective_provider.c").read_text()
        mutants=runpy.run_path(str(Path(__file__).with_name("mutations.py")))["variants"]
        variants=list(mutants(source))
        self.assertEqual(len(variants),5)
        for name,changed in variants:
            self.assertNotEqual(changed,source,name)


if __name__=="__main__":unittest.main()
