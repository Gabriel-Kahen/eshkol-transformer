"""Owned selector, IR/provider instrumentation, and installed-caller controls."""
import os
from pathlib import Path
import subprocess
import unittest
from tests.g3g_manual.check_public_g2_gate import check

ROOT = Path(__file__).resolve().parents[2]
BUILDER = (ROOT / 'scripts/build-e1b-consumer.sh').read_text()
RUNNER = (ROOT / 'scripts/test-g3g-manual-public-g2.sh').read_text()
CALLER = (ROOT / 'tests/g3g_manual/public_g2_runtime.esk').read_text()
PROVIDERS = (ROOT / 'scripts/g3g-manual-native-inputs.sh').read_text()


class GateSource(unittest.TestCase):
    def test_selected_source(self):
        check(BUILDER, RUNNER, CALLER, PROVIDERS)

    def test_wrong_or_missing_instrumentation_is_denied(self):
        changes = (
            (0, 'g3g-manual-public-aggregate &&\n       "${e3_private_test_sanitize}" == 0',
             'e3-private-aggregate &&\n       "${e3_private_test_sanitize}" == 0'),
            (0, 'package_native_cflags+=("${g3g_manual_test_sanitize_flags[@]}")', ':'),
            (0, '"${g3g_manual_test_sanitize_flags[@]}" \\\n  "${package_bridge_flags[@]}"', '"${package_bridge_flags[@]}"'),
            (0, 'G3G generated ASan IR exact streaming rewrite failed', 'unchecked IR'),
            (0, 'G3G generated private object lacks exact ASan IR evidence', 'unchecked object'),
            (0, 'G3G installed sanitizer changed noninstrumentation imports', 'unchecked imports'),
            (1, 'G3G_MANUAL_SANITIZE=1 /usr/bin/bash', 'G3G_MANUAL_SANITIZE=0 /usr/bin/bash'),
            (1, 'ASAN_OPTIONS=detect_leaks=1:halt_on_error=1', 'ASAN_OPTIONS=detect_leaks=0:halt_on_error=1'),
            (1, 'G2 evidence directory already exists', 'reuse evidence'),
            (2, 'first EOS generator closes before all output reads', 'close after read'),
            (3, 'g3g_manual_provider_mode=sanitize', 'g3g_manual_provider_mode=normal'),
        )
        for index, before, after in changes:
            with self.subTest(index=index, before=before):
                texts = [BUILDER, RUNNER, CALLER, PROVIDERS]
                self.assertIn(before, texts[index])
                texts[index] = texts[index].replace(before, after, 1)
                with self.assertRaises(ValueError):
                    check(*texts)
        with self.assertRaises(ValueError):
            check(BUILDER, RUNNER, CALLER + '\n(load "g3g_manual_package_root.esk")\n', PROVIDERS)

    def test_closed_source_inputs_and_public_caller(self):
        from tests.g3c4.test_public_g2_port_source import forms
        members = (ROOT / 'native/g3c4_public_g2_installed_gate_source_closure.txt').read_text().splitlines()
        self.assertEqual(len(members), len(set(members)))
        self.assertTrue(all((ROOT / name).is_file() for name in members))
        self.assertTrue(forms(CALLER))
        self.assertIn('"${e1b_cc}" "${e3_private_test_sanitize_flags[@]}" -c -x ir "${e1b_private_compile_ir}"', BUILDER)
        for source in ('tests/shared/prepare_eshkol_asan_ir.py', 'tests/e3_private/check_asan_ir.py'):
            self.assertIn(source, members)

    def test_actual_selector_rejects_wrong_tuple_and_values(self):
        start = BUILDER.index('g3g_manual_test_sanitize=0')
        end = BUILDER.index('e3_private_test_sanitize_flags=()', start)
        selector = 'die() { exit 73; }\n' + BUILDER[start:end]
        selector += '\nprintf "%s:%s:%s" "$g3g_manual_test_sanitize" "${g3g_manual_test_sanitize_flags[*]}" "$e3_private_test_sanitize"\n'
        for value, policy, e3, allowed in (
            ('0', 'g3g-manual-public-aggregate', '0', True),
            ('0', 'e3-private-aggregate', '1', True),
            ('1', 'g3g-manual-public-aggregate', '0', True),
            ('2', 'g3g-manual-public-aggregate', '0', False),
            ('1', 'g3g-public-aggregate', '0', False),
            ('1', 'e3-private-aggregate', '0', False),
            ('1', 'g3g-manual-public-aggregate', '1', False),
        ):
            env = dict(os.environ, G3G_MANUAL_SANITIZE=value,
                       package_policy=policy, e3_private_test_sanitize=e3)
            result = subprocess.run(['bash'], input=selector, env=env, text=True, capture_output=True)
            with self.subTest(value=value, policy=policy, e3=e3):
                self.assertEqual(result.returncode, 0 if allowed else 73)
                if allowed:
                    self.assertTrue(result.stdout.startswith(value + ':'))
                    self.assertTrue(result.stdout.endswith(':' + e3))
                    self.assertEqual('-fsanitize=address,undefined' in result.stdout, value == '1')


if __name__ == '__main__':
    unittest.main()
