"""Public G2 port source admission. Host controls separately execute real bodies.

These checks admit neither AOT compilation nor native tensor execution.
"""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
COMMIT = ROOT / 'native/g3c4_p2g2_terminal_commit_extension.esk'
DISPATCH = ROOT / 'native/g3c4_public_generation_dispatch_extension.esk'
RESULT = ROOT / 'native/g3c4_public_g2_result_extension.esk'


def forms(text):
    tokens = re.findall(r';[^\n]*|"(?:\\.|[^"\\])*"|[()]|[^\s()]+', text)
    stack, answer = [], []
    for token in tokens:
        if token.startswith(';'):
            continue
        if token == '(':
            stack.append([])
        elif token == ')':
            if not stack:
                raise ValueError('unmatched close')
            value = stack.pop()
            (stack[-1] if stack else answer).append(value)
        else:
            if not stack:
                raise ValueError('top-level atom')
            stack[-1].append(token)
    if stack:
        raise ValueError('unclosed form')
    return answer


def nodes(tree):
    if isinstance(tree, list):
        yield tree
        for child in tree:
            yield from nodes(child)


def calls(tree, name):
    return [node for node in nodes(tree) if node and node[0] == name]


def seam_valid(source):
    try:
        tree = forms(source)
        core = next(node for node in nodes(tree)
                    if len(node) == 2 and node[0] == 'core'
                    and isinstance(node[1], list) and node[1][:2] ==
                    ['lambda', ['generator', 'input', 'enrollment']])
        assert len(calls(tree, 'set!')) >= 8
        assert ['set!', 'installed?', '#t'] in list(nodes(tree))
        assert source.count('(lambda (stage roots)') == 2
        assert '(if (or installed? (not (procedure? stage))' in source
        assert '(not (= (vector-length roots) 7))' in source
        assert '(eq? root (vector-ref original-roots other))' in source
        assert '(vector ticket original-stage original-roots heads)' in source
        assert '(lambda (generator input) (core generator input #f))' in source
        # The only public calls in core are the fixed three prezero phases.
        stages = [call for call in calls(core, 'issuer') if len(call) > 2]
        assert [call[1] for call in stages] == ["'admit", "'bind", "'verify"]
        required = (
            '(eq? (vector-ref frame index)\n                                    (vector-ref final-frame index))',
            '(eq? (vector-ref (vector-ref enrollment-roots index) 0)\n                                        (vector-ref enrollment-heads index))',
            '(= (vector-length enrollment-ticket) 21)',
            '(= (vector-length enrollment-row) 12)',
            '(= (vector-length enrollment-anchor) 12)',
            '(not (eq? enrollment-row enrollment-anchor))',
            '(not (eq? enrollment-row output))',
            '(not (eq? enrollment-anchor output))',
            '(eq? (vector-ref enrollment-ticket 10) enrollment-row)',
            '(eq? (vector-ref enrollment-ticket 11) enrollment-anchor)',
            "(eq? (vector-ref enrollment-ticket 12) 'pending)",
            "(eq? (vector-ref enrollment-row 8) 'pending)",
            "(eq? (vector-ref enrollment-anchor 8) 'pending)",
            '(eq? (vector-ref enrollment-ticket 16) enrollment-authority)',
            '(eq? (vector-ref enrollment-binding 1) enrollment-authority)',
            '(eq? (vector-ref generator-entry 7) creation-policy)',
            '(= (vector-ref creation-policy index)\n                                             (vector-ref creation-words (- index 1)))',
            '(eq? (vector-ref live 10) live-authority)',
            '(eq? (vector-ref live 13) live-anchor-node)',
            '(eq? (vector-ref live 14) live-original-node)',
            '(eq? (vector-ref enrollment-row index)\n                                                 (vector-ref expected index))',
            '(eq? (vector-ref enrollment-anchor index)\n                                                 (vector-ref expected index))',
            '(if (vector-ref barrier 0) (exit 134))',
        )
        assert all(item in source for item in required)
        position = [source.index(part) for part in
                    ("(issuer 'verify", '(final-frame', '(final-checked',
                     '(status', '(g3c4-p2g2-native-terminal-commit')]
        assert position == sorted(position)
        zero = next(node for node in calls(core, 'if')
                    if node[1] == ['=', 'status', '0'])
        tail = zero[2]
        assert tail[:5] == ['begin', ['vector-set!', 'barrier', '0', '#t'],
                           ['vector-set!', 'enrollment-ticket', '12', "'live"],
                           ['vector-set!', 'enrollment-row', '8', "'live"],
                           ['vector-set!', 'enrollment-anchor', '8', "'live"]]
        assert not calls(tail, 'issuer')
        assert not calls(tail, 'core')
        return required
    except (AssertionError, ValueError, StopIteration, IndexError):
        return False


class PublicG2Port(unittest.TestCase):
    def test_closed_enrollment_and_mutation_controls(self):
        source = COMMIT.read_text()
        required = seam_valid(source)
        self.assertTrue(required)
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertFalse(seam_valid(source.replace(fragment, '#t', 1)))
        for fragment in ('(set! installed? #t)',
                         '(eq? root (vector-ref original-roots other))',
                         "(vector-set! enrollment-ticket 12 'live)",
                         "(vector-set! enrollment-row 8 'live)",
                         "(vector-set! enrollment-anchor 8 'live)"):
            with self.subTest(fragment=fragment):
                self.assertFalse(seam_valid(source.replace(fragment, '#t', 1)))
        bad = source.replace('(vector-set! barrier 0 #t)',
                             '(vector-set! barrier 0 #t)\n                  (issuer \'verify)', 1)
        self.assertFalse(seam_valid(bad))

    def test_secret_profile_and_exact_failed_attempt_restore(self):
        source = DISPATCH.read_text()
        tree = forms(source)
        self.assertEqual(len(calls(tree, 'g3c4-p2g2-terminal-enrollment-route-private')), 1)
        self.assertNotIn('terminal-close-core-private', source)
        self.assertNotIn("'restore", source)
        self.assertIn('(eq? (vector-ref ticket 0) secret)', source)
        self.assertIn('(eq? (car (vector-ref g3c4-public-g2-tickets 0))\n                            ticket)', source)
        self.assertIn("(eq? profile 'diagnostic-c4-p2g2)", source)
        self.assertIn('(= (vector-ref policy 5) 2)', source)
        self.assertIn("(g3c4-public-g2-rng-result-live rng 'result-rng operation)", source)
        self.assertIn('(g3c4-native-rng-word-checked native index operation)', source)
        self.assertIn('(vector-ref words (+ index 1))', source)
        self.assertNotIn('g3c4-native-rng-seed', source)
        self.assertIn('(m3t-exact-i64? (vector-ref policy 6))', source)
        self.assertIn('(<= (vector-ref policy 6) 255)', source)
        self.assertLess(source.index('(let ((route'), source.index('(set! g3c4-public-generator-generate'))
        for name in ('tickets', 'output-rows', 'output-anchors', 'output-bindings'):
            self.assertRegex(source, rf'\(eq\? \(vector-ref g3c4-public-g2-{name} 0\) next-')
        self.assertIn("(eq? (vector-ref issued 12) 'pending)", source)

    def test_exact_typed_ownership_and_unchanged_public_manifest(self):
        source = RESULT.read_text()
        tree = forms(source)
        self.assertNotIn('et_g3c4_private_terminal_result_tensor_create_v1', source)
        self.assertNotIn('g3c4-p2g2-terminal-origin-', source)
        self.assertEqual(len(calls(tree, 'g3c4-p2g2-typed-create-private')), 1)
        self.assertEqual(len(calls(tree, 'g3c4-p2g2-terminal-ids-clone-private')), 1)
        self.assertEqual(len(calls(tree, 'g3c4-p2g2-typed-copy-private')), 1)
        self.assertIn('(g3c4-p2g2-native-terminal-snapshot native bytes 72)', source)
        self.assertIn('(g3c4-p2g2-terminal-output-entry-live shell operation)', source)
        self.assertIn("(eq? kind 'result-rng)", source)
        for suffix in ('public_exports', 'public_strings', 'private_renames', 'defined_symbols'):
            name = 'native/g3g_manual_package_' + suffix + '.txt'
            baseline = subprocess.check_output(['git', 'show', 'c71d39a:' + name], cwd=ROOT)
            self.assertEqual((ROOT / name).read_bytes(), baseline)
        adapter = 'native/g3c4_public_output_adapter_extension.esk'
        self.assertEqual((ROOT / adapter).read_bytes(), subprocess.check_output(
            ['git', 'show', 'c71d39a:' + adapter], cwd=ROOT))

    def test_owned_source_closure_and_fixture_ports(self):
        members = (ROOT / 'native/g3c4_public_g2_port_source_closure.txt').read_text().splitlines()
        self.assertEqual(len(members), len(set(members)))
        self.assertTrue(all((ROOT / item).is_file() for item in members))
        for name in ('native/g3g_manual_package_root.esk', 'native/g3c4_public_g2_result_extension.esk',
                     'tests/g3c4/p2_g2_public_result_test.esk', 'tests/g3c4/p2_g2_public_ticket_close_test.esk',
                     'tests/g3c4/p2_g2_public_ticket_host_test.scm',
                     'tests/g3c4/p2_g2_public_typed_result_host_test.scm'):
            self.assertTrue(forms((ROOT / name).read_text()), name)
        fixture = (ROOT / 'tests/g3c4/p2_g2_public_result_test.esk').read_text()
        for name in ('g3g-public-generation-output-ids', 'g3g-public-generation-output-lengths',
                     'g3g-public-generation-output-cache-lengths', 'g3g-public-generation-output-text',
                     'g3g-public-generation-output-rng', 'g3g-public-generation-tensor-release',
                     'g3g-public-generation-output-release', 'g3g-public-generation-rng-release'):
            self.assertIn(name, fixture)
        self.assertNotIn('terminal-close-origin-private', fixture)
        self.assertIn('g3c4-p2g2-terminal-commit-private!', fixture)


if __name__ == '__main__':
    unittest.main()
