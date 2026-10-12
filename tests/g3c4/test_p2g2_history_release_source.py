"""Source gate for the private F/T1 aggregate union and staged runtime witness."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def read(name: str) -> str:
    return (ROOT / name).read_text()


def validate(root: str, closure: str, build: str, fixture: str) -> None:
    t1 = 't1_published_release_private.esk'
    assert root.count(f'(load "{t1}")') == 1
    assert root.index('(load "m3_package_root.esk")') < root.index(
        f'(load "{t1}")')
    assert closure.splitlines().count(f'native/{t1}') == 1
    assert closure.index('native/t1_wave1_root.esk') < closure.index(
        f'native/{t1}') < closure.index('native/g3c4_public_g2_result_extension.esk')
    selected = build[build.index('"${package_native_source}" == '
                                 '"${PROJECT_ROOT}/native/t1_i64_shell.c" ]]; then'):
                     build.index('"${package_native_source}" == '
                                 '"${PROJECT_ROOT}/src/eshkol_transformer/g3c4_model_owner.c"')]
    assert '-DET_T1_PUBLISHED_RELEASE_PRIVATE' in selected
    assert '-DET_G3C4_T1_I1_EXACT_PAIR_PRIVATE' in selected
    assert read('native/g3g_manual_package_public_exports.txt').find(
        'private_release') == -1
    assert read('native/g3g_manual_package_public_strings.txt').find(
        't1-private-published-release') == -1

    ordered = (
        '(load "g3g_manual_package_root.esk")',
        '(define encoded',
        '(define input (g3c4-public-generation-input-create encoded))',
        '(t1-private-published-release! encoded)',
        '(not (t1-wave1-tensor-admitted? encoded))',
        '(t1-wave1-tokenizer-decode tokenizer encoded)',
        '(define native-ids',
        '(define first-id (first-id-after-release))',
        '(g3c4-public-p2g2-history-capture-internal!',
        '(g3c4-public-p2g2-history-admit-once-internal!',
        '(exercise first-id 1)',
        '(exercise (if (= first-id 255) 0 255) 2)',
    )
    positions = [fixture.index(fragment) for fragment in ordered]
    assert positions == sorted(positions)
    assert fixture.count('(load "g3g_manual_package_root.esk")') == 1
    assert '(load "t1_wave1_root.esk")' not in fixture
    assert fixture.count('(t1-shell-read encoded 0)') == 1
    assert fixture.index('(t1-shell-read encoded 0)') < positions[6]
    assert '(t1-shell-read encoded 1)' not in fixture[positions[3]:]
    assert '(vector-ref record 0) (vector-ref native-ids 0)' in fixture
    assert '(vector-ref record 1) (vector-ref native-ids 1)' in fixture
    assert 'history is admitted once' in fixture
    assert 'dead native length rejects' in fixture
    assert 'dead native read rejects' in fixture
    assert 'P1 shape cannot issue P2 input' in fixture
    assert 'foreign T1 release rejects' in fixture


class HistoryReleaseSource(unittest.TestCase):
    def setUp(self) -> None:
        self.root = read('native/g3g_manual_package_root.esk')
        self.closure = read('native/g3g_manual_package_source_closure.txt')
        self.build = read('scripts/build-e1b-consumer.sh')
        self.fixture = read('tests/g3c4/p2g2_history_release_private.esk')

    def test_selected_private_union(self) -> None:
        validate(self.root, self.closure, self.build, self.fixture)

    def test_early_release_or_late_t1_read_is_rejected(self) -> None:
        early = self.fixture.replace(
            '(define input (g3c4-public-generation-input-create encoded))',
            '(t1-private-published-release! encoded)\n'
            '(define input (g3c4-public-generation-input-create encoded))', 1)
        with self.assertRaises(AssertionError):
            validate(self.root, self.closure, self.build, early)
        late = self.fixture.replace('(define native-ids',
                                    '(t1-shell-read encoded 0)\n(define native-ids', 1)
        with self.assertRaises(AssertionError):
            validate(self.root, self.closure, self.build, late)

    def test_missing_feature_gate_or_duplicate_t1_root_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            validate(self.root, self.closure,
                     self.build.replace('-DET_T1_PUBLISHED_RELEASE_PRIVATE', ''),
                     self.fixture)
        with self.assertRaises(AssertionError):
            validate(self.root, self.closure, self.build,
                     self.fixture + '\n(load "t1_wave1_root.esk")\n')


if __name__ == '__main__':
    unittest.main()
