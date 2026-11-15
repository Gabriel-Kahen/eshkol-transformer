"""Keep live predecessor inventories aligned with the current A2 source."""

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = "native/a2_attention_provider.c"
N3K = "tests/n3k/expected/predecessor_sources.sha256"
G3N = "tests/g3n/expected/predecessor_sources.sha256"
G3S = "tests/g3s/expected/predecessor_sources.sha256"
G3C4 = "tests/g3c4/expected/predecessor_sources.sha256"
E3 = "tests/e3_metrics/expected/predecessor_sources.sha256"
DIRECT = {
    N3K,
    G3N,
    G3S,
    G3C4,
}
NESTED = {(G3N, N3K)}


def inventories():
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "*predecessor_sources.sha256"], cwd=ROOT
    ).decode().strip("\0").split("\0")
    return {path: (ROOT / path).read_bytes() for path in paths}


def pin_errors(manifests, source_digest):
    errors = []
    direct, nested = Counter(), Counter()
    for manifest, content in manifests.items():
        for line in content.decode().splitlines():
            digest, separator, path = line.partition("  ")
            if not separator or len(digest) != 64:
                errors.append(f"{manifest}: malformed SHA-256 row")
                continue
            if path == SOURCE:
                direct[manifest] += 1
                if digest != source_digest:
                    errors.append(f"{manifest}: stale {SOURCE} pin")
            if path.endswith("predecessor_sources.sha256"):
                nested[(manifest, path)] += 1
                if path not in manifests:
                    errors.append(f"{manifest}: untracked {path} pin")
                elif digest != hashlib.sha256(manifests[path]).hexdigest():
                    errors.append(f"{manifest}: stale {path} pin")
    for manifest in sorted(DIRECT | direct.keys()):
        if manifest not in DIRECT:
            errors.append(f"{manifest}: unexpected {SOURCE} pin")
        elif direct[manifest] != 1:
            kind = "missing" if direct[manifest] == 0 else "duplicate"
            errors.append(f"{manifest}: {kind} {SOURCE} pin")
    for parent, child in sorted(NESTED | nested.keys()):
        if (parent, child) not in NESTED:
            errors.append(f"{parent}: unexpected {child} pin")
        elif nested[(parent, child)] != 1:
            kind = "missing" if nested[(parent, child)] == 0 else "duplicate"
            errors.append(f"{parent}: {kind} {child} pin")
    return errors


class AttentionPredecessorPinsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifests = inventories()
        cls.source_digest = hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest()

    def test_live_predecessor_closure(self):
        self.assertEqual(pin_errors(self.manifests, self.source_digest), [])

    def test_stale_or_missing_direct_pin_fails(self):
        changed = dict(self.manifests)
        row = f"{self.source_digest}  {SOURCE}\n".encode()
        self.assertIn(row, changed[N3K])
        changed[N3K] = changed[N3K].replace(row, b"0" * 64 + row[64:])
        self.assertIn(f"{N3K}: stale {SOURCE} pin", pin_errors(changed, self.source_digest))
        changed[N3K] = self.manifests[N3K].replace(row, b"")
        self.assertIn(f"{N3K}: missing {SOURCE} pin", pin_errors(changed, self.source_digest))

    def test_extra_or_duplicate_direct_pin_fails(self):
        changed = dict(self.manifests)
        row = f"{self.source_digest}  {SOURCE}\n".encode()
        changed[E3] += row
        self.assertIn(f"{E3}: unexpected {SOURCE} pin", pin_errors(changed, self.source_digest))
        changed = dict(self.manifests)
        changed[G3S] += row
        self.assertIn(f"{G3S}: duplicate {SOURCE} pin", pin_errors(changed, self.source_digest))

    def test_stale_or_missing_nested_pin_fails(self):
        changed = dict(self.manifests)
        row = f"{hashlib.sha256(changed[N3K]).hexdigest()}  {N3K}\n".encode()
        self.assertIn(row, changed[G3N])
        changed[G3N] = changed[G3N].replace(row, b"0" * 64 + row[64:])
        self.assertIn(f"{G3N}: stale {N3K} pin", pin_errors(changed, self.source_digest))
        changed[G3N] = self.manifests[G3N].replace(row, b"")
        self.assertIn(f"{G3N}: missing {N3K} pin", pin_errors(changed, self.source_digest))

    def test_extra_or_duplicate_nested_pin_fails(self):
        changed = dict(self.manifests)
        row = f"{hashlib.sha256(changed[N3K]).hexdigest()}  {N3K}\n".encode()
        changed[G3N] += row
        self.assertIn(f"{G3N}: duplicate {N3K} pin", pin_errors(changed, self.source_digest))
        changed = dict(self.manifests)
        changed[G3C4] += row
        self.assertIn(f"{G3C4}: unexpected {N3K} pin", pin_errors(changed, self.source_digest))


if __name__ == "__main__":
    unittest.main()
