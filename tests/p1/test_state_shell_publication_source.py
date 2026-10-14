"""Source contract for state-shell publication without a fallible index tail."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
TRUSTED = ROOT / "internal/p1/lib/transformer/module.esk"
TEMPLATE = ROOT / "templates/p1/module_roots.esk.tmpl"
PROOF = ROOT / "scripts/check-p1-registry-publication.sh"


def section(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def publication_contract(source: str) -> bool:
    lookup = section(source, "(define (shell-entry ", "(define (index-published-shell!")
    publish = section(source, "(define (shell-for-raw kind raw)",
                      "(define (state-entry-shell-for-raw")
    state = section(source, "(define (state-from-flat ",
                    "(define (prepare-module-state-dict")
    skip = """(if (eq? kind 'state)
             #t
             (index-published-shell!"""
    return (
        "(shell-registry-current)" in lookup
        and "(eq? value (vector-ref (car entries) 0))" in lookup
        and publish.count(skip) == 1
        and publish.index("(vector-set! shell-registry-root 0 next-registry)")
        < publish.index(skip)
        < publish.index("(vector-ref (car (shell-registry-current)) 0)")
        and "(set! state-shell (shell-for-raw 'state state))" in state
        and "(if state-shell" in state
    )


class StateShellPublicationSourceTest(unittest.TestCase):
    def test_generated_and_template_preserve_exact_registry_fallback(self):
        self.assertTrue(publication_contract(TRUSTED.read_text()))
        self.assertTrue(publication_contract(TEMPLATE.read_text()))

    def test_removed_state_skip_is_rejected(self):
        source = TRUSTED.read_text()
        indexed = """(index-published-shell!
               (canonical-published-shell (car (shell-registry-current)))
               (car (shell-registry-current)))"""
        old = """(if (eq? kind 'state)
             #t
             """ + indexed + ")"
        mutant = source.replace(old, indexed, 1)
        self.assertNotEqual(mutant, source)
        self.assertFalse(publication_contract(mutant))

    def test_removed_authoritative_scan_is_rejected(self):
        source = TRUSTED.read_text()
        lookup = section(source, "(define (shell-entry ",
                         "(define (index-published-shell!")
        mutant = source.replace(lookup,
                                lookup.replace("(shell-registry-current)", "'()"), 1)
        self.assertNotEqual(mutant, source)
        self.assertFalse(publication_contract(mutant))

    def test_runtime_overlay_forces_post_publication_cut(self):
        script = PROOF.read_text()
        self.assertIn("'post-publication-state-index-reached", script)
        self.assertIn("(vector-ref entry 1) 'state", script)
        self.assertIn("P1 INDEX FALLBACK PASS: 50 checks", script)

    def test_runtime_overlay_builds_from_exact_generated_source(self):
        script = PROOF.read_text()
        marker = ('python3 - "${generated}" '
                  '"${tmp}/overlay/transformer/module.esk" <<\'PY\'\n')
        overlay_program = script.split(marker, 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "module.esk"
            subprocess.run([sys.executable, "-c", overlay_program,
                            str(TRUSTED), str(target)], check=True)
            overlay = target.read_text()
            self.assertIn("'post-publication-state-index-reached", overlay)
            self.assertIn("(vector-ref p1-trusted-surface 74)", overlay)


if __name__ == "__main__":
    unittest.main()
