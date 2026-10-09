"""Source-only topology and disabled-launch guard for the F loader outer leaf."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "native/g3c4_public_g2_loader_collector_source_closure.txt"
CLOSURE = ROOT / "native/g3c4_public_g2_loader_outer_source_closure.txt"
ADDED = (
    "tests/g3c4/f_loader_guest.py",
    "tests/g3c4/f_loader_outer.py",
    "tests/g3c4/test_f_loader_outer.py",
    "tests/g3c4/check_f_loader_outer.py",
    "docs/g3/G3_C4_F_LOADER_OUTER_SOURCE_LEAF.md",
    "native/g3c4_public_g2_loader_outer_source_closure.txt",
)
UNCHANGED = {
    "tests/g3c4/f_image_loader_collect.py":
        "98cc07a939834466fe77e8c43283159e67f70318bdbdd90b28cd267ff219b0c9",
    "tests/g3c4/f_artifact_attest.py":
        "5aac55665ad5a1767c54591ec3530667cced310f07948257521f34ed43999e78",
    "scripts/common.sh":
        "2749689e6301cad6e489de4b23e0f9b1ec7c9627f47dd1c7007d9ed8e5955e83",
    "toolchain/eshkol.lock":
        "8c168f5d560781245f6d4c73f695e72cbafa413b606d50bc7c36d2119acd9efe",
}
OLD = ("  A separate [source-only F image-loader collector](g3/G3_C4_F_IMAGE_LOADER_COLLECTOR_SOURCE_LEAF.md)\n"
       "  stages bounded readelf/loader raw receipts against the accepted f31 tool and\n"
       "  runtime inventories. The actual image observation, ROOT pinning, package\n"
       "  build, installed G2 execution and sanitizer close remain **unrun**.\n")
NEW = OLD + ("\n  An [inactive test-only F loader outer source](g3/G3_C4_F_LOADER_OUTER_SOURCE_LEAF.md)\n"
             "  specifies the exact f31 guest mounts and bounded owned Docker-client and\n"
             "  container closure path. Only a future immutable ROOT wrapper may pin its\n"
             "  external decision SHA and invoke it; no\n"
             "  Docker launch, loader result, public package or sanitizer proof is claimed.\n"
             "  Its exact nested stage-child path and stopped-state/Docker-wait checks remain\n"
             "  source-only; the accepted attester's default path is unchanged.\n"
             "  The same canonical source-admission JSON and two review seals are checked on\n"
             "  both host and guest bundle surfaces; the image loader is still unobserved.\n"
             "  A scoped source successor raises only the F loader profile's file-size cap\n"
             "  from 16 MiB to 128 MiB, above the preserved 85,874,424-byte static archive;\n"
             "  its actual container profile and copy remain unrun.\n"
             "  A further optional ROOT monitor seam samples the selected aggregate output\n"
             "  during owned capture and leaves physical Docker cleanup independent of a\n"
             "  monitor failure; source controls pass, while live overshoot and runtime\n"
             "  admission remain open.\n"
             "  The callback is deliberately suspended across the single Docker `create`\n"
             "  receipt after a pre-create check, then resumes only after a genuine CID is\n"
             "  retained; an ambiguous failed create still requires separate closure review.\n"
             "  A source-only exact-name/CID/effective-profile recovery route now attempts\n"
             "  owned cleanup after a failed create receipt; absent or foreign candidates\n"
             "  remain quarantined failures, and no actual Docker recovery was run.\n")


def validate(base: list[str], closure: list[str], source: str,
             guest: str, roadmap: str, original_roadmap: str) -> None:
    assert len(base) == 227 and closure == base + list(ADDED)
    assert len(closure) == len(set(closure))
    assert all((ROOT / name).is_file() for name in closure)
    for name, expected in UNCHANGED.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert original_roadmap.count(OLD) == 1
    assert roadmap == original_roadmap.replace(OLD, NEW)
    ast.parse(source, feature_version=(3, 10))
    ast.parse(guest, feature_version=(3, 10))
    for anchor in ('if __name__ == "__main__":',
                   'external immutable ROOT wrapper required',
                   'start_new_session=True',
                   'os.WNOWAIT', 'os.wait4(', 'collector.signal_owned(',
                   '"--network", "none"', '"--read-only"',
                   '"--memory", "4g"', '"--pids-limit", "128"',
                   '"fsize=134217728:134217728"',
                   '"monitor denied:', 'def active_capture(',
                   'created = checked(create_argv(decision), evidence / "create")',
                   'recover_failed_create(decision, image, evidence)',
                   '"--tmpfs"', '"--attach"', '"inspect"', '"rm"',
                   '"initial-absence"', '"closure-inspect"',
                   '"docker-wait"', 'stopped_exit(', 'wait_exit(',
                   '"candidate", "/stage", False',
                   '"source_admission_sha256"'):
        assert anchor in source, anchor
    for anchor in ('attest.main()', 'collector.collect(', '/out/build-off/eshkol-run',
                   'collector.TOOL_PINS', 'collector.RUNTIME_PINS',
                   '"--outer-stage-child"', '"/stage/candidate"'):
        assert anchor in guest or anchor in source, anchor
    attester = (ROOT / "tests/g3c4/f_artifact_attest.py").read_text()
    assert 'parser.add_argument("--outer-stage-child", action="store_true")' in attester
    assert 'Path("/stage/candidate") if args.outer_stage_child' in attester
    assert 'else Path("/candidate")' in attester
    assert attester.count('decision, candidate_path)') == 5
    for anchor in ('outer_bundle_root: Path | None = None',
                   '"/bundle/reviews/independent/SHA256SUMS-INDEPENDENT-SOURCE-REVIEW"',
                   '"/bundle/reviews/root/SHA256SUMS-ROOT-REVIEW"',
                   'review_path.relative_to("/bundle")'):
        assert anchor in attester, anchor


def main() -> None:
    original = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(ROOT), "show",
         "fb752684cfd65e20fbc1b8526709fabbf898ea91:docs/ROADMAP.md"],
        check=True, capture_output=True, timeout=30).stdout.decode()
    validate(BASE.read_text().splitlines(), CLOSURE.read_text().splitlines(),
             (ROOT / ADDED[1]).read_text(), (ROOT / ADDED[0]).read_text(),
             (ROOT / "docs/ROADMAP.md").read_text(), original)
    print("inactive F loader outer source: PASS")


if __name__ == "__main__":
    main()
