"""Pinned host-only source-union preflight; no Eshkol compilation or capture."""

from pathlib import Path
from collections import Counter
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SEAM = "bba972710ee49a3321116079ae12685d0c682597"
REPLAY = "01a584c6d2546b469c5ffd584aab47f55cbdea99"
SEAM_CLOSURE = "native/g3r_c2_receipt_tr3_restore_source_closure.txt"
REPLAY_CLOSURE = "native/g3r_trained_replay_source_closure.txt"
IMPORTS = (
    "native/g3r_rng_import_extension.esk",
    "native/g3r_replayed_p1_g1_extension.esk",
)
REPLAY_ONLY = frozenset((
    "native/g3r_trained_replay_root.esk",
    "native/g3r_trained_replay_compare_extension.esk",
    "native/g3r_trained_replay_producer_extension.esk",
    *IMPORTS,
))
LOAD = re.compile(r'\(load\s+"([^"]+)"\)')


def git_show(ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, text=True
    )


def git_blob(ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{ref}:{path}"], cwd=ROOT, text=True
    ).strip()


def paths(text: str) -> tuple[str, ...]:
    result = tuple(text.splitlines())
    if not result or any(not p or p.startswith("#") for p in result):
        raise AssertionError("empty or annotated source closure")
    if len(set(result)) != len(result):
        raise AssertionError("duplicate source in closure")
    return result


def check_loads_once(closure: tuple[str, ...], read) -> None:
    loaded = Counter(name for path in closure for name in LOAD.findall(read(path)))
    if any(count != 1 for count in loaded.values()):
        raise AssertionError("duplicate transitive load")
    for name in ("cli3_generate_root.esk", "cli3_root.esk",
                 "g3r_trained_pair_producer_root.esk"):
        if loaded[name] != 1:
            raise AssertionError("private package entry dependency changed")


def check_union(seam: tuple[str, ...], replay: tuple[str, ...],
                seam_blob, replay_blob, read_seam, read_replay) -> tuple[str, ...]:
    if len(seam) != 101 or len(replay) != 97:
        raise AssertionError("pinned closure count changed")
    if seam[0] != "native/g3r_c2_receipt_tr3_restore_root.esk":
        raise AssertionError("seam entry root changed")
    if replay[0] != "native/g3r_trained_replay_root.esk":
        raise AssertionError("replay entry root changed")
    if set(replay) - set(seam) != REPLAY_ONLY:
        raise AssertionError("replay-only source set changed")
    shared = set(seam) & set(replay)
    if len(shared) != 92 or any(seam_blob(p) != replay_blob(p) for p in shared):
        raise AssertionError("shared single-registry source blob drift")
    check_loads_once(seam, read_seam)
    check_loads_once(replay, read_replay)
    if LOAD.findall(read_seam(seam[0])) != [
        "g3r_trained_pair_reader_root.esk",
        "g3r_c2_receipt_tr3_restore_extension.esk",
    ]:
        raise AssertionError("seam root load order changed")
    if LOAD.findall(read_replay(replay[0])) != [
        "g3r_trained_pair_producer_root.esk",
        "g3r_rng_import_extension.esk",
        "g3r_replayed_p1_g1_extension.esk",
        "g3r_trained_replay_compare_extension.esk",
        "g3r_trained_replay_producer_extension.esk",
    ]:
        raise AssertionError("producer root load order changed")
    if any(LOAD.search(read_replay(p)) for p in IMPORTS):
        raise AssertionError("consumer replay leaf loads another source")
    consumer = seam + IMPORTS
    if len(set(consumer)) != 103:
        raise AssertionError("consumer union duplicates a source")
    if "native/g3r_trained_replay_root.esk" in consumer:
        raise AssertionError("consumer would reload producer registry")
    return consumer


def main() -> None:
    seam = paths(git_show(SEAM, SEAM_CLOSURE))
    replay = paths(git_show(REPLAY, REPLAY_CLOSURE))
    consumer = check_union(
        seam, replay,
        lambda p: git_blob(SEAM, p), lambda p: git_blob(REPLAY, p),
        lambda p: git_show(SEAM, p), lambda p: git_show(REPLAY, p),
    )
    print(f"G3-R SOURCE UNION PREFLIGHT PASS: shared=92 producer={len(replay)} consumer={len(consumer)}")


if __name__ == "__main__":
    main()
