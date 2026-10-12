#!/usr/bin/env python3
"""Audit the native first-frame leaf's ordered raw command receipts."""

from hashlib import sha256
from pathlib import Path
from collections import Counter
import json
import re
import stat
import sys

FILES = ("label", "cwd", "argv.nul", "exit", "stdout.raw", "stderr.raw")
LABELS = (
    "toolchain-verify", "static",
    *(f"owner-{mode}-{step}" for mode in ("off", "on")
      for step in ("compile", "nm", "awk", "sort")),
    "symbol-parity", "invalid-compile", "invalid-diagnostic",
    "feature-off-compile", "feature-off-run", "feature-off-stderr-empty",
    "feature-off-pass",
    *(f"{mode}-{step}" for mode in ("normal", "sanitize")
      for step in ("compile", "run", "stderr-empty", "pass")),
    "oracle", "repeat-run", "repeat-stderr-empty",
    "repeat-parity", "sanitize-parity", "receipts-audit",
)
ALIASES = {
    "static": ("static.stdout", "static.stderr"),
    "owner-off-sort": ("off.symbols", None),
    "owner-on-sort": ("on.symbols", None),
    "invalid-compile": ("invalid.stdout", "invalid.stderr"),
    "feature-off-run": ("feature-off.stdout", "feature-off.stderr"),
    "normal-run": ("normal.stdout", "normal.stderr"),
    "sanitize-run": ("sanitize.stdout", "sanitize.stderr"),
    "oracle": ("oracle.stdout", "oracle.stderr"),
    "repeat-run": ("repeat.stdout", "repeat.stderr"),
    "receipts-audit": ("receipts-audit.stdout", "receipts-audit.stderr"),
}
MANIFEST = "SHA256SUMS-RECEIPTS"


def nested_profile(evidence: Path) -> dict[str, str]:
    path = evidence / "nested-profile.tsv"
    regular(path)
    match = re.fullmatch(
        rb"cc_default\t([01])\ncxx_default\t([01])\n"
        rb"cc_program\t([A-Za-z0-9._+\-]+)\n"
        rb"cxx_program\t([A-Za-z0-9._+\-]+)\n"
        rb"llvm_program\t([A-Za-z0-9._+\-]+)\n"
        rb"cc_relative\t([01])\ncxx_relative\t([01])\n",
        path.read_bytes())
    if match is None:
        raise ValueError("nested compiler choice profile differs")
    values = [item.decode() for item in match.groups()]
    return dict(zip(("cc_default", "cxx_default", "cc_program", "cxx_program",
                     "llvm_program", "cc_relative", "cxx_relative"), values))


def expected_common(profile: dict[str, str]) -> Counter:
    result = Counter()
    for key, count in {
        "supported_os": 1, "supported_arch": 1, "llvm_version": 2,
        "eshkol_commit": 2, "eshkol_version": 1, "clang_version": 3,
        **{f"minimum_{name.lower()}_version": 1
           for name in ("CMake", "Ninja", "Make", "Bash", "Git")},
        "supported_cc": int(profile["cc_default"]),
        "supported_cxx": int(profile["cxx_default"]),
    }.items():
        result[f"lock-{key}-count"] += count
        result[f"lock-{key}-value"] += count
    for key, count in {
        "eshkol_commit": 1, "eshkol_source_dir": 1, "cc_path": 2,
        "cxx_path": 4, "eshkol_binary_sha256": 1, "llvm_version": 1,
        "cc_version": 2, "cxx_version": 2,
        **{f"{name.lower()}_version": 1
           for name in ("CMake", "Ninja", "Make", "Bash", "Git")},
    }.items():
        result[f"tsv-{key}-count"] += count
        result[f"tsv-{key}-value"] += count
    for label in (
        "host-uname-m", "host-uname-s", "llvm-version",
        "eshkol-git-head", "eshkol-git-status", "eshkol-version",
        "source-realpath", "cache-home", "cache-home-realpath",
        "cache-cc", "cache-cxx", "provenance-cc-realpath",
        "cache-cc-realpath", "provenance-cxx-realpath",
        "cache-cxx-realpath", "eshkol-sha256", "eshkol-sha256-awk",
        "cc-version", "cc-version-sed", "cxx-version", "cxx-version-sed",
    ):
        result[label] += 1
    for name in ("CMake", "Ninja", "Make", "Bash", "Git"):
        result[f"minimum-{name}-sort"] += 1
        result[f"minimum-{name}-head"] += 1
    for kind in ("cc", "cxx"):
        if profile[f"{kind}_relative"] == "1":
            for step in ("dirname", "basename"):
                result[f"resolve-resolved_{kind}-{step}"] += 1
    return +result


def regular(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing receipt file: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"missing or unsafe receipt file: {path}")


def receipt_root(evidence: Path) -> Path:
    pointer = evidence / "commands-path.txt"
    regular(pointer)
    value = pointer.read_text()
    root = Path(value.rstrip("\n"))
    if value != f"{root}\n" or root.parent != evidence or \
            not root.name.startswith("commands.") or root.is_symlink() or \
            not root.is_dir():
        raise ValueError("invalid command root pointer")
    return root


def expected_program(label: str) -> str:
    if label == "toolchain-verify":
        return "bash"
    if label in ("static", "oracle", "receipts-audit"):
        return "python3"
    if label.endswith("-compile"):
        return "clang-21"
    if label.endswith("-nm"):
        return "nm"
    if label.endswith("-awk"):
        return "awk"
    if label.endswith("-sort") or label in ("normal-run", "sanitize-run"):
        return "env"
    if label.endswith("-stderr-empty"):
        return "test"
    if label.endswith("-pass") or label == "invalid-diagnostic":
        return "grep"
    if label in ("symbol-parity", "repeat-parity", "sanitize-parity"):
        return "cmp"
    if label in ("feature-off-run", "repeat-run"):
        return label.removesuffix("-run").replace("repeat", "normal")
    raise ValueError(f"unexpected label: {label}")


def nested_program(label: str, family: str, profile: dict[str, str]) -> str:
    if family == "static":
        return Path(sys.executable).name
    if label.startswith(("lock-", "tsv-")) or label in (
        "cache-home", "cache-cc", "cache-cxx", "eshkol-sha256-awk"):
        return "awk"
    if label.startswith("host-uname-"):
        return "uname"
    if label == "llvm-version":
        return profile["llvm_program"]
    if label.startswith("eshkol-git-"):
        return "git"
    if label == "eshkol-version":
        return "eshkol-run"
    if label.endswith("-realpath") or label == "source-realpath":
        return "readlink"
    if label == "eshkol-sha256":
        return "sha256sum"
    if label == "cc-version":
        return profile["cc_program"]
    if label == "cxx-version":
        return profile["cxx_program"]
    if label.startswith("resolve-resolved_"):
        return label.rsplit("-", 1)[-1]
    if label.endswith("-sed"):
        return "sed"
    if label.endswith("-sort"):
        return "sort"
    if label.endswith("-head"):
        return "head"
    raise ValueError(f"unexpected nested label: {label}")


def check_nested_argv(label: str, family: str, args: list[str],
                      profile: dict[str, str]) -> None:
    if not args or Path(args[0]).name != nested_program(label, family, profile):
        raise ValueError(f"nested executable differs: {label}")
    if family == "static":
        predecessor = {
            "output-reservation": "check-g3c4-output-reservation.py",
            "p2g1-pending": "check-g3c4-p2g1-pending.py",
        }.get(label)
        if predecessor is None or args != [
            sys.executable, str(Path(__file__).resolve().parents[2] / "scripts" / predecessor)
        ]:
            raise ValueError(f"nested static argv differs: {label}")
    elif label.startswith(("lock-", "tsv-")):
        family_name, key, kind = label.split("-", 2)
        if kind not in ("count", "value") or len(args) != 7 or \
                args[1:4] != ["-F", "\\t", "-v"] or \
                args[4] != f"key={key}" or \
                (kind == "count" and "count++" not in args[5]) or \
                (kind == "value" and "print $2" not in args[5]) or \
                (family_name == "lock" and
                 args[6] != str(Path(__file__).resolve().parents[2] / "toolchain/eshkol.lock")) or \
                (family_name == "tsv" and
                 Path(args[6]).name != "eshkol-transformer-provenance.tsv"):
            raise ValueError(f"nested lookup argv differs: {label}")
    elif label in ("host-uname-m", "host-uname-s"):
        if args[1:] != ["-" + label[-1]]:
            raise ValueError(f"nested uname argv differs: {label}")
    elif label.endswith("-version") and label in (
        "llvm-version", "eshkol-version", "cc-version", "cxx-version"):
        if args[1:] != ["--version"]:
            raise ValueError(f"nested version argv differs: {label}")
    elif label.startswith("eshkol-git-"):
        if len(args) < 5 or args[1] != "-C" or \
                args[3:] != (["rev-parse", "HEAD"] if label == "eshkol-git-head"
                             else ["status", "--porcelain", "--untracked-files=all"]):
            raise ValueError(f"nested Git argv differs: {label}")
    elif label.endswith("-realpath") or label == "source-realpath":
        if len(args) != 3 or args[1] != "-f":
            raise ValueError(f"nested realpath argv differs: {label}")
    elif label.startswith("minimum-"):
        if args[1:] != (["-V"] if label.endswith("-sort") else ["-n", "1"]):
            raise ValueError(f"nested minimum argv differs: {label}")
    elif label.endswith("-sed"):
        if len(args) != 3 or args[1] != "-n" or "version" not in args[2]:
            raise ValueError(f"nested sed argv differs: {label}")
    elif label == "eshkol-sha256":
        if len(args) != 2 or Path(args[1]).name != "eshkol-run":
            raise ValueError("nested Eshkol hash argv differs")
    elif label == "eshkol-sha256-awk":
        if args[1:] != ["{ print $1 }"]:
            raise ValueError("nested Eshkol hash parse differs")
    elif label in ("cache-home", "cache-cc", "cache-cxx"):
        if len(args) != 4 or args[1] != "-F=" or \
                Path(args[3]).name != "CMakeCache.txt":
            raise ValueError(f"nested cache argv differs: {label}")
    elif label.startswith("resolve-resolved_"):
        kind = label.split("_", 1)[-1].split("-", 1)[0]
        if kind not in ("cc", "cxx") or len(args) != 3 or \
                args[1] != "--" or args[2].startswith("/") or \
                Path(args[2]).name != profile[f"{kind}_program"]:
            raise ValueError(f"nested relative resolver argv differs: {label}")


def nested_inventory(evidence: Path, complete: bool):
    profile_file = evidence / "nested-profile.tsv"
    if complete or profile_file.exists():
        profile = nested_profile(evidence)
        expected = expected_common(profile)
    else:
        profile = {}
        expected = Counter()
    static_expected = Counter({"output-reservation": 1, "p2g1-pending": 1})
    records = []
    incomplete = []
    for family, filename, prefix, required in (
        ("common", "nested-path.txt", "nested.", expected),
        ("static", "static-nested-path.txt", "static-nested.", static_expected),
    ):
        pointer = evidence / filename
        if not pointer.exists() and not complete:
            incomplete.extend(
                f"missing {family}:{label}:{count}"
                for label, count in sorted(required.items())
            )
            continue
        regular(pointer)
        value = pointer.read_text()
        directory = Path(value.rstrip("\n"))
        if value != f"{directory}\n" or directory.parent != evidence or \
                not directory.name.startswith(prefix) or directory.is_symlink() or \
                not directory.is_dir():
            raise ValueError(f"unsafe {family} nested root")
        seen = Counter()
        for child in sorted(directory.iterdir()):
            match = re.fullmatch(
                rf"{family}-([A-Za-z0-9_-]+)\.([A-Za-z0-9]+)", child.name)
            if not match or child.is_symlink() or not child.is_dir():
                raise ValueError(f"unsafe nested child receipt: {child}")
            label = match.group(1)
            if label not in required:
                raise ValueError(f"extra nested child: {label}")
            seen[label] += 1
            if seen[label] > required[label]:
                raise ValueError(f"extra nested child: {label}")
            files = {part.name for part in child.iterdir()}
            if files != set(FILES):
                if complete or not files <= set(FILES):
                    raise ValueError(f"nested child receipt shape differs: {label}")
                incomplete.append(f"{family}:{label}:{','.join(sorted(set(FILES)-files))}")
            for name in files:
                regular(child / name)
            if "label" in files and (child / "label").read_text() != label + "\n":
                raise ValueError(f"nested child label differs: {label}")
            if "cwd" in files:
                cwd = (child / "cwd").read_text()
                if not cwd.startswith("/") or not cwd.endswith("\n") or \
                        Path(cwd[:-1]).resolve() != Path(cwd[:-1]):
                    raise ValueError(f"nested cwd differs: {label}")
            if "argv.nul" in files:
                argv = (child / "argv.nul").read_bytes()
                if not argv or not argv.endswith(b"\0") or b"\0\0" in argv:
                    raise ValueError(f"nested NUL argv differs: {label}")
                try:
                    args = [piece.decode() for piece in argv[:-1].split(b"\0")]
                except UnicodeDecodeError as exc:
                    raise ValueError(f"nested argv encoding differs: {label}") from exc
                check_nested_argv(label, family, args, profile)
            if "exit" in files:
                value = (child / "exit").read_bytes()
                if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", value) or \
                        int(value) > 255 or (complete and int(value) != 0):
                    raise ValueError(f"nested exit differs: {label}")
            if complete:
                records.extend(child / name for name in FILES)
        if complete and seen != required:
            raise ValueError(f"{family} nested child multiplicity differs: {seen-required}, {required-seen}")
        if not complete:
            incomplete.extend(
                f"missing {family}:{label}:{count}"
                for label, count in sorted((required - seen).items())
            )
    return records, incomplete


def inspect(evidence: Path, seal_current: bool = False) -> int:
    root = receipt_root(evidence)
    manifest = root / MANIFEST
    children = sorted(path for path in root.iterdir() if path.name != MANIFEST)
    if len(children) != len(LABELS):
        raise ValueError("missing or extra command receipt")
    records = []
    cwd_seen = set()
    for number, (child, label) in enumerate(zip(children, LABELS), 1):
        if child.name != f"{number:04d}-{label}" or child.is_symlink() or \
                not child.is_dir():
            raise ValueError(f"receipt order or label differs: {child}")
        current = seal_current and label == "receipts-audit"
        expected_files = set(FILES) - ({"exit"} if current else set())
        if {part.name for part in child.iterdir()} != expected_files:
            raise ValueError(f"receipt files differ: {label}")
        for filename in expected_files:
            regular(child / filename)
        if (child / "label").read_bytes() != (label + "\n").encode():
            raise ValueError(f"receipt label differs: {label}")
        cwd_bytes = (child / "cwd").read_bytes()
        try:
            cwd = cwd_bytes.decode()
        except UnicodeDecodeError as exc:
            raise ValueError("invalid cwd encoding") from exc
        if not cwd.startswith("/") or not cwd.endswith("\n") or \
                "\n" in cwd[:-1] or Path(cwd[:-1]).resolve() != Path(cwd[:-1]):
            raise ValueError(f"nonphysical cwd: {label}")
        cwd_seen.add(cwd)
        argv = (child / "argv.nul").read_bytes()
        if not argv.endswith(b"\0") or b"\0\0" in argv or not argv:
            raise ValueError(f"malformed NUL argv: {label}")
        try:
            args = [piece.decode() for piece in argv[:-1].split(b"\0")]
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid argv encoding: {label}") from exc
        if Path(args[0]).name != expected_program(label):
            raise ValueError(f"wrong executable in argv: {label}")
        if not current:
            raw_exit = (child / "exit").read_bytes()
            if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", raw_exit) or \
                    int(raw_exit) > 255:
                raise ValueError(f"malformed numeric exit: {label}")
            status = int(raw_exit)
            if label == "invalid-compile":
                if not 1 <= status <= 123:
                    raise ValueError("invalid compile was not an ordinary rejection")
            elif status != 0:
                raise ValueError(f"unexpected command failure: {label}")
            aliases = ALIASES.get(label, (None, None))
            for name, stream in zip(aliases, ("stdout.raw", "stderr.raw")):
                if name is not None:
                    target = evidence / name
                    regular(target)
                    if not target.samefile(child / stream):
                        raise ValueError(f"raw alias differs: {label}/{name}")
            if label == "invalid-compile" and \
                    b"C4 P2/G2 first frame requires pending output and token forward" \
                    not in (child / "stderr.raw").read_bytes():
                raise ValueError("current invalid compiler diagnostic absent")
            if label != "receipts-audit":
                records.extend((child / filename, filename) for filename in FILES)
    if len(cwd_seen) != 1:
        raise ValueError("command cwd changed")
    nested_records, incomplete = nested_inventory(evidence, complete=True)
    if incomplete:
        raise ValueError("incomplete nested child receipts")
    records.extend((path, path.name) for path in nested_records)
    records.extend((evidence / name, name) for name in (
        "nested-path.txt", "static-nested-path.txt", "nested-profile.tsv"))
    for path, _ in records:
        regular(path)
    lines = "".join(
        f"{sha256(path.read_bytes()).hexdigest()}  {path.relative_to(evidence)}\n"
        for path, name in records
    )
    if seal_current:
        if manifest.exists():
            raise ValueError("receipt seal already exists")
        with manifest.open("x") as output:
            output.write(lines)
    else:
        regular(manifest)
        if manifest.read_text() != lines:
            raise ValueError("receipt raw or metadata changed")
    return len(children)


def inspect_prefix(evidence: Path) -> int:
    """Record a fail-closed snapshot without converting a failed gate to PASS."""
    problems = []
    observed = []
    present = {}
    try:
        root = receipt_root(evidence)
        for child in sorted(root.iterdir()):
            if child.name == MANIFEST:
                continue
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"unsafe command child: {child}")
            number = len(observed)
            expected_name = (f"{number+1:04d}-{LABELS[number]}"
                             if number < len(LABELS) else "")
            postfailure_name = f"{number+1:04d}-postfailure-audit"
            if child.name not in (expected_name, postfailure_name):
                raise ValueError(f"command prefix differs: {child.name}")
            observed.append("postfailure-audit" if child.name == postfailure_name
                            else LABELS[number])
            names = {part.name for part in child.iterdir()}
            if not names <= set(FILES):
                raise ValueError(f"unexpected command receipt file: {child.name}")
            missing = sorted(set(FILES) - names)
            if missing:
                problems.append(f"{child.name} missing {','.join(missing)}")
            for name in sorted(names):
                path = child / name
                regular(path)
                if child.name != postfailure_name:
                    present[str(path.relative_to(evidence))] = sha256(path.read_bytes()).hexdigest()
            if child.name == postfailure_name:
                problems.append("active postfailure auditor is volatile and requires outer seal")
            if "label" in names and (child / "label").read_text() != observed[-1] + "\n":
                problems.append(f"{child.name} label differs")
            if "argv.nul" in names:
                raw = (child / "argv.nul").read_bytes()
                if not raw.endswith(b"\0") or b"\0\0" in raw:
                    problems.append(f"{child.name} argv malformed")
            if "exit" in names:
                raw = (child / "exit").read_bytes()
                if not re.fullmatch(rb"(?:0|[1-9][0-9]{0,2})\n", raw) or int(raw) > 255:
                    problems.append(f"{child.name} exit malformed")
        if "postfailure-audit" in observed:
            if observed[-1] != "postfailure-audit":
                raise ValueError("postfailure audit is not final")
            completed = len(observed) - 1
        else:
            completed = len(observed)
        if completed < len(LABELS):
            problems.append(f"remaining direct commands: {len(LABELS)-completed}")
    except (ValueError, OSError) as exc:
        problems.append(str(exc))
    try:
        _, incomplete = nested_inventory(evidence, complete=False)
        problems.extend(incomplete)
        for name in ("nested-path.txt", "static-nested-path.txt", "nested-profile.tsv"):
            path = evidence / name
            if path.exists():
                regular(path)
                present[name] = sha256(path.read_bytes()).hexdigest()
        for root_name in ("nested-path.txt", "static-nested-path.txt"):
            pointer = evidence / root_name
            if not pointer.exists():
                problems.append(f"missing {root_name}")
                continue
            directory = Path(pointer.read_text().strip())
            if directory.parent != evidence or directory.is_symlink() or not directory.is_dir():
                raise ValueError(f"unsafe {root_name}")
            for child in sorted(directory.iterdir()):
                for part in sorted(child.iterdir()):
                    regular(part)
                    present[str(part.relative_to(evidence))] = sha256(part.read_bytes()).hexdigest()
    except (ValueError, OSError) as exc:
        problems.append(str(exc))
    result = {
        "status": "HOLD/INCOMPLETE", "observed_direct": observed,
        "issues": problems, "present_sha256": present,
    }
    (evidence / "postfailure-inventory.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("first-frame postfailure HOLD/INCOMPLETE")
    return len(observed)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] not in
                                     ("--seal-current", "--prefix")):
        raise SystemExit("usage: check_first_frame_command_receipts.py EVIDENCE [--seal-current|--prefix]")
    if len(sys.argv) == 3 and sys.argv[2] == "--prefix":
        inspect_prefix(Path(sys.argv[1]))
    else:
        print(f"first-frame command receipts PASS: {inspect(Path(sys.argv[1]), len(sys.argv) == 3)}")
