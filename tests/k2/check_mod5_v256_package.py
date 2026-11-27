"""Check distinct default, native-profile and compiler-discovered K2 boundaries."""

import hashlib
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "native" / "k2_mod5_v256"
DEFAULT = ROOT / "native" / "k2_wave2"
CANONICAL = {
    "global-defined.txt": "defined_symbols",
    "undefined.txt": "undefined_symbols",
    "package-exports.txt": "public_exports",
    "public-strings.txt": "public_strings",
    "source-closure.txt": "source_closure",
    "native-source-closure.txt": "native_source_closure",
}
PROFILE_SYMBOL = re.compile(r"^(?:k2-(?:mod5-)?v256-|et_(?:k2|i2)_private_.*v256)")
PROFILE_STRING = re.compile(
    r"^(?:transformer\.capability-(?:entry|report|request):mod5-v256"
    r"|k2-mod5-v256-query-v1|mod5-width16-v256"
    r"|I2:mod5-v256-exact-f32-storage\.copy-v1)$"
)


def manifest(prefix, stem):
    return prefix.with_name(f"{prefix.name}_{stem}.txt")


def depfile_paths(path):
    _, marker, dependencies = path.read_text().replace("\\\n", " ").partition(": ")
    if not marker or not dependencies.strip():
        raise ValueError(f"invalid compiler dependency record: {path}")
    return [
        (ROOT / name).resolve(strict=True)
        for name in shlex.split(dependencies)
    ]


def check_closure(actual, expected):
    wanted = set(expected.read_text().splitlines())
    if actual != wanted:
        raise ValueError(
            f"{expected.name}: missing {sorted(wanted - actual)}; "
            f"unexpected {sorted(actual - wanted)}"
        )


def write_sources(output, paths):
    output.with_suffix(".txt").write_text("".join(f"{name}\n" for name in paths))
    output.with_suffix(".sha256").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {name}\n"
        for name, path in paths.items()
    ))


def native(evidence):
    for profile in ("default", "v256"):
        directory = evidence / profile
        paths = set()
        for name in ("k2_capabilities", "f32_tensor", "kernel_abi"):
            paths.update(depfile_paths(directory / f"{name}.d"))
        normalized = {path.relative_to(ROOT).as_posix(): path for path in sorted(paths)}
        check_closure(set(normalized), manifest(PROFILE, "native_closure"))
        write_sources(directory / "native-source-closure", normalized)


def canonical(evidence):
    for observed, expected in CANONICAL.items():
        data = (evidence / observed).read_bytes()
        if data != manifest(DEFAULT, expected).read_bytes():
            raise ValueError(f"canonical default K2 inventory drifted: {observed}")
        print(f"{hashlib.sha256(data).hexdigest()}  {observed}")


def witness_boundary(binary):
    for mode in (("-g", "--defined-only"), ("-D", "--defined-only")):
        symbols = subprocess.check_output(["nm", *mode, "--format=posix", binary], text=True)
        profile_defined = {line.split()[0] for line in symbols.splitlines()
                           if PROFILE_SYMBOL.match(line)}
        # The development executable retains Eshkol entry names in its
        # dynamic table. The separate native/package audit forbids C exports.
        check_closure(profile_defined, manifest(PROFILE, "scheme_defined_symbols"))
    undefined = subprocess.check_output(["nm", "-u", "--format=posix", binary], text=True)
    if any(PROFILE_SYMBOL.match(line) for line in undefined.splitlines()):
        raise ValueError("opt-in AOT witness has an undefined profile symbol")
    strings = subprocess.check_output(["strings", "-a", binary], text=True)
    profile_strings = {line for line in strings.splitlines() if PROFILE_STRING.fullmatch(line)}
    check_closure(profile_strings, manifest(PROFILE, "scheme_interface_strings"))


def scheme(evidence):
    paths = depfile_paths(evidence / "witness.d")
    normalized = {path.relative_to(ROOT).as_posix(): path for path in paths}
    if len(normalized) != len(paths):
        raise ValueError("duplicate path in the opt-in compiler dependency record")
    check_closure(set(normalized), manifest(PROFILE, "scheme_expected_closure"))
    write_sources(evidence / "scheme-source-closure", normalized)
    witness_boundary(evidence / "witness")


def main(argv):
    if len(argv) == 3 and argv[1] == "native":
        native(Path(argv[2]))
    elif len(argv) == 3 and argv[1] == "default":
        canonical(Path(argv[2]))
    elif len(argv) == 3 and argv[1] == "scheme":
        scheme(Path(argv[2]))
    elif len(argv) == 3 and argv[1] == "witness":
        witness_boundary(Path(argv[2]))
    else:
        raise SystemExit("usage: check_mod5_v256_package {native|default|scheme|witness} PATH")


if __name__ == "__main__":
    main(sys.argv)
