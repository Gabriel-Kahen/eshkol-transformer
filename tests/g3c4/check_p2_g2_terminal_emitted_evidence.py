#!/usr/bin/env python3
"""Mechanical terminal compiler evidence closure; never certifies the zero tail."""

from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys

from prepare_terminal_asan_ir import scan_pair

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    "terminal", "feature-off", "parity-original", "parity-overlay",
    "unmarked", "decode-first", "decode-second", "precommit", "postzero",
    "lease-i1", "lease-a2",
)
NATIVE = (
    "composer_native", "kernel_fail_allocator", "i2_native_helpers",
    "call_entry_allocation_shim", "p1_identity", "data_io", "checkpoint_io",
    "n3k_primitives_provider", "g3n_primitives_provider",
    "n2_primitives_provider", "g3c4_primitives_provider",
    "g3s_sampling_provider", "a2_attention_provider", "a2_kv_cache",
    "t1_i64_shell", "m3_i64_integration", "m3_model",
)
SYSTEM = frozenset("""
__errno_location __stack_chk_fail __tls_get_addr _exit abort bcmp calloc
close cosf erff exit expf fclose fegetenv fegetround fesetenv fgetc fmod
fork fprintf fputc fread free fseek fstat fsync ftell getpid getrandom
malloc memchr memcpy memmove memset open openat printf putc putchar puts
qsort read realloc renameat setjmp sinf snprintf sqrtf stderr stdout
strcmp strlen strncmp strrchr syscall unlink unlinkat waitpid write
""".split())
WRAPS = frozenset("""
arena_allocate_vector_with_header arena_allocate_with_header
arena_allocate_cons_with_header malloc eshkol_push_exception_handler
et_kernel_runtime_dispatch et_i64_tensor_borrow_view_v1
et_g3c4_private_p2g2_first_frame_carrier_v1
et_g3c4_private_p2g2_eos_first_frame_carrier_v1
et_g3c4_private_p2g2_second_frame_carrier_v1
et_g3c4_private_p2g2_prefix_commit_v1
et_g3c4_private_p2g2_terminal_commit_v1
et_g3c4_private_p2g2_terminal_snapshot_v1
et_g3c4_private_p2g2_terminal_ids_clone_copy_v1
et_g3c4_private_call_abort_v1
et_a2_kv_cache_transaction_abort_v1
et_a2_kv_cache_transaction_commit_v1
""".split())
TERMINAL_FFI = frozenset("""
et_g3c4_private_p2g2_eos_first_frame_carrier_v1
et_g3c4_private_p2g2_second_frame_carrier_v1
et_g3c4_private_p2g2_terminal_commit_v1
et_g3c4_private_p2g2_terminal_snapshot_v1
et_g3c4_private_p2g2_terminal_ids_clone_v1
et_g3c4_private_p2g2_terminal_ids_clone_copy_v1
et_g3c4_private_p2g2_terminal_ids_clone_release_v1
""".split())
ASAN = re.compile(r"^__(?:asan|ubsan|lsan|sanitizer)_")
ASAN_REPORT = re.compile(r"^__asan_report_(?:load|store)\d+$")
MANIFEST = ROOT / "tests/g3c4/p2_g2_terminal_emitted_dependencies.json"
DYNAMIC = ROOT / "tests/g3c4/p2_g2_terminal_historical_dynamic_imports.txt"


def digest(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def regular(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size == 0:
        raise ValueError(f"missing, empty or unsafe artifact: {path}")


def directory(path: Path) -> None:
    if not stat.S_ISDIR(path.lstat().st_mode):
        raise ValueError(f"unsafe directory: {path}")


def exact_files(path: Path, expected: set[str]) -> None:
    directory(path)
    actual = {child.name for child in path.iterdir()}
    if actual != expected:
        raise ValueError(f"{path}: artifact set differs: {sorted(actual ^ expected)}")
    for child in path.iterdir():
        regular(child)


def dep_words(path: Path) -> tuple[str, list[str]]:
    """One complete Make rule; reject variables, comments and duplicate paths."""
    regular(path)
    data = path.read_text()
    if not data.endswith("\n") or "\r" in data:
        raise ValueError("unterminated or CR dependency rule")
    words = []
    current = []
    separators = 0
    index = 0
    while index < len(data):
        char = data[index]
        if char == "\\":
            if index + 1 == len(data):
                raise ValueError("dangling Make escape")
            following = data[index + 1]
            if following == "\n":
                if current or index + 2 == len(data) or data[index + 2] not in " \t":
                    raise ValueError("continuation inside dependency token")
                index += 2
                continue
            if following not in (" ", "#", ":", "\\"):
                raise ValueError("unsupported Make escape")
            current.append(following)
            index += 2
            continue
        if char in "$#":
            raise ValueError("Make variable or comment in dependency")
        if char == ":":
            if separators or not current:
                raise ValueError("duplicate/missing dependency target")
            words.append("".join(current))
            current.clear()
            separators += 1
        elif char in " \t\n":
            if current:
                words.append("".join(current))
                current.clear()
            if char == "\n" and index != len(data) - 1:
                raise ValueError("multiple dependency rules")
        else:
            current.append(char)
        index += 1
    if current or separators != 1 or len(words) < 2 or len(words) != len(set(words)):
        raise ValueError("malformed or duplicate dependency")
    return words[0], words[1:]


def expected_dependencies(mode_dir: Path, family: str, name: str) -> set[Path]:
    manifest = json.loads(MANIFEST.read_text())
    if manifest["schema"] != 1 or set(manifest["native"]) != set(NATIVE) or \
            set(manifest["eshkol"]) != set(FIXTURES):
        raise ValueError("reviewed dependency inventory shape changed")
    compiler = Path(os.environ.get("ESHKOL_SOURCE_DIR", ROOT / ".deps/eshkol-src"))
    result = set()
    for token in manifest[family][name]:
        if token.startswith("$OVERLAY/"):
            path = mode_dir / "overlays" / token.removeprefix("$OVERLAY/")
        elif token.startswith("$ESHKOL_SRC/"):
            path = compiler / token.removeprefix("$ESHKOL_SRC/")
        elif "$" not in token:
            path = ROOT / token
        else:
            raise ValueError("unknown dependency token")
        if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
            raise ValueError(f"missing or unsafe dependency: {path}")
        result.add(path.resolve(strict=True))
    if len(result) != len(manifest[family][name]):
        raise ValueError("duplicate canonical dependency")
    return result


def check_depfile(path: Path, target: Path, expected: set[Path]) -> None:
    actual_target, inputs = dep_words(path)
    if actual_target != str(target) or not all(Path(item).is_absolute() for item in inputs):
        raise ValueError(f"{path}: wrong target or relative input")
    if any(part.is_symlink() for item in inputs
           for part in (Path(item), *Path(item).parents)):
        raise ValueError(f"{path}: symlink dependency")
    resolved = {Path(item).resolve(strict=True) for item in inputs}
    if len(resolved) != len(inputs) or resolved != expected:
        raise ValueError(f"{path}: source dependency closure differs")


def nm(path: Path, *options: str) -> set[str]:
    regular(path)
    output = subprocess.check_output(["nm", *options, "-P", str(path)], text=True)
    result = set()
    for line in output.splitlines():
        if not line or line.endswith("]:"):
            continue
        words = line.split()
        if len(words) < 2 or not re.fullmatch(r"[A-Za-z_.$?][A-Za-z0-9_.$?@]*", words[0]):
            raise ValueError(f"unsupported nm line: {line}")
        result.add(words[0])
    return result


def imported(path: Path, dynamic: bool = False) -> set[str]:
    return nm(path, "-D", "-u") if dynamic else nm(path, "-u")


def providers(archive: Path, runtime_build: Path) -> set[str]:
    result = set()
    for path in (archive, runtime_build / "libeshkol-runtime.a",
                 runtime_build / "libeshkol-static.a"):
        result |= nm(path, "-g", "--defined-only")
    return result


def check_imports(symbols: set[str], defined: set[str]) -> None:
    unknown = symbols - defined - SYSTEM
    unknown = {name for name in unknown
               if not (name.startswith("__real_") and name[7:] in WRAPS)}
    if unknown:
        raise ValueError(f"unreviewed raw object imports: {sorted(unknown)}")


def check_terminal_ffi(symbols: set[str]) -> None:
    observed = symbols & TERMINAL_FFI
    if observed != TERMINAL_FFI:
        raise ValueError(f"terminal FFI import set differs: {sorted(observed)}")
    prefix = "et_g3c4_private_p2g2_terminal_"
    if {name for name in symbols if name.startswith(prefix)} - TERMINAL_FFI:
        raise ValueError("unreviewed terminal FFI import")


def check_dynamic(symbols: set[str], mode: str) -> None:
    ceiling = {line for line in DYNAMIC.read_text().splitlines()
               if line and not line.startswith("#")}
    if len(ceiling) != 341:
        raise ValueError("historical dynamic import ceiling changed")
    unknown = symbols - ceiling
    if mode == "sanitize":
        unknown = {name for name in unknown if not ASAN.match(name)}
    if unknown:
        raise ValueError(f"unreviewed linked dynamic imports: {sorted(unknown)}")


def check_source_record(evidence: Path, label: str, expected: Path) -> None:
    records = []
    for suffix in ("before", "after"):
        path = evidence / f"{label}-source.{suffix}"
        regular(path)
        records.append(path.read_text())
    if records[0] != records[1] or records[0] != f"{digest(expected)}  {expected}\n":
        raise ValueError(f"{label}: source identity drift")


def verify(mode_dir: Path, evidence: Path, mode: str) -> None:
    if mode not in ("normal", "sanitize"):
        raise ValueError("unsupported compiler mode")
    directory(mode_dir)
    directory(evidence)
    composer = (ROOT / "native/g3c4_p2g2_terminal_composer_extension.esk").read_text()
    declared = set(re.findall(r":real (et_g3c4_private_p2g2_[A-Za-z0-9_]+)\)", composer))
    if declared != TERMINAL_FFI:
        raise ValueError("terminal FFI source declaration differs")
    expected_generated = {f"{name}.{suffix}" for name in FIXTURES
                          for suffix in ("raw.o", "raw.o.bc", "raw.o.ll", "d")}
    if mode == "sanitize":
        expected_generated |= {f"{name}.{suffix}" for name in FIXTURES
                               for suffix in ("asan.ll", "asan.json", "asan.o")}
    exact_files(mode_dir / "generated", expected_generated)
    exact_files(mode_dir / "objects", {f"{name}.{suffix}" for name in NATIVE
                                      for suffix in ("o", "d")})
    archive = mode_dir / "libg3c4_p2g2_terminal_composer.a"
    regular(archive)
    members = subprocess.check_output(["ar", "t", str(archive)], text=True).splitlines()
    if len(members) != len(NATIVE) or set(members) != {name + ".o" for name in NATIVE}:
        raise ValueError("native archive member set differs")
    build = Path(os.environ.get("ESHKOL_BUILD_DIR", ROOT / ".deps/eshkol-build"))
    defined = providers(archive, build)
    wrapper = (mode_dir / "cxx-wrap").read_text()
    if set(re.findall(r"-Wl,--wrap=([A-Za-z0-9_]+)", wrapper)) != WRAPS:
        raise ValueError("linker wrap boundary differs")
    inventory = {}
    for name in NATIVE:
        obj = mode_dir / "objects" / f"{name}.o"
        expected = expected_dependencies(mode_dir, "native", name)
        check_depfile(mode_dir / "objects" / f"{name}.d", obj, expected)
        check_imports(imported(obj), defined)
        inventory[str(obj.relative_to(mode_dir))] = sorted(imported(obj))
    for name in FIXTURES:
        source = (ROOT / ("tests/g3c4/p2_g2_terminal_composer_test.esk"
                         if name == "terminal" else
                         "tests/g3c4/p2_g2_terminal_composer_feature_off_test.esk")
                  if name in ("terminal", "feature-off") else
                  mode_dir / "overlays" / f"{name}.esk")
        label = mode if name == "terminal" else f"{mode}-{name}"
        check_source_record(evidence, label, source)
        raw = mode_dir / "generated" / f"{name}.raw.o"
        check_depfile(mode_dir / "generated" / f"{name}.d", raw,
                      expected_dependencies(mode_dir, "eshkol", name))
        imports = imported(raw)
        check_imports(imports, defined)
        check_terminal_ffi(imports)
        inventory[str(raw.relative_to(mode_dir))] = sorted(imports)
        if mode == "sanitize":
            derived = mode_dir / "generated" / f"{name}.asan.ll"
            actual = scan_pair(mode_dir / "generated" / f"{name}.raw.o.ll", derived)
            receipt = json.loads((mode_dir / "generated" / f"{name}.asan.json").read_text())
            if actual != receipt:
                raise ValueError(f"{name}: ASan IR receipt differs")
            asan_imports = imported(mode_dir / "generated" / f"{name}.asan.o")
            if not any(ASAN_REPORT.fullmatch(symbol) for symbol in asan_imports):
                raise ValueError(f"{name}: compiled ASan object lacks reports")
            check_imports({symbol for symbol in asan_imports if not ASAN.match(symbol)},
                          defined)
            check_terminal_ffi(asan_imports)
            inventory[f"generated/{name}.asan.o"] = sorted(asan_imports)
        linked = mode_dir / f"composer-{name}"
        regular(linked)
        dynamic = imported(linked, dynamic=True)
        check_dynamic(dynamic, mode)
        inventory[linked.name] = sorted(dynamic)
    output = evidence / f"{mode}-emitted-imports.json"
    if output.exists():
        raise ValueError(f"emitted import inventory already exists: {output}")
    output.write_text(json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n")
    hashes = evidence / f"{mode}-emitted-artifacts.sha256"
    if hashes.exists():
        raise ValueError(f"emitted hash inventory already exists: {hashes}")
    artifacts = sorted(
        [path for folder in (mode_dir / "generated", mode_dir / "objects")
         for path in folder.iterdir()] +
        [archive] + [mode_dir / f"composer-{name}" for name in FIXTURES])
    hashes.write_text("".join(f"{digest(path)}  {path}\n" for path in artifacts))
    print(f"terminal emitted artifact/import closure PASS: mode={mode} "
          f"fixtures={len(FIXTURES)} native={len(NATIVE)}; zero-tail proof HOLD")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: check_p2_g2_terminal_emitted_evidence.py "
                         "MODE_DIR EVIDENCE MODE")
    verify(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
