#!/usr/bin/env python3
"""Build-only, fixed-byte D2 source adaptation for the private E3 prerequisite."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

MANIFEST = "native/e3_d2_source_variant.json"
MANIFEST_SHA256 = "9349a0d6c0f3b108975789f022c9aefc800722dc85d595b8034f5610c71a319d"
SOURCE = "internal/d2/lib/d2_dataset.esk"
OUTPUT = "source/e3_d2_dataset.esk"
PROVENANCE = "source/e3_d2_source_provenance.json"
GENERATOR = "scripts/generate-e3-d2-source.py"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute(path):
    path = Path(path)
    require(".." not in path.parts, "parent traversal is not permitted")
    return path.absolute()


@contextmanager
def directory(path, create=False):
    """Walk no-follow directory descriptors, retaining the actual admitted directory."""
    path = absolute(path)
    descriptor = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=descriptor)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)


def read_regular(path):
    with directory(path.parent) as parent:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent)
        with os.fdopen(descriptor, "rb") as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
                    "input must be a regular file")
            data = stream.read(1024 * 1024 + 1)
            require(len(data) <= 1024 * 1024, "input exceeds fixed source bound")
            return data


def write_regular(parent, name, data):
    try:
        mode = os.stat(name, dir_fd=parent, follow_symlinks=False).st_mode
        require(stat.S_ISREG(mode), "output must be a regular file, not a symlink")
    except FileNotFoundError:
        pass
    temporary = ".e3-d2-" + uuid.uuid4().hex
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o644, dir_fd=parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
    finally:
        try:
            os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError:
            pass


def generate(output_dir):
    # Do not resolve symlinks: every input component is admitted with O_NOFOLLOW.
    script = absolute(__file__)
    root = script.parent.parent
    generator = read_regular(script)
    manifest_bytes = read_regular(root / MANIFEST)
    require(digest(manifest_bytes) == MANIFEST_SHA256, "manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    require(manifest["source"] == SOURCE and manifest["output"] == OUTPUT,
            "fixed source/output paths changed")
    source = read_regular(root / SOURCE)
    expected = manifest["expected_form"].encode("utf-8")
    replacement = manifest["replacement_form"].encode("utf-8")
    require(digest(expected) == manifest["expected_form_sha256"], "expected form hash mismatch")
    require(digest(replacement) == manifest["replacement_form_sha256"], "replacement hash mismatch")
    require(source.count(expected) == 1, "expected exactly one predecessor form")
    require(source.count(b"(define (d2-tokenizer-identity ") == 1,
            "expected exactly one tokenizer identity definition")
    require(digest(source) == manifest["source_sha256"] and
            len(source) == manifest["source_bytes"], "predecessor hash/size mismatch")
    prefix, suffix = source.split(expected)
    generated = prefix + replacement + suffix
    require(generated.count(replacement) == 1 and
            generated.count(b"(define (d2-tokenizer-identity ") == 1,
            "expected exactly one replacement definition")
    require(b"t2-private-tokenizer-" not in generated, "T2 identity reference remains")
    require(b"\r" not in generated, "generated source must use LF bytes")
    for name, data in (("prefix", prefix), ("suffix", suffix), ("generated", generated)):
        require(len(data) == manifest[name + "_bytes"] and
                digest(data) == manifest[name + "_sha256"], name + " hash/size mismatch")
    require(generated[:len(prefix)] == source[:len(prefix)] and
            generated[len(prefix) + len(replacement):] == source[len(prefix) + len(expected):],
            "prefix/suffix preservation failed")
    report = {key: value for key, value in manifest.items()
              if key not in ("expected_form", "replacement_form")}
    report.update(format="eshkol-e3-d2-source-provenance", generator=GENERATOR,
                  generator_sha256=digest(generator), manifest=MANIFEST,
                  manifest_sha256=digest(manifest_bytes))
    evidence = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    # The logical leaves are fixed, and the no-follow directory descriptor prevents
    # an output symlink from redirecting creation outside the explicit destination.
    with directory(absolute(output_dir) / "source", create=True) as destination:
        for name in (Path(OUTPUT).name, Path(PROVENANCE).name):
            try:
                mode = os.stat(name, dir_fd=destination, follow_symlinks=False).st_mode
                require(stat.S_ISREG(mode), "output must be a regular file, not a symlink")
            except FileNotFoundError:
                pass
        write_regular(destination, Path(OUTPUT).name, generated)
        write_regular(destination, Path(PROVENANCE).name, evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True, help="explicit private artifact directory")
    arguments = parser.parse_args()
    try:
        evidence = generate(arguments.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "E3-D2 source generation rejected: " + str(error) + "\n")
    sys.stdout.buffer.write(evidence)


if __name__ == "__main__":
    main()
