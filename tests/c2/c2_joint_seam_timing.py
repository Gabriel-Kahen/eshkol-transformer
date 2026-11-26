#!/usr/bin/env python3
"""Development-only C2 joint seam overlay and complete timing attribution."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "tests/c2/c2_checkpoint_operational_runtime.esk"
SPECS = (
    ("load-native-measure", "c2-load-native-measure", 7, 1),
    ("load-native-stage", "c2-load-native-stage", 16, 1),
    ("load-parse", "c2-load-parse-o2-internal", 5, 1),
    ("model", "c2-save-model-encode-internal", 4, 6),
    ("o2", "c2-save-o2-stage-internal", 1, 6),
    ("save-native-measure", "c2-save-native-measure", 18, 3),
    ("save-native-encode", "c2-save-native-encode-validate", 19, 3),
)
NATIVE = {"load-native-measure", "load-native-stage", "save-native-measure", "save-native-encode"}
RECORD = re.compile(r"^C2 SEAM TIMING: name=([a-z0-9-]+) index=([0-9]+) start-ms=([0-9]+) end-ms=([0-9]+)$")
PROGRESS = re.compile(r"^C2 OPERATIONAL PROGRESS: phase=([0-9]+) monotonic-ms=([0-9]+)$")
SHA_RECORD = re.compile(
    r"^C2 C1 SHA TIMING: stage=(tensor|container) calls=([0-9]+) "
    r"bytes=([0-9]+) assembly-ms=([0-9]+) sha-ms=([0-9]+)$")
JOINT_MODEL_TENSORS = 62  # The fixed joint fixture uses 62 C1 records per model.
# Six byte-identical model images: sum(domain + record + payload) and
# six times (container domain + unsigned C1 image), from joint-exact.c2.
JOINT_SHA_INPUT_BYTES = {"tensor": 100656114, "container": 100646766}


def c1_overlay(overlay):
    source = (ROOT / "internal/c1/lib/transformer/checkpoint_internal.esk").read_text()
    tensor = "(c1-sha256 (c1-concat (list c1-tensor-domain record payload)))"
    container = "(c1-sha256 (c1-concat\n                                         (list c1-container-domain unsigned)))"
    if source.count(tensor) != 1 or source.count(container) != 1:
        raise ValueError("C1 checksum call sites changed")
    source = source.replace("(provide c1-persistence-policy-internal",
                            "(provide c2-timing-c1-stats\n         c1-persistence-policy-internal", 1)
    anchor = "(define (c1-build-record descriptor)"
    if source.count(anchor) != 1:
        raise ValueError("C1 record definition changed")
    probe = """;;; Generated for the single-compiler C2 diagnostic; no C1 source is edited.
(extern i64 c2-timing-c1-now :real et_c2_operational_monotonic_millis_v1)
;;; Per stage: calls, input bytes, input assembly milliseconds, SHA milliseconds.
(define c2-timing-c1-stats (vector (vector 0 0 0 0) (vector 0 0 0 0)))
(define (c2-timed-c1-hash-parts parts stage)
  (let* ((started (c2-timing-c1-now))
         (input (c1-concat parts))
         (assembled (c2-timing-c1-now))
         (digest (c1-sha256 input))
         (finished (c2-timing-c1-now))
         (stats (vector-ref c2-timing-c1-stats stage)))
    (vector-set! stats 0 (+ (vector-ref stats 0) 1))
    (vector-set! stats 1 (+ (vector-ref stats 1) (bytevector-length input)))
    (vector-set! stats 2 (+ (vector-ref stats 2) (- assembled started)))
    (vector-set! stats 3 (+ (vector-ref stats 3) (- finished assembled)))
    digest))

"""
    source = source.replace(anchor, probe + anchor)
    source = source.replace(tensor,
                            "(c2-timed-c1-hash-parts (list c1-tensor-domain record payload) 0)")
    source = source.replace(container,
                            "(c2-timed-c1-hash-parts (list c1-container-domain unsigned) 1)")
    target = overlay / "transformer/checkpoint_internal.esk"
    target.parent.mkdir()
    target.write_text(source)


def wrapper(name, target, arity, count, native=False):
    args = [f"arg{i}" for i in range(arity)]
    clock = "c2-timing-load-now" if name.startswith("load-native") else (
        "c2-timing-save-now" if name.startswith("save-native") else "c2-operational-now")
    binding = "" if native else f"(set! {target} c2-timed-{name})\n"
    saved = "" if native else f"(define saved-timing-{name} {target})\n"
    called = target if native else f"saved-timing-{name}"
    return f"""(define timing-{name}-starts (make-vector {count} -1))
(define timing-{name}-ends (make-vector {count} -1))
(define timing-{name}-calls 0)
{saved}
(define (c2-timed-{name} {' '.join(args)})
    (let* ((index timing-{name}-calls)
           (started ({clock})))
      (if (< index {count}) (vector-set! timing-{name}-starts index started) #f)
      (let ((result ({called} {' '.join(args)})))
        (if (< index {count})
            (vector-set! timing-{name}-ends index ({clock})) #f)
        (set! timing-{name}-calls (+ index 1))
        result)))
{binding}
"""


def generate(path):
    source = SOURCE.read_text()
    anchor = "(define owner\n  (begin\n    (c2-operational-progress 1)"
    footer = '(display "C2 OPERATIONAL RUNTIME PASS: ")'
    if source.count(anchor) != 1 or source.count(footer) < 1:
        raise ValueError("joint fixture anchors changed")
    probes = "\n;;; Generated only for the opt-in single-compiler diagnostic.\n"
    probes += "\n".join(wrapper(*spec) for spec in SPECS if spec[0] not in NATIVE)
    source = source.replace(anchor, probes + "\n" + anchor)
    records = '\n;;; Read-only timing receipt; Python validates count and nesting.\n'
    for name, target, arity, count in SPECS:
        records += f"""(if (> timing-{name}-calls {count})
    (begin (display "C2 SEAM TIMING OVERFLOW: name={name} calls=")
           (display timing-{name}-calls) (newline)) #f)
"""
        records += f"""(let timing-output-{name} ((index 0))
  (if (= index {count}) #t
      (begin
        (display "C2 SEAM TIMING: name={name} index=") (display index)
        (display " start-ms=") (display (vector-ref timing-{name}-starts index))
        (display " end-ms=") (display (vector-ref timing-{name}-ends index))
        (newline)
        (timing-output-{name} (+ index 1)))))
"""
    for stage, index in (("tensor", 0), ("container", 1)):
        records += f"""(let ((stats (vector-ref c2-timing-c1-stats {index})))
  (display "C2 C1 SHA TIMING: stage={stage} calls=") (display (vector-ref stats 0))
  (display " bytes=") (display (vector-ref stats 1))
  (display " assembly-ms=") (display (vector-ref stats 2))
  (display " sha-ms=") (display (vector-ref stats 3)) (newline))
"""
    before, after = source.rsplit(footer, 1)
    source = before + records + "\n" + footer + after
    path.write_text(source)

    overlay = path.parent / "overlay"
    overlay.mkdir()
    c1_overlay(overlay)
    original_root = (ROOT / "native/c2_wave2_root.esk").read_text()
    for kind in ("save", "load"):
        filename = f"c2_checkpoint_{kind}_extension.esk"
        original_root = original_root.replace(
            f'(load "{filename}")', f'(load "{overlay / filename}")')
        extension = (ROOT / "native" / filename).read_text()
        selected = [spec for spec in SPECS if spec[0].startswith(kind + "-native")]
        for name, target, arity, count in selected:
            needle = f"({target}\n"
            if extension.count(needle) != 1:
                raise ValueError(f"native call anchor changed: {target}")
            extension = extension.replace(needle, f"(c2-timed-{name}\n")
        clock = f"c2-timing-{kind}-now"
        injected = f"(extern i64 {clock} :real et_c2_operational_monotonic_millis_v1)\n"
        injected += "\n".join(wrapper(*spec, native=True) for spec in selected)
        insertion = f"(define c2-{kind}-result-bytes"
        if extension.count(insertion) != 1:
            raise ValueError(f"native definition anchor changed: {kind}")
        extension = extension.replace(insertion, injected + "\n" + insertion)
        (overlay / filename).write_text(extension)
    (overlay / "c2_wave2_root.esk").write_text(original_root)


def validate(stdout, progress):
    if len(re.findall(r"^C2 OPERATIONAL RUNTIME PASS: 26 checks mode=joint\b", stdout.read_text(), re.M)) != 1:
        raise ValueError("missing exact 26-check joint PASS")
    observations = {}
    for line in stdout.read_text().splitlines():
        if line.startswith("C2 SEAM TIMING OVERFLOW:"):
            raise ValueError("seam observation overflow")
        if line.startswith("C2 SEAM TIMING:"):
            match = RECORD.fullmatch(line)
            if not match:
                raise ValueError(f"malformed seam record: {line}")
            name, index, start, end = match.groups()
            if name not in {spec[0] for spec in SPECS}:
                raise ValueError(f"unexpected seam: {name}")
            key = (name, int(index))
            if key in observations or int(start) > int(end):
                raise ValueError(f"duplicate or inverted seam: {key}")
            observations[key] = (int(start), int(end))
    expected = {(name, i) for name, _, _, count in SPECS for i in range(count)}
    if set(observations) != expected:
        raise ValueError(f"incomplete observations: missing={expected - set(observations)}, extra={set(observations) - expected}")
    sha = {}
    for line in stdout.read_text().splitlines():
        if line.startswith("C2 C1 SHA TIMING:"):
            match = SHA_RECORD.fullmatch(line)
            if not match or match[1] in sha:
                raise ValueError(f"malformed or duplicate C1 SHA record: {line}")
            sha[match[1]] = tuple(map(int, match.groups()[1:]))
    if set(sha) != {"tensor", "container"}:
        raise ValueError("incomplete C1 SHA records")
    if sha["container"][0] != 6 or sha["tensor"][0] != 6 * JOINT_MODEL_TENSORS:
        raise ValueError("C1 SHA call counts differ from six model encodes")
    if any(sha[stage][1] != expected for stage, expected in JOINT_SHA_INPUT_BYTES.items()):
        raise ValueError("C1 SHA input bytes differ from the fixed joint fixture")
    model_ms = sum(end - start for (name, _), (start, end) in observations.items()
                   if name == "model")
    assembly_ms = sum(stage[2] for stage in sha.values())
    digest_ms = sum(stage[3] for stage in sha.values())
    if assembly_ms + digest_ms > model_ms:
        raise ValueError("C1 SHA stages exceed measured model seam")
    phases = {}
    for line in progress.read_text().splitlines():
        match = PROGRESS.fullmatch(line)
        if not match or int(match[1]) in phases:
            raise ValueError("malformed or duplicate progress marker")
        phases[int(match[1])] = int(match[2])
    phase_order = [1, 2, *range(10, 17), *range(20, 27), *range(30, 37), 40, 41]
    if list(phases) != phase_order or any(a > b for a, b in zip(phases.values(), list(phases.values())[1:])):
        raise ValueError("incomplete or nonmonotonic 25-marker progress")

    def inside(key, start, end):
        measured = observations[key]
        if not (start <= measured[0] <= measured[1] <= end):
            raise ValueError(f"seam outside parent phase: {key} {measured} [{start}, {end}]")

    for name in ("load-native-measure", "load-native-stage", "load-parse"):
        inside((name, 0), phases[1], phases[2])
    if not (observations["load-native-measure", 0][1] <=
            observations["load-native-stage", 0][0] <=
            observations["load-native-stage", 0][1] <=
            observations["load-parse", 0][0]):
        raise ValueError("load seam order changed")
    for save in range(3):
        base = 10 + save * 10
        measure = (phases[base + 2], phases[base + 3])
        encode = (phases[base + 3], phases[base + 4])
        inside(("save-native-measure", save), *measure)
        inside(("save-native-encode", save), *encode)
        for offset, parent in ((0, measure), (1, encode)):
            for name in ("model", "o2"):
                inside((name, 2 * save + offset), *parent)
            model = observations["model", 2 * save + offset]
            o2 = observations["o2", 2 * save + offset]
            native = observations[("save-native-measure" if offset == 0 else
                                   "save-native-encode"), save]
            if not (model[1] <= o2[0] <= o2[1] <= native[0]):
                raise ValueError("save seam order changed")
        print(f"save-{save + 1} measure-parent-ms={measure[1]-measure[0]} encode-parent-ms={encode[1]-encode[0]} "
              f"model-measure-ms={observations['model', 2*save][1]-observations['model', 2*save][0]} "
              f"o2-measure-ms={observations['o2', 2*save][1]-observations['o2', 2*save][0]} "
              f"native-measure-ms={observations['save-native-measure', save][1]-observations['save-native-measure', save][0]} "
              f"model-encode-ms={observations['model', 2*save+1][1]-observations['model', 2*save+1][0]} "
              f"o2-encode-ms={observations['o2', 2*save+1][1]-observations['o2', 2*save+1][0]} "
              f"native-encode-ms={observations['save-native-encode', save][1]-observations['save-native-encode', save][0]}")
    print(f"C2 C1 SHA ATTRIBUTION: model-ms={model_ms} assembly-ms={assembly_ms} "
          f"sha-ms={digest_ms} other-model-ms={model_ms-assembly_ms-digest_ms} "
          f"tensor-calls={sha['tensor'][0]} tensor-bytes={sha['tensor'][1]} "
          f"container-calls={sha['container'][0]} container-bytes={sha['container'][1]}")
    print(f"C2 SEAM TIMING VALIDATED: observations={len(observations)} progress={len(phases)}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "generate":
        generate(Path(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "validate":
        validate(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("usage: c2_joint_seam_timing.py generate OUTPUT | validate STDOUT PROGRESS")


if __name__ == "__main__":
    main()
