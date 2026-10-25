"""Validate a compiled trained-C2 terminal reference line against its input C2."""
import hashlib
import re


LINE = re.compile(
    r"G3-R TRAINED EOS REFERENCE PASS c2_sha256=([0-9a-f]{64}) "
    r"prompt=([0-9a-f]{2}) id=([0-9a-f]{2}) "
    r"terminal=1 generated=1 history=2 "
    r"rng=(-?[0-9]+),(-?[0-9]+),(-?[0-9]+),(-?[0-9]+)\n?\Z"
)


def verify(line: str, trained_c2: bytes, prompt: int) -> tuple[int, tuple[int, ...]]:
    match = LINE.fullmatch(line)
    if match is None or not 0 <= prompt <= 255:
        raise ValueError("terminal reference shape or prompt differs")
    digest, actual_prompt, generated, *words = match.groups()
    if digest != hashlib.sha256(trained_c2).hexdigest():
        raise ValueError("terminal reference used a different C2 image")
    if int(actual_prompt, 16) != prompt:
        raise ValueError("terminal reference prompt differs")
    rng = tuple(int(word) for word in words)
    if rng != (1, 1729, 1, 0):
        raise ValueError("terminal reference successor RNG differs")
    return int(generated, 16), rng
