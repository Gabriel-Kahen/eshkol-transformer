#!/usr/bin/env python3
"""Run the native private SHA bridge on a valid bounded C1 model image."""

import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

from test_checkpoint_format import make_c1


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_c2_model_sha.py NATIVE_UNIT")
    model, _ = make_c1()
    assert len(model) < 1056
    with tempfile.TemporaryDirectory(prefix="c2-model-sha-") as temp:
        fixture = Path(temp) / "model.c1"
        fixture.write_bytes(model)
        subprocess.run(
            [sys.argv[1], str(fixture), hashlib.sha256(model).hexdigest()],
            check=True,
        )


if __name__ == "__main__":
    main()
