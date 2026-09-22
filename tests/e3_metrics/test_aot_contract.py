"""Private compiled reachability closure; never a production transport API."""
from pathlib import Path
import re
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
directory = Path(sys.argv[1]).resolve()
depfile = (directory / "aot_private.d").read_text().replace("\\\n", "")
target, sources = depfile.split(":", 1)
assert Path(target).resolve() == directory / "aot_private.o"
assert [Path(p).resolve() for p in shlex.split(sources)] == [root / "tests/e3_metrics/aot_private.esk"]
undefined = subprocess.check_output(["nm", "-u", "--format=posix", str(directory / "aot_private.o")], text=True)
assert "et_e3_metrics_test_aot_bridge_v1" in undefined
for filename in ("aot-private-1", "aot-private-2"):
    symbols = subprocess.check_output(["nm", "--format=posix", str(directory / filename)], text=True)
    assert not re.search(r"\b(Py[A-Z_][A-Za-z0-9_]*|_?PyInit_\w*|THP\w*|torch\w*)\b", symbols)
    for required in ("et_e3_metrics_kernel_provider_v1", "et_l3s_kernel_provider_v1",
                     "et_l2_indexed_cross_entropy_provider_v1", "et_f32_tensor_create_v1",
                     "et_i64_tensor_create_v1", "et_e3_metrics_test_aot_bridge_v1"):
        assert re.search(rf"^{required} [Tt] ", symbols, re.M), required
print("E3-METRICS private AOT package PASS: exact Eshkol depfile and real native closure without Python")
