from pathlib import Path
import re


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_policy.sh"


def test_eval_policy_consumes_xpolicylab_bench_name_argument():
    source = SCRIPT.read_text(encoding="utf-8")
    handler = re.search(r"(?ms)^\s*--bench_name\)\s*(.*?)(?=^\s*;;)", source)

    assert handler is not None, "eval_policy.sh must handle XPolicyLab's --bench_name option"
    assert 'bench_name="$2"' in handler.group(1)
    assert re.search(r"\bshift\s+2\b", handler.group(1))
    assert 'extra_args+=("$1")' not in handler.group(1)
