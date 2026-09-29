import sys
from pathlib import Path


ROBODOJO_ROOT = Path(__file__).resolve().parents[1]
if str(ROBODOJO_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBODOJO_ROOT))

from scripts.run_xlens_geometry import build_parser  # noqa: E402


def test_cli_parser_accepts_manifest_checkpoint_and_output():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--manifest",
            "scene.json",
            "--ckpt",
            "xlens.pth",
            "--out-dir",
            "out",
            "--xlens-root",
            "../XLens",
        ]
    )

    assert args.manifest == "scene.json"
    assert args.ckpt == "xlens.pth"
    assert args.out_dir == "out"
    assert args.xlens_root == "../XLens"

