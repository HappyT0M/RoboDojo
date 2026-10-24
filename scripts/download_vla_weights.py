#!/usr/bin/env python3
"""Batch-download RoboDojo evaluation checkpoints using its existing downloader.

Edit DEFAULT_ROOT / DEFAULT_MODELS or pass --root / --models.
Requires Bash 4+, git, git-lfs and installed XPolicyLab policy adapters.
No GPU, huggingface_hub, or gcloud needed.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

DEFAULT_ROOT = "/data/xiebingsheng/ckpt"
DEFAULT_MODELS = ["DreamZero", "Pi_0", "LingBot_VLA"]
DOWNLOADER = Path(__file__).resolve().parent / "RoboDojo" / "download_ckpt.sh"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', default=os.environ.get('VLA_CKPT_ROOT', DEFAULT_ROOT))
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS,
                        help='Policy names/aliases accepted by the RoboDojo downloader')
    parser.add_argument('--source', choices=['huggingface', 'modelscope'], default='huggingface')
    parser.add_argument('--dry-run', action='store_true', help='Preview without network or writes')
    args = parser.parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    env = dict(os.environ)
    env['ROBO_DOJO_CKPT_OUTPUT_ROOT'] = str(root)
    env['ROBO_DOJO_CKPT_CACHE'] = str(root / '.cache' / ('robodojo_ckpt_' + args.source + '_repo'))
    failures = []
    for name in dict.fromkeys(args.models):
        print(f'[RoboDojo] {args.source}: {name} -> {root}/<resolved-policy-name>', flush=True)
        if args.dry_run:
            continue
        try:
            subprocess.run(['bash', str(DOWNLOADER), args.source, name], env=env, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures.append(name)
            print(f'[FAILED] {name}: {exc}', file=sys.stderr, flush=True)
    if failures:
        print('Failed models: ' + ', '.join(failures) +
              '. Check the downloader messages above; no base-weight fallback was used.',
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
