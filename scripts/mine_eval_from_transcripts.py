"""Standalone wrapper around `code_rag.eval.mine_transcripts.mine_all`.

Usage:
    python -m scripts.mine_eval_from_transcripts \
        --output src/code_rag/eval/fixtures/mined_eval.json \
        --max-pairs 500 \
        --root C:/Users/Alex/RiderProjects \
        --root C:/Users/Alex/signals \
        --root "C:/Users/Alex/Documents/NinjaTrader 8/bin/Custom/Strategies"

For most workflows, `code-rag eval-mine` is friendlier — it pulls roots
straight from your active config.toml. This script is the no-config
fallback for CI / one-off runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

from code_rag.eval.mine_transcripts import mine_all


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "--transcripts-dir", type=Path,
        default=Path.home() / ".claude" / "projects",
        help="Root directory of Claude Code session transcripts.",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Where to write the JSON eval fixture.",
    )
    p.add_argument(
        "--root", type=Path, action="append", required=True,
        help="Indexed root (used to normalize paths). Pass once per root.",
    )
    p.add_argument(
        "--max-pairs", type=int, default=500,
        help="Cap the output size (default 500).",
    )
    p.add_argument(
        "--include-source", action="store_true",
        help="Include `_source` debug metadata on each case.",
    )
    return p.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if not args.transcripts_dir.exists():
        print(f"transcripts dir not found: {args.transcripts_dir}", file=sys.stderr)
        return 2

    pairs = mine_all(args.transcripts_dir, args.root, max_pairs=args.max_pairs)
    cases = [p.to_case() for p in pairs]
    if not args.include_source:
        for c in cases:
            c.pop("_source", None)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"mined {len(cases)} pairs from {args.transcripts_dir}", file=sys.stderr)
    print(f"wrote -> {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
