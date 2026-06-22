"""Deduplicate run directories that share the same (dataset, model, seed).

Per CLAUDE.md §16.1: duplicate runs with identical seeds are NOT independent
measurements — keeping them falsely tightens the reported std. This tool keeps
the most recent run per (dataset, model, seed) and removes older duplicates.

Run dir naming: <YYYY-MM-DD_HH-MM-SS>_<dataset>_<model>_seed<N>

Usage:
    python scripts/dedup_runs.py                 # dry-run (lists, deletes nothing)
    python scripts/dedup_runs.py --delete        # actually remove duplicates
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# <ts>_<dataset>_<model>_seed<N> where dataset/model may contain underscores.
# We anchor on the trailing _seed<N> and the leading timestamp.
RUN_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_(?P<rest>.+)_seed(?P<seed>\d+)$"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--delete", action="store_true", help="actually delete (default: dry-run)")
    args = parser.parse_args()

    groups: dict[tuple[str, str], list[tuple[str, Path]]] = defaultdict(list)
    for d in args.runs_dir.iterdir():
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        key = (m.group("rest"), m.group("seed"))  # (dataset_model, seed)
        groups[key].append((m.group("ts"), d))

    def has_metrics(d: Path) -> bool:
        return (d / "val_metrics.json").exists()

    n_dup = 0
    n_kept = 0
    n_orphan = 0
    for key, runs in sorted(groups.items()):
        runs.sort(key=lambda x: x[0])  # by timestamp ascending
        # Prefer the latest run that actually has val_metrics.json (a completed run).
        # CLAUDE.md §16.1 + safety: never drop a good run to keep a crashed newer one.
        completed = [r for r in runs if has_metrics(r[1])]
        if completed:
            keep = completed[-1]
        else:
            keep = runs[-1]
            n_orphan += 1
            print(f"[{key[0]} seed={key[1]}] WARNING: no completed run, keeping {keep[1].name}")
        drop = [r for r in runs if r[1] != keep[1]]
        n_kept += 1
        if drop:
            print(f"[{key[0]} seed={key[1]}] keep {keep[1].name}")
            for ts, d in drop:
                print(f"    {'DELETE' if args.delete else 'would delete'}: {d.name}")
                n_dup += 1
                if args.delete:
                    shutil.rmtree(d)

    print(f"\n{n_kept} unique (dataset,model,seed) groups; "
          f"{n_dup} duplicate runs {'deleted' if args.delete else 'to delete'}.")
    if n_dup and not args.delete:
        print("Re-run with --delete to remove them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
