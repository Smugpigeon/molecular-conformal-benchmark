"""Aggregate val_metrics.json across runs/ into mean±std tables.

Per CLAUDE.md §8.2: report multi-seed mean ± std. Group by (dataset, model).

Usage:
    python scripts/aggregate_runs.py                    # all runs
    python scripts/aggregate_runs.py --model rf         # filter
    python scripts/aggregate_runs.py --out results/baseline.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import pandas as pd
import yaml


def collect_runs(runs_dir: Path) -> list[dict]:
    """Walk runs/, load (config, val_metrics) for each."""
    records = []
    for cfg_path in runs_dir.rglob("config.yaml"):
        run_dir = cfg_path.parent
        metrics_path = run_dir / "val_metrics.json"
        if not metrics_path.exists():
            continue
        try:
            cfg = yaml.safe_load(cfg_path.read_text())
            metrics = json.loads(metrics_path.read_text())
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            print(f"WARN: skipping {run_dir}: {e}", file=sys.stderr)
            continue

        rec = {
            "run_dir": str(run_dir.relative_to(runs_dir.parent)),
            "dataset": cfg.get("dataset", {}).get("name", "?"),
            "model": cfg.get("model", {}).get("name", "?"),
            "seed": cfg.get("seed", -1),
            **metrics,
        }
        records.append(rec)
    return records


def aggregate(
    records: list[dict],
    model_filter: str | None = None,
    dataset_filter: str | None = None,
) -> pd.DataFrame:
    """Group by (dataset, model), compute mean ± std across seeds."""
    if model_filter:
        records = [r for r in records if r["model"] == model_filter]
    if dataset_filter:
        records = [r for r in records if r["dataset"] == dataset_filter]

    if not records:
        return pd.DataFrame()

    metric_keys = sorted(
        {k for r in records for k in r if k not in {"run_dir", "dataset", "model", "seed"}}
    )

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        key = (r["dataset"], r["model"])
        for m in metric_keys:
            if m in r and isinstance(r[m], (int, float)):
                grouped[key][m].append(float(r[m]))

    rows = []
    for (dataset, model), metric_vals in sorted(grouped.items()):
        row = {"dataset": dataset, "model": model, "n_seeds": 0}
        for m in metric_keys:
            vals = metric_vals.get(m, [])
            if not vals:
                row[m] = "—"
                continue
            row["n_seeds"] = max(row["n_seeds"], len(vals))
            if len(vals) == 1:
                row[m] = f"{vals[0]:.3f}"
            else:
                row[m] = f"{mean(vals):.3f} ± {stdev(vals):.3f}"
        rows.append(row)

    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, title: str = "Baseline Results") -> str:
    if df.empty:
        return f"# {title}\n\n_No runs found._\n"
    return f"# {title}\n\n{df.to_markdown(index=False)}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--model", type=str, default=None, help="Filter by model name")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset")
    parser.add_argument("--out", type=Path, default=None, help="Write Markdown here")
    parser.add_argument(
        "--csv", type=Path, default=None, help="Also write raw per-run CSV here"
    )
    args = parser.parse_args()

    if not args.runs_dir.exists():
        print(f"ERROR: {args.runs_dir} does not exist", file=sys.stderr)
        return 1

    records = collect_runs(args.runs_dir)
    print(f"Collected {len(records)} runs", file=sys.stderr)
    if not records:
        return 1

    df = aggregate(records, model_filter=args.model, dataset_filter=args.dataset)
    md = to_markdown(df, title="Baseline Results (val split)")
    print(md)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"Wrote {args.out}", file=sys.stderr)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
