#!/usr/bin/env python3
"""Summarize a Root A feedback-loop scoreq-summary.json into gates.

Reconstructed replacement for the tool referenced by
docs/roota-language-porting-recipe.md (Stage 7). Reads the summary written by
tools/run_roota_feedback_loop.py and writes <out-dir>/gates.json plus a
human-readable table to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoreq-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--param-budget", type=int, default=1_500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary: dict[str, Any] = json.loads(args.scoreq_summary.read_text(encoding="utf-8"))
    lanes = summary.get("lanes") or {}
    parameters = dict(summary.get("parameters") or {})
    parameters["budget"] = int(args.param_budget)
    parameters["total"] = int(
        parameters.get("acoustic", 0) + parameters.get("duration", 0) + parameters.get("decoder", 0)
    )
    parameters["within_budget"] = parameters["total"] <= parameters["budget"]

    teacher = (lanes.get("teacher_piper") or {}).get("scoreq_mean")
    student_full = (lanes.get("student_full") or {}).get("scoreq_mean")
    teacher_student_dec = (lanes.get("teacher_latent_student_dec") or {}).get("scoreq_mean")
    piper_dec = (lanes.get("teacher_latent_piper_dec") or {}).get("scoreq_mean")

    def delta(value: float | None) -> float | None:
        if value is None or teacher is None:
            return None
        return round(value - teacher, 4)

    gates: dict[str, Any] = {
        "candidate_label": summary.get("candidate_label"),
        "rows_rendered": summary.get("rows_rendered"),
        "parameters": parameters,
        "gates": {
            "within_param_budget": parameters["within_budget"],
            "student_full_rendered": bool(student_full is not None),
        },
        "lane_table": {
            name: {
                "scoreq": lane.get("scoreq_mean"),
                "delta_vs_teacher": delta(lane.get("scoreq_mean")),
                "n": lane.get("n"),
            }
            for name, lane in lanes.items()
        },
    }
    if student_full is not None and teacher is not None:
        gates["gates"]["student_full_scoreq"] = student_full
        gates["gates"]["student_full_delta_vs_teacher"] = round(student_full - teacher, 4)
    if teacher_student_dec is not None and piper_dec is not None:
        gates["gates"]["decoder_compression_delta"] = round(teacher_student_dec - piper_dec, 4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gates.json").write_text(
        json.dumps(gates, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"candidate {gates['candidate_label']}  rows={gates['rows_rendered']}")
    print(
        f"params acoustic={parameters.get('acoustic')} duration={parameters.get('duration')} "
        f"decoder={parameters.get('decoder')} total={parameters['total']} "
        f"budget={parameters['budget']} within={parameters['within_budget']}"
    )
    print(f"{'lane':32s}{'SCOREQ':>8s}{'Δteacher':>10s}{'n':>5s}")
    for name, row in gates["lane_table"].items():
        scoreq = row["scoreq"]
        delta = row["delta_vs_teacher"]
        print(
            f"{name:32s}"
            f"{(f'{scoreq:.4f}' if scoreq is not None else '—'):>8s}"
            f"{(f'{delta:+.4f}' if delta is not None else '—'):>10s}"
            f"{(row['n'] or 0):>5d}"
        )
    print(f"gates written to {args.out_dir / 'gates.json'}")


if __name__ == "__main__":
    main()
