"""Evaluate DataCrossBench predictions with four weighted metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
CLASSIFICATION_FILE = SCRIPT_DIR / "flag_classification.json"

sys.path.insert(0, str(SCRIPT_DIR))
from metrics_custom import FourDimScores, compute_four_dim_score  # noqa: E402


def load_flag_classification(path: Path) -> Dict[str, str]:
    """Load the easy/hard category for each flag."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("flags", {})


def _normalise_records(data: Any, source: str) -> Dict[str, dict]:
    """Normalize supported prediction JSON shapes to a flag mapping."""
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, dict):
        records = []
        for flag_id, item in data.items():
            if isinstance(item, list):
                item = {"pred_insights": item}
            if not isinstance(item, dict):
                raise ValueError(f"{source}: {flag_id} must be a list or object")
            record = dict(item)
            record.setdefault("flag_id", flag_id)
            records.append(record)
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(f"{source}: expected an object, list, or predictions wrapper")

    normalized: Dict[str, dict] = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source}: record {index + 1} is not an object")
        flag_id = record.get("flag_id")
        if not isinstance(flag_id, str) or not flag_id:
            raise ValueError(f"{source}: record {index + 1} has no valid flag_id")
        if flag_id in normalized:
            raise ValueError(f"{source}: duplicate flag_id: {flag_id}")
        normalized[flag_id] = record
    return normalized


def load_predictions(path: str) -> Dict[str, dict]:
    """Load participant predictions from JSON."""
    prediction_path = Path(path).expanduser()
    with prediction_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    predictions = _normalise_records(data, str(prediction_path))
    for flag_id, record in predictions.items():
        insights = record.get("pred_insights")
        if not isinstance(insights, list) or not all(isinstance(item, str) for item in insights):
            raise ValueError(f"{prediction_path}: {flag_id}.pred_insights must be a string list")
        record["pred_insights"] = insights
    return predictions


def _flag_sort_key(flag_id: str):
    """Sort standard flag-N IDs numerically and other IDs lexicographically."""
    try:
        return (0, int(flag_id.removeprefix("flag-")))
    except ValueError:
        return (1, flag_id)


def load_benchmark_results(
    bench_dir: str,
    predictions_path: str,
    allow_subset: bool = False,
) -> Dict[str, dict]:
    """Join participant predictions with ground truth from the benchmark."""
    bench_path = Path(bench_dir).expanduser().resolve()
    if not bench_path.is_dir():
        raise FileNotFoundError(f"Benchmark directory does not exist: {bench_path}")

    predictions = load_predictions(predictions_path)
    bench_flags = {path.name for path in bench_path.glob("flag-*") if path.is_dir()}
    unknown_flags = sorted(set(predictions) - bench_flags, key=_flag_sort_key)
    if unknown_flags:
        raise ValueError(f"Predictions contain unknown flags: {', '.join(unknown_flags[:10])}")

    missing_flags = sorted(bench_flags - set(predictions), key=_flag_sort_key)
    if missing_flags and not allow_subset:
        preview = ", ".join(missing_flags[:10])
        suffix = "..." if len(missing_flags) > 10 else ""
        raise ValueError(
            f"Predictions must cover all {len(bench_flags)} flags; "
            f"missing {len(missing_flags)}: {preview}{suffix}. "
            "Use --allow-subset only for debugging."
        )

    joined: Dict[str, dict] = {}
    for flag_id, prediction in predictions.items():
        meta_path = bench_path / flag_id / "meta-info.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata for {flag_id}: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        gt_insights = meta.get("insights", [])
        if not isinstance(gt_insights, list) or not all(isinstance(item, str) for item in gt_insights):
            raise ValueError(f"{meta_path}: insights must be a string list")
        joined[flag_id] = {
            "pred_insights": prediction["pred_insights"],
            "gt_insights": gt_insights,
            "goal": meta.get("goal", ""),
        }
    return joined


def score_single_flag(
    flag_id: str,
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str,
    model_name: str,
) -> dict:
    """Score one flag and convert the result to a JSON-compatible dictionary."""
    print(f"\n[{flag_id}] scoring with {model_name}")
    print(f"  predicted insights: {len(pred_insights)}")
    print(f"  ground-truth insights: {len(gt_insights)}")
    try:
        scores = compute_four_dim_score(
            pred_insights=pred_insights,
            gt_insights=gt_insights,
            goal=goal,
            model_name=model_name,
            remove_prefixes=True,
        )
        result = _scores_to_dict(flag_id, scores)
        print(f"  weighted average: {result['weighted_avg']:.4f}")
        return result
    except Exception as exc:
        print(f"  evaluation error: {exc}")
        return _error_result(flag_id, str(exc))


def _scores_to_dict(flag_id: str, scores: FourDimScores) -> dict:
    """Convert the score dataclass to rounded output fields."""
    return {
        "flag_id": flag_id,
        "factuality": round(scores.factuality, 6),
        "factuality_hard": round(scores.factuality_hard, 6),
        "factuality_llm": round(scores.factuality_llm, 6),
        "completeness": round(scores.completeness, 6),
        "logic": round(scores.logic, 6),
        "insightfulness": round(scores.insightfulness, 6),
        "weighted_avg": round(scores.weighted_avg, 6),
    }


def _error_result(flag_id: str, error: str) -> dict:
    """Create a zero-valued result that records an evaluation error."""
    return {
        "flag_id": flag_id,
        "factuality": 0.0,
        "factuality_hard": 0.0,
        "factuality_llm": 0.0,
        "completeness": 0.0,
        "logic": 0.0,
        "insightfulness": 0.0,
        "weighted_avg": 0.0,
        "error": error,
    }


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint or return an empty checkpoint structure."""
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {"completed": [], "results": []}


def save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Write a checkpoint, creating its parent directory when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(checkpoint, handle, ensure_ascii=False, indent=2)


def compute_stratified_summary(results: List[dict], classification: Dict[str, str]) -> dict:
    """Compute easy, hard, and overall averages for each metric."""
    valid_results = [result for result in results if "error" not in result]
    easy_results = [r for r in valid_results if classification.get(r["flag_id"]) == "easy"]
    hard_results = [r for r in valid_results if classification.get(r["flag_id"]) == "hard"]
    dimensions = [
        "factuality",
        "factuality_hard",
        "factuality_llm",
        "completeness",
        "logic",
        "insightfulness",
        "weighted_avg",
    ]
    names = {
        "factuality": "fact",
        "factuality_hard": "fact_hard",
        "factuality_llm": "fact_llm",
        "insightfulness": "insight",
        "weighted_avg": "overall",
    }

    summary = {
        "total_flags": len(valid_results),
        "easy_count": len(easy_results),
        "hard_count": len(hard_results),
    }
    for dimension in dimensions:
        name = names.get(dimension, dimension)
        for suffix, group in (
            ("easy", easy_results),
            ("hard", hard_results),
            ("avg", valid_results),
        ):
            values = [result[dimension] for result in group]
            summary[f"{name}_{suffix}"] = round(sum(values) / len(values), 6) if values else 0.0
    return summary


def write_detailed_json(path: Path, results: List[dict]) -> None:
    """Write per-flag evaluation results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"evaluated_at": datetime.now().isoformat(), "flags": results},
            handle,
            ensure_ascii=False,
            indent=2,
        )


def write_summary_csv(path: Path, summary: dict) -> None:
    """Write one row containing stratified averages."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "fact_easy", "fact_hard", "fact_avg",
        "completeness_easy", "completeness_hard", "completeness_avg",
        "logic_easy", "logic_hard", "logic_avg",
        "insight_easy", "insight_hard", "insight_avg",
        "overall_easy", "overall_hard", "overall_avg",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric"] + fields)
        writer.writeheader()
        writer.writerow({"metric": "ALL", **{field: summary.get(field, 0.0) for field in fields}})


def print_summary(summary: dict) -> None:
    """Print a compact stratified summary."""
    print("\n" + "=" * 80)
    print("Stratified summary")
    print("=" * 80)
    print(f"{'Metric':<18} {'Easy':>10} {'Hard':>10} {'Average':>10}")
    print("-" * 52)
    for metric, prefix in (
        ("factuality", "fact"),
        ("completeness", "completeness"),
        ("logic", "logic"),
        ("insightfulness", "insight"),
        ("overall", "overall"),
    ):
        print(
            f"{metric:<18} {summary.get(prefix + '_easy', 0.0):>10.4f} "
            f"{summary.get(prefix + '_hard', 0.0):>10.4f} "
            f"{summary.get(prefix + '_avg', 0.0):>10.4f}"
        )
    print("-" * 52)
    print(f"Total flags: {summary.get('total_flags', 0)} "
          f"(easy: {summary.get('easy_count', 0)}, hard: {summary.get('hard_count', 0)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DataCrossBench predictions with easy/hard stratification."
    )
    parser.add_argument("--predictions", "-p", required=True, help="Participant predictions JSON")
    parser.add_argument("--bench", required=True, help="DataCrossBench root directory")
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="Allow partial predictions for debugging only",
    )
    parser.add_argument(
        "--model", "-m", default=os.getenv("EVAL_MODEL", "gpt-4o"),
        help="LLM evaluator model (default: EVAL_MODEL or gpt-4o)",
    )
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from the checkpoint")
    parser.add_argument(
        "--classification", default=str(CLASSIFICATION_FILE),
        help=f"Easy/hard classification JSON (default: {CLASSIFICATION_FILE})",
    )
    parser.add_argument(
        "--output", "-o", default=str(RESULTS_DIR / "detailed_eval_results.json"),
        help="Detailed JSON output path",
    )
    parser.add_argument(
        "--csv", default=str(RESULTS_DIR / "eval_summary.csv"),
        help="Summary CSV output path",
    )
    args = parser.parse_args()

    try:
        all_data = load_benchmark_results(
            args.bench,
            args.predictions,
            allow_subset=args.allow_subset,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    classification = load_flag_classification(Path(args.classification))
    flag_ids = sorted(all_data, key=_flag_sort_key)
    output_path = Path(args.output).expanduser()
    checkpoint_path = output_path.parent / "checkpoint.json"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        completed = set(checkpoint.get("completed", []))
        results = checkpoint.get("results", [])
    else:
        completed = set()
        results = []

    for flag_id in flag_ids:
        if flag_id in completed:
            continue
        item = all_data[flag_id]
        result = score_single_flag(
            flag_id,
            item["pred_insights"],
            item["gt_insights"],
            item["goal"],
            args.model,
        )
        results = [existing for existing in results if existing.get("flag_id") != flag_id]
        results.append(result)
        if "error" not in result:
            completed.add(flag_id)
        save_checkpoint(checkpoint_path, {"completed": sorted(completed), "results": results})
        time.sleep(1)

    summary = compute_stratified_summary(results, classification)
    write_detailed_json(output_path, results)
    write_summary_csv(Path(args.csv).expanduser(), summary)
    print_summary(summary)
    print(f"Detailed results: {output_path}")
    print(f"Summary CSV: {args.csv}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
