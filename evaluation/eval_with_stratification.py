"""
eval_with_stratification.py

对 Bench + predictions，或旧版 detailed_results.json 中的 flag 进行四维打分，并按 easy/hard 分层统计。

功能：
1. 调用 LLM 对 pred_insights 和 gt_insights 进行四维评估
2. 按 easy（无图片）和 hard（有图片）分层统计
3. 输出详细的分层结果

输出：
  - results/detailed_eval_results.json    每条 flag 的详细打分结果
  - results/eval_summary.csv             汇总表（含 easy/hard/avg 分层）
"""

import os
import sys
import json
import csv
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
RESULTS_DIR = SCRIPT_DIR / "results"
CLASSIFICATION_FILE = SCRIPT_DIR / "flag_classification.json"

# 默认数据文件路径（仅用于兼容已经合并好的旧格式）
DEFAULT_DATA_FILE = SCRIPT_DIR / "detailed_results.json"

# metrics_custom.py 在同一目录下
sys.path.insert(0, str(SCRIPT_DIR))
from metrics_custom import compute_four_dim_score, FourDimScores, WEIGHTS


# ── 分层分类加载 ──────────────────────────────────────────────────────────────

def load_flag_classification(path: Path) -> Dict[str, str]:
    """加载 flag 分类信息，返回 {flag_id: 'easy'|'hard'}"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("flags", {})


# ── 数据加载 ──────────────────────────────────────────────────────────────────

def _normalise_flag_records(data: Any, source: str) -> Dict[str, dict]:
    """将旧版 mapping、记录列表或 ``{"flags": [...]}`` 统一为 flag mapping。"""
    if isinstance(data, dict) and isinstance(data.get("flags"), list):
        records = data["flags"]
    elif isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = []
        for flag_id, item in data.items():
            if not isinstance(item, dict):
                raise ValueError(f"{source}: {flag_id} 的值必须是对象")
            record = dict(item)
            record.setdefault("flag_id", flag_id)
            records.append(record)
    else:
        raise ValueError(f"{source}: 顶层必须是对象、记录列表或 {{'flags': [...]}}")

    normalised = {}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{source}: 第 {index + 1} 条记录不是对象")
        flag_id = record.get("flag_id")
        if not isinstance(flag_id, str) or not flag_id:
            raise ValueError(f"{source}: 第 {index + 1} 条记录缺少非空 flag_id")
        if flag_id in normalised:
            raise ValueError(f"{source}: 重复的 flag_id: {flag_id}")
        normalised[flag_id] = record
    return normalised


def load_detailed_results(path: str) -> Dict[str, dict]:
    """读取已经合并的评测输入，兼容 mapping、列表和 ``flags`` 包装格式。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalise_flag_records(data, path)


def load_predictions(path: str) -> Dict[str, dict]:
    """读取模型预测，返回 ``{flag_id: {"pred_insights": [...]}}``。

    支持以下输入：
    - ``{"flag-1": {"pred_insights": [...]}}``
    - ``{"flag-1": [...]}``
    - ``[{"flag_id": "flag-1", "pred_insights": [...]}]``
    - ``{"predictions": [...]}`` 或 ``{"predictions": {...}}``
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "predictions" in data:
        data = data["predictions"]

    if isinstance(data, dict):
        records = []
        for flag_id, item in data.items():
            if isinstance(item, list):
                item = {"pred_insights": item}
            if not isinstance(item, dict):
                raise ValueError(f"{path}: {flag_id} 的预测必须是列表或对象")
            record = dict(item)
            record.setdefault("flag_id", flag_id)
            records.append(record)
    else:
        records = data

    predictions = _normalise_flag_records(records, path)
    for flag_id, record in predictions.items():
        insights = record.get("pred_insights")
        if not isinstance(insights, list) or not all(isinstance(x, str) for x in insights):
            raise ValueError(f"{path}: {flag_id}.pred_insights 必须是字符串列表")
        record["pred_insights"] = insights
    return predictions


def load_benchmark_results(
    bench_dir: str,
    predictions_path: str,
    allow_subset: bool = False,
) -> Dict[str, dict]:
    """把公开 Bench 的 meta-info 与模型预测连接成评测器输入。"""
    bench_path = Path(bench_dir).expanduser().resolve()
    if not bench_path.is_dir():
        raise FileNotFoundError(f"Bench 目录不存在: {bench_path}")

    predictions = load_predictions(predictions_path)
    bench_flags = {path.name for path in bench_path.glob("flag-*") if path.is_dir()}
    unknown_flags = sorted(set(predictions) - bench_flags, key=_flag_sort_key)
    if unknown_flags:
        raise ValueError(f"预测包含 Bench 中不存在的 flag: {', '.join(unknown_flags[:10])}")
    missing_flags = sorted(bench_flags - set(predictions), key=_flag_sort_key)
    if missing_flags and not allow_subset:
        preview = ", ".join(missing_flags[:10])
        more = "..." if len(missing_flags) > 10 else ""
        raise ValueError(
            f"预测未覆盖全部 {len(bench_flags)} 个 flag，缺少 {len(missing_flags)} 个: "
            f"{preview}{more}。调试子集时显式传 --allow-subset。"
        )
    joined = {}
    for flag_id, prediction in predictions.items():
        meta_path = bench_path / flag_id / "meta-info.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{flag_id} 的 meta-info.json 不存在: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        gt_insights = meta.get("insights", [])
        goal = meta.get("goal", "")
        if not isinstance(gt_insights, list) or not all(isinstance(x, str) for x in gt_insights):
            raise ValueError(f"{meta_path}: insights 必须是字符串列表")
        joined[flag_id] = {
            "pred_insights": prediction["pred_insights"],
            "gt_insights": gt_insights,
            "goal": goal,
        }
    return joined


def _flag_sort_key(flag_id: str):
    """按 flag 数字排序，同时对非标准 ID 保持稳定排序。"""
    try:
        return (0, int(flag_id.removeprefix("flag-")))
    except ValueError:
        return (1, flag_id)


# ── 打分 ───────────────────────────────────────────────────────────────────────

def score_single_flag(
    flag_id: str,
    pred_insights: list,
    gt_insights: list,
    goal: str,
    model_name: str,
    max_retries: int = 3,
) -> dict:
    """对单条 flag 调用四维打分"""
    print(f"\n  [{flag_id}] 开始打分 (model={model_name})")
    print(f"    pred_insights: {len(pred_insights)} 条")
    print(f"    gt_insights:   {len(gt_insights)} 条")

    try:
        scores = compute_four_dim_score(
            pred_insights=pred_insights,
            gt_insights=gt_insights,
            goal=goal,
            model_name=model_name,
            remove_prefixes=True,
        )
        result = _scores_to_dict(flag_id, scores)
        print(f"    完成 → weighted_avg={result['weighted_avg']:.4f}")
        return result
    except Exception as e:
        print(f"    出错: {e}")
        return _error_result(flag_id, str(e))


def _scores_to_dict(flag_id: str, s: FourDimScores) -> dict:
    return {
        "flag_id": flag_id,
        "factuality":       round(s.factuality,      6),
        "factuality_hard":  round(s.factuality_hard, 6),
        "factuality_llm":   round(s.factuality_llm,  6),
        "completeness":     round(s.completeness,    6),
        "logic":            round(s.logic,           6),
        "insightfulness":   round(s.insightfulness,  6),
        "weighted_avg":     round(s.weighted_avg,    6),
    }


def _error_result(flag_id: str, err: str) -> dict:
    return {
        "flag_id": flag_id,
        "factuality": 0.0, "factuality_hard": 0.0, "factuality_llm": 0.0,
        "completeness": 0.0, "logic": 0.0, "insightfulness": 0.0,
        "weighted_avg": 0.0, "error": err,
    }


# ── 断点管理 ──────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(path: Path, checkpoint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)


# ── 分层统计 ──────────────────────────────────────────────────────────────────

def compute_stratified_summary(
    results: List[dict],
    classification: Dict[str, str],
) -> dict:
    """
    根据分层信息计算各维度的 easy/hard/avg 统计。

    输出 15 个分数：
    - factuality_easy, factuality_hard, factuality_avg
    - completeness_easy, completeness_hard, completeness_avg
    - logic_easy, logic_hard, logic_avg
    - insightfulness_easy, insightfulness_hard, insightfulness_avg
    - overall_easy, overall_hard, overall_avg
    """
    # 分离 easy 和 hard
    easy_results = []
    hard_results = []
    all_results = []

    for r in results:
        if "error" in r:
            continue
        flag_id = r["flag_id"]
        category = classification.get(flag_id, "unknown")
        all_results.append(r)
        if category == "easy":
            easy_results.append(r)
        elif category == "hard":
            hard_results.append(r)

    # 计算各维度的平均分
    dimensions = ["factuality", "factuality_hard", "factuality_llm",
                  "completeness", "logic", "insightfulness", "weighted_avg"]

    summary = {
        "total_flags": len(all_results),
        "easy_count": len(easy_results),
        "hard_count": len(hard_results),
        "classification_source": str(CLASSIFICATION_FILE),
    }

    for dim in dimensions:
        easy_vals = [r[dim] for r in easy_results] if easy_results else [0]
        hard_vals = [r[dim] for r in hard_results] if hard_results else [0]
        all_vals = [r[dim] for r in all_results] if all_results else [0]

        short_dim = {
            "factuality": "fact",
            "factuality_hard": "fact_hard",
            "factuality_llm": "fact_llm",
            "insightfulness": "insight",
            "weighted_avg": "overall",
        }.get(dim, dim)

        summary[f"{short_dim}_easy"] = round(sum(easy_vals) / len(easy_vals), 6) if easy_vals else 0.0
        summary[f"{short_dim}_hard"] = round(sum(hard_vals) / len(hard_vals), 6) if hard_vals else 0.0
        summary[f"{short_dim}_avg"]  = round(sum(all_vals) / len(all_vals), 6) if all_vals else 0.0

    return summary


# ── 输出 ──────────────────────────────────────────────────────────────────────

def write_detailed_json(out_path: Path, results: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"evaluated_at": datetime.now().isoformat(),
                   "flags": results}, f, ensure_ascii=False, indent=2)


def write_summary_csv(csv_path: Path, summary: dict):
    """写入单行分层汇总 CSV"""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    layers = [
        "fact_easy", "fact_hard", "fact_avg",
        "completeness_easy", "completeness_hard", "completeness_avg",
        "logic_easy", "logic_hard", "logic_avg",
        "insight_easy", "insight_hard", "insight_avg",
        "overall_easy", "overall_hard", "overall_avg",
    ]

    fieldnames = ["metric"] + layers

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        row = {"metric": "ALL"}
        for key in layers:
            row[key] = summary.get(key, 0.0)
        writer.writerow(row)


def print_summary_table(summary: dict):
    """打印分层汇总表"""
    print("\n" + "=" * 90)
    print("分层汇总结果")
    print("=" * 90)

    # 表头
    header = f"{'Metric':<18} {'Easy':>10} {'Hard':>10} {'Avg':>10}"
    print(header)
    print("-" * 50)

    for metric, prefix in [
        ("factuality", "fact"),
        ("completeness", "completeness"),
        ("logic", "logic"),
        ("insightfulness", "insight"),
        ("overall", "overall"),
    ]:
        easy = summary.get(f"{prefix}_easy", 0.0)
        hard = summary.get(f"{prefix}_hard", 0.0)
        avg  = summary.get(f"{prefix}_avg", 0.0)
        print(f"{metric:<18} {easy:>10.4f} {hard:>10.4f} {avg:>10.4f}")

    print("-" * 50)
    print(f"Total flags: {summary.get('total_flags', 0)} "
          f"(easy: {summary.get('easy_count', 0)}, hard: {summary.get('hard_count', 0)})")
    print("=" * 90)

    # 打印详细表格（每个 flag）
    print("\n" + "=" * 120)
    print("详细结果")
    print("=" * 120)
    print(f"{'flag_id':<10} {'category':>8} {'fact':>8} {'fact_h':>8} {'fact_llm':>9} "
          f"{'comp':>8} {'logic':>8} {'insight':>9} {'w_avg':>8}")
    print("-" * 120)


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="四维打分（分层版）：评估 Bench 预测，并按 easy/hard 分层统计"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="已经合并好的评测输入 JSON（含 pred_insights/gt_insights）",
    )
    input_group.add_argument(
        "--predictions", "-p",
        type=str,
        default=None,
        help="模型预测 JSON；需与 --bench 一起使用",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="DataCrossBench 根目录（与 --predictions 一起使用）",
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help="允许 --predictions 只覆盖部分 flag（仅建议调试使用）",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=os.getenv("EVAL_MODEL", "gpt-4o"),
        help="用于评估的模型 (默认: gpt-4o 或环境变量 EVAL_MODEL)",
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="从 checkpoint.json 恢复，继续执行中断的任务",
    )
    parser.add_argument(
        "--classification",
        type=str,
        default=str(CLASSIFICATION_FILE),
        help=f"flag 分类 JSON 文件路径 (默认: {CLASSIFICATION_FILE})",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(RESULTS_DIR / "detailed_eval_results.json"),
        help="详细结果 JSON 输出路径",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "eval_summary.csv"),
        help="汇总 CSV 输出路径",
    )
    args = parser.parse_args()

    if args.predictions and not args.bench:
        parser.error("--predictions 必须与 --bench 一起使用")
    if args.bench and not args.predictions:
        parser.error("--bench 必须与 --predictions 一起使用")

    if args.predictions:
        try:
            all_data = load_benchmark_results(
                args.bench,
                args.predictions,
                allow_subset=args.allow_subset,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        data_source = f"predictions={args.predictions}, bench={args.bench}"
    else:
        data_path = Path(args.data).expanduser() if args.data else DEFAULT_DATA_FILE
        if not data_path.exists():
            parser.error(
                "缺少评测输入。请使用 --predictions PREDICTIONS.json --bench DataCrossBench，"
                f"或提供旧格式 --data FILE（默认文件不存在: {data_path}）"
            )
        all_data = load_detailed_results(str(data_path))
        data_source = str(data_path)

    print(f"\n[eval_with_stratification] 开始执行")
    print(f"  数据来源       : {data_source}")
    print(f"  分类文件       : {args.classification}")
    print(f"  评估模型       : {args.model}")
    print(f"  断点恢复       : {'是' if args.resume else '否'}")

    # 加载分层分类
    classification = load_flag_classification(Path(args.classification))
    print(f"  已加载 {len(classification)} 个 flag 的分类信息")

    # flag ID 允许标准 flag-N，也允许旧数据中的非标准 ID
    flag_ids = sorted(all_data.keys(), key=_flag_sort_key)

    # 检查点
    checkpoint_path = RESULTS_DIR / "checkpoint.json"
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_path)
        completed = set(checkpoint.get("completed", []))
        results = checkpoint.get("results", [])
        print(f"  已完成 {len(completed)} 条，从断点恢复")
    else:
        completed = set()
        results = []

    # 逐条打分
    for flag_id in flag_ids:
        if flag_id in completed:
            print(f"  [{flag_id}] 已完成，跳过")
            continue

        item = all_data.get(flag_id)
        if not item:
            print(f"  [{flag_id}] 未找到数据，跳过")
            completed.add(flag_id)
            continue

        pred_insights = item.get("pred_insights", [])
        gt_insights   = item.get("gt_insights", [])
        goal          = item.get("goal", "")

        if not isinstance(pred_insights, list) or not all(isinstance(x, str) for x in pred_insights):
            result = _error_result(flag_id, "pred_insights 必须是字符串列表")
            results.append(result)
            print(f"  [{flag_id}] pred_insights 格式错误，未标记为完成")
            save_checkpoint(checkpoint_path, {"completed": list(completed), "results": results})
            continue
        if not isinstance(gt_insights, list) or not all(isinstance(x, str) for x in gt_insights):
            result = _error_result(flag_id, "gt_insights 必须是字符串列表")
            results.append(result)
            print(f"  [{flag_id}] gt_insights 格式错误，未标记为完成")
            save_checkpoint(checkpoint_path, {"completed": list(completed), "results": results})
            continue

        result = score_single_flag(flag_id, pred_insights, gt_insights, goal, args.model)
        results.append(result)
        if "error" not in result:
            completed.add(flag_id)
        else:
            print(f"  [{flag_id}] 评测异常，保留在结果中并允许下次 --resume 重试")

        # 每次打分后立即保存断点
        save_checkpoint(checkpoint_path, {"completed": list(completed), "results": results})

        # API 限速
        time.sleep(1)

    # 计算分层汇总
    summary = compute_stratified_summary(results, classification)

    # 输出
    write_detailed_json(Path(args.output), results)
    write_summary_csv(Path(args.csv), summary)

    # 打印结果
    print_summary_table(summary)
    print("\n" + "-" * 120)
    for r in results:
        if "error" in r:
            continue
        category = classification.get(r["flag_id"], "unknown")
        print(
            f"{r['flag_id']:<10} "
            f"{category:>8} "
            f"{r['factuality']:>8.4f} "
            f"{r['factuality_hard']:>8.4f} "
            f"{r['factuality_llm']:>9.4f} "
            f"{r['completeness']:>8.4f} "
            f"{r['logic']:>8.4f} "
            f"{r['insightfulness']:>9.4f} "
            f"{r['weighted_avg']:>8.4f}"
        )

    print("\n[eval_with_stratification] 完成！")
    print(f"  详细结果: {args.output}")
    print(f"  汇总 CSV: {args.csv}")
    print(f"  断点文件: {checkpoint_path}")


if __name__ == "__main__":
    main()
