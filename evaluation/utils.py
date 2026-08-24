"""Text utilities used by the DataCrossBench evaluator."""

import re
from typing import List, Tuple


INSIGHT_PREFIXES = ("Trend:", "Comparison:", "Extreme:", "Attribution:")


def remove_insight_prefix(insight: str) -> str:
    """Remove one supported insight prefix from the beginning of a string."""
    insight = insight.strip()
    for prefix in INSIGHT_PREFIXES:
        if insight.startswith(prefix):
            return insight[len(prefix):].strip()
    return insight


def remove_all_prefixes(insights: List[str]) -> List[str]:
    """Remove supported insight prefixes from a list of strings."""
    return [remove_insight_prefix(insight) for insight in insights]


def extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract numeric values and their immediately following units."""
    pattern = (
        r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
        r"(%|亿|万|元|亿元|万元|个|次|人|家|条|%)?"
    )
    results: List[Tuple[float, str]] = []
    for number, unit in re.findall(pattern, text):
        try:
            results.append((float(number.replace(",", "")), unit or ""))
        except ValueError:
            continue
    return results


def compute_number_match_rate(
    pred_text: str,
    gt_text: str,
    tolerance: float = 0.1,
) -> float:
    """Return the fraction of ground-truth numbers matched by a prediction."""
    pred_numbers = extract_numbers(pred_text)
    gt_numbers = extract_numbers(gt_text)
    if not gt_numbers:
        return 1.0

    matched = 0
    for gt_number, gt_unit in gt_numbers:
        for pred_number, pred_unit in pred_numbers:
            if gt_unit != pred_unit and (gt_unit or pred_unit):
                continue
            if gt_number == 0:
                is_match = pred_number == 0
            else:
                is_match = abs(pred_number - gt_number) / abs(gt_number) <= tolerance
            if is_match:
                matched += 1
                break
    return matched / len(gt_numbers)


def format_insights_for_eval(insights: List[str], with_numbers: bool = True) -> str:
    """Format insights as a numbered or unnumbered prompt block."""
    if with_numbers:
        return "\n".join(f"{index + 1}. {insight}" for index, insight in enumerate(insights))
    return "\n".join(insights)
