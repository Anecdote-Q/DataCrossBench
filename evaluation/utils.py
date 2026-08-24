"""
Utility functions for ablation experiment scoring.
"""

import os
import re
import json
import glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


# =====================
# Text Processing
# =====================

def remove_insight_prefix(insight: str) -> str:
    """Remove insight type prefixes (Trend:, Comparison:, Extreme:, Attribution:).
    
    Args:
        insight: The insight string that may contain a prefix.
        
    Returns:
        The insight string with the prefix removed.
    """
    prefixes = ["Trend:", "Comparison:", "Extreme:", "Attribution:"]
    insight = insight.strip()
    for prefix in prefixes:
        if insight.startswith(prefix):
            return insight[len(prefix):].strip()
    return insight


def remove_all_prefixes(insights: List[str]) -> List[str]:
    """Remove prefixes from a list of insights.
    
    Args:
        insights: List of insight strings.
        
    Returns:
        List of insights with prefixes removed.
    """
    return [remove_insight_prefix(i) for i in insights]


def extract_numbers(text: str) -> List[Tuple[float, str]]:
    """Extract numbers with their units from text.
    
    Extracts patterns like:
    - 65% -> (65, '%')
    - 119.66亿元 -> (119.66, '亿元')
    - 28.98% -> (28.98, '%')
    - 539,029.89元 -> (539029.89, '元')
    
    Args:
        text: The text to extract numbers from.
        
    Returns:
        List of tuples (number, unit).
    """
    # Pattern to match numbers with optional units
    # Matches: integers, decimals, numbers with commas, followed by optional Chinese/English units
    pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s*(%|亿|万|元|亿元|万元|个|次|人|家|条|%)?'
    
    matches = re.findall(pattern, text)
    results = []
    
    for num_str, unit in matches:
        # Remove commas from number
        num_str = num_str.replace(',', '')
        try:
            num = float(num_str)
            results.append((num, unit if unit else ''))
        except ValueError:
            continue
    
    return results


def compute_number_match_rate(pred_text: str, gt_text: str, tolerance: float = 0.1) -> float:
    """Compute the match rate of numbers between prediction and ground truth.
    
    Args:
        pred_text: The predicted text.
        gt_text: The ground truth text.
        tolerance: Relative tolerance for number matching (default 10%).
        
    Returns:
        Match rate between 0 and 1.
    """
    pred_numbers = extract_numbers(pred_text)
    gt_numbers = extract_numbers(gt_text)
    
    if not gt_numbers:
        return 1.0  # No numbers to match
    
    matched = 0
    for gt_num, gt_unit in gt_numbers:
        for pred_num, pred_unit in pred_numbers:
            # Check if units match (or both are empty)
            if gt_unit == pred_unit or (not gt_unit and not pred_unit):
                # Check if numbers are close enough
                if gt_num == 0:
                    if pred_num == 0:
                        matched += 1
                        break
                elif abs(pred_num - gt_num) / abs(gt_num) <= tolerance:
                    matched += 1
                    break
    
    return matched / len(gt_numbers)


# =====================
# Data Loading
# =====================

@dataclass
class ExperimentData:
    """Data class for experiment results."""
    flag_id: str
    experiment_name: str
    synthesized_insights: List[str]
    raw_insights: List[str]
    summary: str
    config: Dict[str, Any]
    execution_time: float


@dataclass
class GTData:
    """Data class for ground truth data."""
    flag_id: str
    goal: str
    insights: List[str]
    summary: str
    role: str
    classifier: str


def load_comparison_report(report_path: str) -> Dict[str, Any]:
    """Load a comparison_report.json file.
    
    Args:
        report_path: Path to the comparison_report.json file.
        
    Returns:
        Dictionary containing the comparison report data.
    """
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_gt_data(meta_info_path: str) -> GTData:
    """Load ground truth data from meta-info.json.
    
    Args:
        meta_info_path: Path to the meta-info.json file.
        
    Returns:
        GTData object containing ground truth data.
    """
    with open(meta_info_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return GTData(
        flag_id=data.get('flag_id', ''),
        goal=data.get('goal', ''),
        insights=data.get('insights', []),
        summary=data.get('summary', ''),
        role=data.get('role', ''),
        classifier=data.get('classifier', ''),
    )


def extract_raw_insights(raw_insights_list: List[Dict]) -> List[str]:
    """Extract insight strings from raw_insights list.
    
    Each raw insight is a dict with 'source', 'insight', 'question' keys.
    We only extract the 'insight' field.
    
    Args:
        raw_insights_list: List of raw insight dictionaries.
        
    Returns:
        List of insight strings.
    """
    return [item.get('insight', '') for item in raw_insights_list if item.get('insight')]


def parse_experiment_data(exp_data: Dict, flag_id: str, exp_name: str) -> ExperimentData:
    """Parse experiment data from comparison report.
    
    Args:
        exp_data: Dictionary containing experiment data.
        exp_name: Name of the experiment.
        flag_id: Flag ID.
        
    Returns:
        ExperimentData object.
    """
    return ExperimentData(
        flag_id=flag_id,
        experiment_name=exp_name,
        synthesized_insights=exp_data.get('synthesized_insights', []),
        raw_insights=extract_raw_insights(exp_data.get('raw_insights', [])),
        summary=exp_data.get('summary', ''),
        config=exp_data.get('config', {}),
        execution_time=exp_data.get('execution_time_seconds', 0),
    )


def collect_all_experiment_results(
    ablation_base_dir: str,
    gt_base_dir: str,
    skip_flags: Optional[List[int]] = None
) -> Dict[str, Dict[str, Any]]:
    """Collect all experiment results from ablation directories.
    
    Args:
        ablation_base_dir: Base directory containing ablation_* folders.
        gt_base_dir: Base directory containing flag-* folders with meta-info.json.
        skip_flags: List of flag numbers to skip (e.g., [15, 37]).
        
    Returns:
        Dictionary mapping flag_id to experiment results and GT data.
    """
    if skip_flags is None:
        skip_flags = [15, 37]
    
    results = {}
    
    # Find all ablation directories
    ablation_dirs = sorted(glob.glob(os.path.join(ablation_base_dir, "ablation_*")))
    
    if not ablation_dirs:
        print(f"Warning: No ablation directories found in {ablation_base_dir}")
        return results
    
    # Process each ablation directory
    for ablation_dir in ablation_dirs:
        # Find all flag directories
        flag_dirs = glob.glob(os.path.join(ablation_dir, "flag-*"))
        
        for flag_dir in flag_dirs:
            flag_name = os.path.basename(flag_dir)
            
            # Extract flag number and check if should skip
            try:
                flag_num = int(flag_name.replace("flag-", ""))
                if flag_num in skip_flags:
                    continue
            except ValueError:
                continue
            
            # Check if comparison_report.json exists
            report_path = os.path.join(flag_dir, "comparison_report.json")
            if not os.path.exists(report_path):
                continue
            
            # Load GT data
            gt_path = os.path.join(gt_base_dir, flag_name, "meta-info.json")
            if not os.path.exists(gt_path):
                print(f"Warning: GT data not found for {flag_name} at {gt_path}")
                continue
            
            gt_data = load_gt_data(gt_path)
            
            # Load comparison report
            try:
                report = load_comparison_report(report_path)
            except Exception as e:
                print(f"Error loading {report_path}: {e}")
                continue
            
            # Parse experiments
            experiments = {}
            for exp_name, exp_data in report.get('experiments', {}).items():
                experiments[exp_name] = parse_experiment_data(exp_data, flag_name, exp_name)
            
            # Only add if we have experiments
            if experiments:
                # If flag already exists (from another ablation dir), merge or update
                if flag_name in results:
                    # Keep the one with more experiments or the newer one
                    if len(experiments) > len(results[flag_name].get('experiments', {})):
                        results[flag_name] = {
                            'gt': gt_data,
                            'experiments': experiments,
                            'source_dir': ablation_dir,
                        }
                else:
                    results[flag_name] = {
                        'gt': gt_data,
                        'experiments': experiments,
                        'source_dir': ablation_dir,
                    }
    
    return results


def save_json(path: str, data: Any, indent: int = 2) -> None:
    """Save data to a JSON file.
    
    Args:
        path: Path to save the file.
        data: Data to save.
        indent: JSON indentation level.
    """
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_single_flag_from_gather(
    gather_dir: str,
    flag_num: int,
    gt_dir: str,
) -> Optional[Dict[str, Any]]:
    """Load experiment results for a single flag from the gather directory.
    
    This is optimized for the consolidated gather directory structure:
    gather/flag-*/comparison_report.json
    
    Args:
        gather_dir: Path to the gather directory (e.g., AblationExps/gather)
        flag_num: Flag number to load
        gt_dir: Directory containing ground truth meta-info.json files
        
    Returns:
        Dictionary with 'gt' and 'experiments' keys, or None if not found
    """
    flag_id = f"flag-{flag_num}"
    
    # Load comparison report
    report_path = os.path.join(gather_dir, flag_id, "comparison_report.json")
    if not os.path.exists(report_path):
        return None
    
    # Load GT data
    gt_path = os.path.join(gt_dir, flag_id, "meta-info.json")
    if not os.path.exists(gt_path):
        print(f"Warning: GT data not found for {flag_id}")
        return None
    
    try:
        gt_data = load_gt_data(gt_path)
        report = load_comparison_report(report_path)
    except Exception as e:
        print(f"Error loading data for {flag_id}: {e}")
        return None
    
    # Parse experiments
    experiments = {}
    for exp_name, exp_data in report.get('experiments', {}).items():
        experiments[exp_name] = parse_experiment_data(exp_data, flag_id, exp_name)
    
    if not experiments:
        return None
    
    return {
        'gt': gt_data,
        'experiments': experiments,
    }


def collect_flags_from_gather(
    gather_dir: str,
    gt_dir: str,
    flag_nums: List[int],
) -> Dict[str, Dict[str, Any]]:
    """Collect experiment results from gather directory for specified flags.
    
    Args:
        gather_dir: Path to the gather directory
        gt_dir: Directory containing ground truth data
        flag_nums: List of flag numbers to load
        
    Returns:
        Dictionary mapping flag_id to experiment results
    """
    results = {}
    
    for flag_num in flag_nums:
        flag_id = f"flag-{flag_num}"
        data = load_single_flag_from_gather(gather_dir, flag_num, gt_dir)
        if data:
            results[flag_id] = data
    
    return results


def format_insights_for_eval(insights: List[str], with_numbers: bool = True) -> str:
    """Format a list of insights for LLM evaluation.
    
    Args:
        insights: List of insight strings.
        with_numbers: Whether to number the insights.
        
    Returns:
        Formatted string.
    """
    if with_numbers:
        return "\n".join([f"{i+1}. {insight}" for i, insight in enumerate(insights)])
    return "\n".join(insights)
