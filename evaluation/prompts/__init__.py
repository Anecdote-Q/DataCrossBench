"""
Prompt templates for the four-dimension evaluation metrics.

Used by metrics_custom.py to generate LLM evaluation prompts for:
- Factuality: 30% weight
- Completeness: 25% weight
- Logic: 20% weight
- Insightfulness: 25% weight
"""

# ==============================================================================
# System Messages
# ==============================================================================

SYSTEM_MESSAGE_FACTUALITY = (
    "You are an expert data analyst evaluating the factual accuracy of insights. "
    "You will compare predicted insights against ground truth and assign a score of 1-10. "
    "A higher score means the prediction is more factually accurate."
)

SYSTEM_MESSAGE_COMPLETENESS = (
    "You are an expert data analyst evaluating the completeness of insights. "
    "You will compare predicted insights against ground truth and assign a score of 1-10. "
    "A higher score means the prediction covers more ground truth content."
)

SYSTEM_MESSAGE_LOGIC = (
    "You are an expert data analyst evaluating the logical quality of insights. "
    "You will compare predicted insights against ground truth and assign a score of 1-10. "
    "A higher score means the prediction has better logical reasoning and structure."
)

SYSTEM_MESSAGE_INSIGHTFULNESS = (
    "You are an expert data analyst evaluating the insightfulness and depth of insights. "
    "You will compare predicted insights against ground truth and assign a score of 1-10. "
    "A higher score means the prediction provides deeper, more valuable insights. "
    "The analysis goal is provided as context to help you judge the value of the insights."
)


# ==============================================================================
# Evaluation Templates
# ==============================================================================

EVAL_TEMPLATE_FACTUALITY = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

You are evaluating the **factual accuracy** of predicted insights against ground truth insights.

**Scoring Criteria (1-10):**
- 9-10: All key facts (numbers, entities, relationships) in the ground truth are correctly captured in the prediction.
- 7-8: Most key facts are correct; minor inaccuracies or omissions.
- 5-6: Some facts are correct, but significant factual errors or important omissions exist.
- 3-4: Many factual errors; only a few surface-level facts match.
- 1-2: Mostly incorrect or fabricated facts; no meaningful factual alignment.

**Ground Truth Insights (reference standard):**
{gt_insights}

**Predicted Insights (to be evaluated):**
{pred_insights}

### Response Format:
Please provide a brief explanation of your rating, then output only the numerical rating wrapped in <rating> tags.

Example:
<rating>8</rating>

### Response:
"""

EVAL_TEMPLATE_COMPLETENESS = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

You are evaluating the **completeness** of predicted insights against ground truth insights.

**Scoring Criteria (1-10):**
- 9-10: The prediction covers all key aspects and dimensions of the ground truth; nothing important is missed.
- 7-8: The prediction covers most key aspects; a minor dimension or two is missing.
- 5-6: The prediction covers some aspects but misses important dimensions or details.
- 3-4: Significant coverage gaps; most key aspects are missing.
- 1-2: The prediction barely touches on the ground truth topics.

**Ground Truth Insights (reference standard):**
{gt_insights}

**Predicted Insights (to be evaluated):**
{pred_insights}

### Response Format:
Please provide a brief explanation of your rating, then output only the numerical rating wrapped in <rating> tags.

Example:
<rating>7</rating>

### Response:
"""

EVAL_TEMPLATE_LOGIC = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

You are evaluating the **logical quality** of predicted insights. Judge the prediction ONLY against the ground truth insights for logical consistency and reasoning quality.

**Scoring Criteria (1-10):**
- 9-10: The prediction demonstrates flawless logical reasoning; every conclusion is fully supported by the ground truth data, with clear cause-effect relationships and a coherent structure.
- 7-8: The prediction has strong logical flow; minor unsupported inferences exist but do not undermine the main argument.
- 5-6: The prediction shows some logical structure, yet contains one or more contradictions, unsupported leaps, or gaps in reasoning.
- 3-4: Weak logical coherence; multiple contradictions or non-sequiturs significantly weaken the argument.
- 1-2: No discernible logical structure; the content is fragmented, self-contradictory, or entirely unsupported by the ground truth.

**Ground Truth Insights (reference standard):**
{gt_insights}

**Predicted Insights (to be evaluated):**
{pred_insights}

### Response Format:
Please provide a brief explanation of your rating, then output only the numerical rating wrapped in <rating> tags.

Example:
<rating>8</rating>

### Response:
"""

EVAL_TEMPLATE_INSIGHTFULNESS = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

You are evaluating the **insightfulness and depth** of predicted insights against the ground truth. Judge whether the prediction adds genuine analytical value and uncovers meaningful patterns beyond what the ground truth already states.

**Scoring Criteria (1-10):**
- 9-10: The prediction reveals genuinely deep, non-trivial insights that go beyond surface-level restatement. It uncovers causal relationships, hidden patterns, or strategic implications that are highly relevant to the goal. Novel and non-obvious.
- 7-8: The prediction provides useful insights relevant to the goal, with clear added analytical value. Some depth is present, though not exceptional.
- 5-6: The prediction touches on relevant topics but remains largely at a surface level, merely restating obvious facts from the ground truth without meaningful extension.
- 3-4: Very shallow; mostly repeating ground truth content verbatim with negligible additional insight or analytical depth.
- 1-2: No meaningful insights; purely descriptive or a near-verbatim copy of the ground truth with no added value.

**Analysis Goal:**
{goal}

**Ground Truth Insights (reference standard):**
{gt_insights}

**Predicted Insights (to be evaluated):**
{pred_insights}

### Response Format:
Please provide a brief explanation of your rating, then output only the numerical rating wrapped in <rating> tags.

Example:
<rating>9</rating>

### Response:
"""


# ==============================================================================
# Main API
# ==============================================================================

def get_eval_prompt(metric_name: str):
    """
    Return the appropriate evaluation prompt template and system message
    for a given metric name.

    Args:
        metric_name: One of 'factuality', 'completeness', 'logic', 'insightfulness'.

    Returns:
        A tuple of (template_str, system_message_str).

    Raises:
        ValueError: If metric_name is not recognized.
    """
    if metric_name == "factuality":
        return EVAL_TEMPLATE_FACTUALITY, SYSTEM_MESSAGE_FACTUALITY
    elif metric_name == "completeness":
        return EVAL_TEMPLATE_COMPLETENESS, SYSTEM_MESSAGE_COMPLETENESS
    elif metric_name == "logic":
        return EVAL_TEMPLATE_LOGIC, SYSTEM_MESSAGE_LOGIC
    elif metric_name == "insightfulness":
        return EVAL_TEMPLATE_INSIGHTFULNESS, SYSTEM_MESSAGE_INSIGHTFULNESS
    else:
        raise ValueError(
            f"Unknown metric_name '{metric_name}'. "
            f"Expected one of: factuality, completeness, logic, insightfulness"
        )


__all__ = ["get_eval_prompt"]
