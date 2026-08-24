"""
Custom Four-Dimension Metrics for Insight Evaluation.

Dimensions:
- Factuality: 30% weight
- Completeness: 25% weight
- Logic: 20% weight
- Insightfulness: 25% weight
"""

import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import openai
from openai import OpenAI

try:
    from .prompts import get_eval_prompt
    from .utils import (
        format_insights_for_eval,
        compute_number_match_rate,
        remove_all_prefixes,
    )
except ImportError:
    from prompts import get_eval_prompt
    from utils import (
        format_insights_for_eval,
        compute_number_match_rate,
        remove_all_prefixes,
    )

# =====================
# Configuration
# =====================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

# Dimension weights
WEIGHTS = {
    "factuality": 0.30,
    "completeness": 0.25,
    "logic": 0.20,
    "insightfulness": 0.25,
}

# Default model for LLM evaluation
DEFAULT_MODEL = "gpt-4o"


@dataclass
class FourDimScores:
    """Data class for four-dimension evaluation scores."""
    factuality: float
    factuality_hard: float  # Hard metric component
    factuality_llm: float   # LLM component
    completeness: float
    logic: float
    insightfulness: float
    weighted_avg: float
    details: Dict[str, Any]


# =====================
# LLM-based Evaluation
# =====================

def compute_llm_score(
    pred_insights: List[str],
    gt_insights: List[str],
    metric_name: str,
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
    top_logprobs: int = 5,
    max_retries: int = 3,
) -> Tuple[float, str]:
    """Compute LLM-based evaluation score for a given metric.
    
    Uses logprobs for more stable scoring (similar to G-Eval).
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        metric_name: One of 'factuality', 'completeness', 'logic', 'insightfulness'.
        goal: Analysis goal (used for insightfulness evaluation).
        model_name: OpenAI model to use.
        top_logprobs: Number of top logprobs to use for weighted scoring.
        max_retries: Maximum number of retries on API errors.
        
    Returns:
        Tuple of (score normalized to 0-1, raw response text).
    """
    client = OpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY)
    template, system_message = get_eval_prompt(metric_name)
    
    # Format insights for evaluation
    pred_formatted = format_insights_for_eval(pred_insights)
    gt_formatted = format_insights_for_eval(gt_insights)
    
    # Fill in the template
    if metric_name == "insightfulness":
        prompt = template.format(
            pred_insights=pred_formatted,
            gt_insights=gt_formatted,
            goal=goal if goal else "No analysis goal was provided."
        )
    else:
        prompt = template.format(
            pred_insights=pred_formatted,
            gt_insights=gt_formatted
        )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=500,
                top_p=1,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            
            response_text = response.choices[0].message.content
            
            # Extract rating from response
            rating_match = re.findall(r'<rating>(\d+)</rating>', response_text)
            if not rating_match:
                # Try to find just a number at the end
                rating_match = re.findall(r'\b([1-9]|10)\b', response_text)
            
            if rating_match:
                rating_str = rating_match[0]
                
                # Try to use logprobs for weighted scoring
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    tokens = [o.token for o in response.choices[0].logprobs.content]
                    try:
                        rating_idx = tokens.index(rating_str)
                        top_probs = response.choices[0].logprobs.content[rating_idx].top_logprobs
                        
                        # Convert logprobs to probs and compute weighted score
                        probs = [np.exp(obj.logprob) for obj in top_probs]
                        probs = [p / sum(probs) for p in probs]
                        ratings = [float(obj.token) if obj.token.isdigit() else 0 for obj in top_probs]
                        
                        score = sum([r * p for r, p in zip(ratings, probs)])
                        return score / 10.0, response_text
                    except (ValueError, IndexError):
                        pass
                
                # Fallback to direct rating
                score = float(rating_str) / 10.0
                return score, response_text
            
            # If no rating found, return a default low score
            print(f"Warning: Could not extract rating from response for {metric_name}")
            return 0.5, response_text
            
        except openai.RateLimitError:
            print(f"Rate limit hit, sleeping for 60 seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(60)
        except openai.APIError as e:
            print(f"API error: {e}, sleeping for 30 seconds (attempt {attempt + 1}/{max_retries})")
            time.sleep(30)
        except Exception as e:
            print(f"Unexpected error in LLM scoring: {e}")
            if attempt == max_retries - 1:
                return 0.0, str(e)
            time.sleep(10)
    
    return 0.0, "Max retries exceeded"


# =====================
# Embedding-based Completeness
# =====================

def compute_embedding_completeness(
    pred_insights: List[str],
    gt_insights: List[str],
    model_name: str = "text-embedding-3-small",
) -> float:
    """Compute completeness using embedding similarity.
    
    For each GT insight, find the best matching pred insight and compute similarity.
    Completeness = mean of max similarities.
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        model_name: OpenAI embedding model to use.
        
    Returns:
        Completeness score between 0 and 1.
    """
    if not pred_insights or not gt_insights:
        return 0.0
    
    client = OpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY)
    
    try:
        # Get embeddings for all texts
        all_texts = pred_insights + gt_insights
        response = client.embeddings.create(
            model=model_name,
            input=all_texts,
        )
        
        embeddings = [item.embedding for item in response.data]
        pred_embeddings = embeddings[:len(pred_insights)]
        gt_embeddings = embeddings[len(pred_insights):]
        
        # Compute cosine similarity matrix
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # For each GT insight, find max similarity with any pred insight
        max_similarities = []
        for gt_emb in gt_embeddings:
            similarities = [cosine_similarity(gt_emb, pred_emb) for pred_emb in pred_embeddings]
            max_similarities.append(max(similarities))
        
        return np.mean(max_similarities)
        
    except Exception as e:
        print(f"Error computing embedding completeness: {e}")
        return 0.0


# =====================
# Main Scoring Functions
# =====================

def compute_factuality(
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[float, Dict[str, Any]]:
    """Compute factuality score (30% weight).
    
    Combines:
    - Hard metric (50%): Number match rate
    - LLM metric (50%): LLM-based factuality evaluation
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        goal: Analysis goal.
        model_name: OpenAI model to use.
        
    Returns:
        Tuple of (score, details dict).
    """
    # Hard metric: number match rate
    pred_text = "\n".join(pred_insights)
    gt_text = "\n".join(gt_insights)
    hard_score = compute_number_match_rate(pred_text, gt_text)
    
    # LLM metric
    llm_score, llm_response = compute_llm_score(
        pred_insights, gt_insights, "factuality", goal, model_name
    )
    
    # Combined score (50% hard, 50% LLM)
    combined_score = 0.5 * hard_score + 0.5 * llm_score
    
    details = {
        "hard_score": hard_score,
        "llm_score": llm_score,
        "llm_response": llm_response,
    }
    
    return combined_score, details


def compute_completeness(
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[float, Dict[str, Any]]:
    """Compute completeness score (25% weight).
    
    Uses embedding-based similarity for coverage measurement.
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        goal: Analysis goal (unused but kept for consistency).
        model_name: OpenAI model to use (unused for embeddings).
        
    Returns:
        Tuple of (score, details dict).
    """
    embedding_score = compute_embedding_completeness(pred_insights, gt_insights)
    
    details = {
        "embedding_score": embedding_score,
    }
    
    return embedding_score, details


def compute_logic(
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[float, Dict[str, Any]]:
    """Compute logic score (20% weight).
    
    Uses LLM-based evaluation for logical reasoning quality.
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        goal: Analysis goal.
        model_name: OpenAI model to use.
        
    Returns:
        Tuple of (score, details dict).
    """
    llm_score, llm_response = compute_llm_score(
        pred_insights, gt_insights, "logic", goal, model_name
    )
    
    details = {
        "llm_score": llm_score,
        "llm_response": llm_response,
    }
    
    return llm_score, details


def compute_insightfulness(
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
) -> Tuple[float, Dict[str, Any]]:
    """Compute insightfulness score (25% weight).
    
    Uses LLM-based evaluation for insight value and depth.
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        goal: Analysis goal.
        model_name: OpenAI model to use.
        
    Returns:
        Tuple of (score, details dict).
    """
    llm_score, llm_response = compute_llm_score(
        pred_insights, gt_insights, "insightfulness", goal, model_name
    )
    
    details = {
        "llm_score": llm_score,
        "llm_response": llm_response,
    }
    
    return llm_score, details


def compute_four_dim_score(
    pred_insights: List[str],
    gt_insights: List[str],
    goal: str = "",
    model_name: str = DEFAULT_MODEL,
    remove_prefixes: bool = True,
) -> FourDimScores:
    """Compute all four-dimension scores.
    
    Args:
        pred_insights: List of predicted insights.
        gt_insights: List of ground truth insights.
        goal: Analysis goal.
        model_name: OpenAI model to use.
        remove_prefixes: Whether to remove insight type prefixes before evaluation.
        
    Returns:
        FourDimScores object with all scores.
    """
    # Optionally remove prefixes
    if remove_prefixes:
        pred_insights = remove_all_prefixes(pred_insights)
        gt_insights = remove_all_prefixes(gt_insights)
    
    # Compute each dimension
    factuality_score, factuality_details = compute_factuality(
        pred_insights, gt_insights, goal, model_name
    )
    
    completeness_score, completeness_details = compute_completeness(
        pred_insights, gt_insights, goal, model_name
    )
    
    logic_score, logic_details = compute_logic(
        pred_insights, gt_insights, goal, model_name
    )
    
    insightfulness_score, insightfulness_details = compute_insightfulness(
        pred_insights, gt_insights, goal, model_name
    )
    
    # Compute weighted average
    weighted_avg = (
        WEIGHTS["factuality"] * factuality_score +
        WEIGHTS["completeness"] * completeness_score +
        WEIGHTS["logic"] * logic_score +
        WEIGHTS["insightfulness"] * insightfulness_score
    )
    
    return FourDimScores(
        factuality=factuality_score,
        factuality_hard=factuality_details["hard_score"],
        factuality_llm=factuality_details["llm_score"],
        completeness=completeness_score,
        logic=logic_score,
        insightfulness=insightfulness_score,
        weighted_avg=weighted_avg,
        details={
            "factuality": factuality_details,
            "completeness": completeness_details,
            "logic": logic_details,
            "insightfulness": insightfulness_details,
        }
    )
