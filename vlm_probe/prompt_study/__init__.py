"""Prompt sensitivity and multi-prompt ensembling suite (P3.1-P3.4)."""

from .design import build_dataset_templates, build_counting_prompts, build_negation_prompts
from .metrics import (
    text_embedding_ensemble,
    logit_average_ensemble,
    probability_average_ensemble,
    summarize_template_suite,
)
from .offline_analysis import analyze_saved_probing_ensembles

__all__ = [
    "build_dataset_templates",
    "build_counting_prompts",
    "build_negation_prompts",
    "text_embedding_ensemble",
    "logit_average_ensemble",
    "probability_average_ensemble",
    "summarize_template_suite",
    "analyze_saved_probing_ensembles",
]

