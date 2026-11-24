"""
Dataset and Evaluation Question loaders.
"""

import json
from pathlib import Path
from typing import List, Tuple
from core.evaluation import EvalQuestion

DEFAULT_DOC_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_document.txt"
DEFAULT_EVAL_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_evaluation_set.json"


def load_demo_document() -> str:
    """Load the built-in reproducible demo document text."""
    if DEFAULT_DOC_PATH.exists():
        with open(DEFAULT_DOC_PATH, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def load_evaluation_questions(json_source: str = "") -> List[EvalQuestion]:
    """Load evaluation questions from a JSON string or the default file path."""
    if json_source and json_source.strip():
        data = json.loads(json_source)
    elif DEFAULT_EVAL_PATH.exists():
        with open(DEFAULT_EVAL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        return []

    questions = []
    for item in data:
        questions.append(
            EvalQuestion(
                id=item.get("id", f"q{len(questions)+1}"),
                question=item.get("question", ""),
                ground_truth_answers=item.get("ground_truth_answers", []),
                relevant_keywords=item.get("relevant_keywords", []),
                expected_section=item.get("expected_section"),
            )
        )
    return questions
