import pytest
import json
from src.evaluation.faithfulness_judge import grade_faithfulness
from advanced_rag_ops_langsmith import run_with_cache_and_eval as run_rag_pipeline

def load_cases():
    with open("tests/eval_dataset.json", "r") as f:
        return json.load(f)

@pytest.mark.parametrize("case", load_cases())
def test_rag_faithfulness(case):
    # 1. Run pipeline (now returns 4 items)
    answer, tokens, internal_verdict, source_context = run_rag_pipeline(case['question'])
    
    # 2. Get the Judge's verdict using REAL context
    score, reason = grade_faithfulness(case['question'], source_context, answer)
    
    print(f"\n--- FDE Audit Result ---")
    print(f"Question: {case['question']}")
    print(f"Faithfulness Score: {score}")
    print(f"Judge's Reasoning: {reason}")
    print(f"------------------------\n")
    
    # 3. The Gate
    assert score >= case['min_faithfulness'], f"Regression! {reason}"