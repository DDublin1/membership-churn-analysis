"""
Manager Agent — query classification and routing.

Receives a raw user query, classifies it as RETROSPECTIVE or PROSPECTIVE
using Claude API, and routes to the appropriate worker agent.
The manager never generates analytical content itself.
"""

import anthropic

from agent.prompts import SYSTEM_PROMPT, MANAGER_PROMPT
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    CLAUDE_MAX_TOKENS_CLASSIFY,
)


_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def classify_query(query: str) -> str:
    """
    Classify a user query as RETROSPECTIVE or PROSPECTIVE.

    Uses the Claude API with the manager classification prompt.
    Returns exactly one of: "RETROSPECTIVE", "PROSPECTIVE".
    Raises ValueError if the model returns an unexpected response.
    """
    user_message = MANAGER_PROMPT.format(query=query)

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS_CLASSIFY,
        temperature=CLAUDE_TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    classification = response.content[0].text.strip().upper()

    if classification not in ("RETROSPECTIVE", "PROSPECTIVE"):
        raise ValueError(
            f"Manager agent returned unexpected classification: '{classification}'. "
            f"Expected RETROSPECTIVE or PROSPECTIVE."
        )

    return classification


def route_query(query: str) -> dict:
    """
    Classify and route a query to the appropriate worker agent.

    Returns a dict with:
      - query: the original query string
      - classification: RETROSPECTIVE or PROSPECTIVE
    """
    classification = classify_query(query)
    return {"query": query, "classification": classification}


# ── Verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        # 3 RETROSPECTIVE
        ("What is the churn rate for Nurse members in Northern Ireland?", "RETROSPECTIVE"),
        ("Which age band has the highest post-surge uplift?", "RETROSPECTIVE"),
        ("What are the top SHAP features driving churn?", "RETROSPECTIVE"),
        # 2 PROSPECTIVE
        ("What is the projected churn for Students over the next 12 months?", "PROSPECTIVE"),
        ("Will Nurse member churn in Wales increase or decrease by mid-2026?", "PROSPECTIVE"),
    ]

    print("Manager Agent — 5-Query Verification Test")
    print("=" * 60)

    all_pass = True
    for query, expected in test_queries:
        result = route_query(query)
        status = "PASS" if result["classification"] == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] {result['classification']:<15} (expected {expected})")
        print(f"         \"{query}\"")

    print()
    if all_pass:
        print("All 5 queries classified correctly.")
    else:
        print("WARNING: Some queries were misclassified.")
