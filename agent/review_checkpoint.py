"""
Human Review Checkpoint — mandatory review gate for all agent responses.

MANDATORY: Every response passes through this checkpoint before being
marked as final. Auto-approval is NEVER acceptable under any circumstances.

Displays the original query, the generated response, and the source
segments used. Prompts for y (approve), n (reject), or e (edit).
Logs every decision to review_log.jsonl with timestamp.
"""

import json
from datetime import datetime, timezone

from config import REVIEW_LOG_PATH


def _log_decision(entry: dict) -> None:
    """Append a review decision to the JSONL log file."""
    with open(REVIEW_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _display_for_review(query: str, response: str, sources: list[dict]) -> None:
    """Display the query, response, and sources for human review."""
    print("\n" + "=" * 70)
    print("HUMAN REVIEW CHECKPOINT")
    print("=" * 70)

    print("\n--- ORIGINAL QUERY ---")
    print(query)

    print("\n--- GENERATED RESPONSE ---")
    print(response)

    print("\n--- SOURCE DATA USED ---")
    print(f"({len(sources)} segment(s) retrieved)")
    for i, src in enumerate(sources, 1):
        table = src.get("source_table", "unknown")
        summary = src.get("summary", "")
        dist = src.get("distance", None)
        dist_str = f" [dist={dist:.4f}]" if dist is not None else ""
        print(f"  [{i}] ({table}){dist_str}")
        # Show first 120 chars of summary to keep display manageable
        if len(summary) > 120:
            print(f"      {summary[:120]}...")
        else:
            print(f"      {summary}")

    print("\n" + "-" * 70)


def _prompt_decision() -> tuple[str, str]:
    """
    Prompt for y/n/e and loop until valid input is received.
    Returns (decision, reviewer_notes).
    """
    while True:
        choice = input("Approve this response? (y/n/e for edit): ").strip().lower()

        if choice == "y":
            return "approved", ""

        elif choice == "n":
            notes = input("Reason for rejection (optional): ").strip()
            return "rejected", notes

        elif choice == "e":
            print("Enter your corrected response. Type END on a line by itself to finish.")
            lines = []
            while True:
                line = input()
                if line.strip() == "END":
                    break
                lines.append(line)
            edited_response = "\n".join(lines)
            return "edited", edited_response

        else:
            print(f"Invalid input: '{choice}'. Please enter y, n, or e.")


def review(agent_result: dict) -> dict:
    """
    Run the human review checkpoint on an agent result.

    Parameters
    ----------
    agent_result : dict
        Must contain keys: "query", "response", "sources".
        May also contain "forecast_data", "segment", "classification".

    Returns
    -------
    dict with keys:
        - approved: bool — True if approved or edited, False if rejected
        - response: str — the final response (original or edited)
        - decision: str — "approved", "rejected", or "edited"
        - reviewer_notes: str — rejection reason or edited response
    """
    query = agent_result["query"]
    response = agent_result["response"]
    sources = agent_result.get("sources", [])

    _display_for_review(query, response, sources)
    decision, notes = _prompt_decision()

    # Determine final response
    if decision == "edited":
        final_response = notes  # the edited text
    else:
        final_response = response

    # Build log entry
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "response": final_response,
        "sources_count": len(sources),
        "decision": decision,
        "reviewer_notes": notes if decision != "edited" else "(see response for edited text)",
    }
    _log_decision(log_entry)

    approved = decision in ("approved", "edited")

    if approved:
        print(f"\nResponse {decision.upper()} and logged.")
    else:
        print(f"\nResponse REJECTED and logged.")

    return {
        "approved": approved,
        "response": final_response,
        "decision": decision,
        "reviewer_notes": notes,
    }


# ── Verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulated agent result for testing
    test_result = {
        "query": "What is the churn rate for Nurse members in Northern Ireland?",
        "response": (
            "The churn rate for Nurse members in Northern Ireland is 0.4998% "
            "per month (overall). Pre-surge: 0.4298%, post-surge: 0.5438%, "
            "representing a +26.51% uplift. Intervention priority: High."
        ),
        "sources": [
            {
                "source_table": "churn_risk_summary",
                "summary": "Region dimension: Northern Ireland. Overall monthly churn rate: 0.4998%.",
                "distance": 0.9864,
                "data": {"Dimension": "Region", "Segment": "Northern Ireland"},
            },
            {
                "source_table": "monthly_churn_trends",
                "summary": "Nurse member (Northern Ireland) segment, Nov 2025. Churn rate: 0.4988%.",
                "distance": 0.3556,
                "data": {"segment": "Nurse member (Northern Ireland)"},
            },
        ],
    }

    print("Review Checkpoint — Interactive Test")
    print("Submit 'y', 'n', or 'e' when prompted.\n")
    result = review(test_result)
    print(f"\nResult: {result}")
