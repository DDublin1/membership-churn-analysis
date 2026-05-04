"""
Membership Churn Analysis Insight Agent — Interactive Entry Point.

Wires the manager agent to the correct worker agent (Insight or Forecast)
to the human review checkpoint in a single call stack. Includes out-of-scope
detection for queries that reference individual-level data (which does not
exist in the system) and graceful error handling so no unhandled exceptions
reach the user during the interactive loop.

Usage:
    python run_agent.py                  # interactive REPL
    python run_agent.py --query "..."    # single query mode
"""

import argparse
import re
import sys
import io
import traceback

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", errors="replace")

from agent.manager import classify_query
from agent.insight_agent import InsightAgent
from agent.forecast_agent import ForecastAgent
from agent.review_checkpoint import review


# ── Out-of-scope detection ────────────────────────────────────────────
# The system contains only aggregate population-level statistics.
# Individual member records do not exist anywhere in the pipeline.
_OUT_OF_SCOPE_PATTERN = re.compile(
    r"\b(individual|personal|name|record|identify|identified|"
    r"identifiable|patient|employee|staff member|specific person)\b",
    re.IGNORECASE,
)

_OUT_OF_SCOPE_RESPONSE = (
    "This query appears to reference individual-level data. "
    "The churn analysis system operates exclusively on aggregate "
    "population-level statistics. No individual member records, "
    "names, or personally identifiable information exist anywhere "
    "in the system. This is an architectural constraint — the data "
    "was never ingested. Please rephrase your query in terms of "
    "membership categories, regions, age bands, or segments."
)


def _is_out_of_scope(query: str) -> bool:
    """Check if the query references individual-level data."""
    return bool(_OUT_OF_SCOPE_PATTERN.search(query))


# ── Agent pipeline ────────────────────────────────────────────────────

# Lazy-initialised singletons to avoid loading models until first query
_insight_agent: InsightAgent | None = None
_forecast_agent: ForecastAgent | None = None


def _get_insight_agent() -> InsightAgent:
    global _insight_agent
    if _insight_agent is None:
        print("Loading Insight Agent (FAISS index + embedding model)...")
        _insight_agent = InsightAgent()
    return _insight_agent


def _get_forecast_agent() -> ForecastAgent:
    global _forecast_agent
    if _forecast_agent is None:
        print("Loading Forecast Agent (forecast data)...")
        _forecast_agent = ForecastAgent()
    return _forecast_agent


def process_query(query: str) -> dict:
    """
    Full pipeline: classify -> route to worker -> review checkpoint.

    Returns the review result dict with keys:
        approved, response, decision, reviewer_notes
    """
    # Step 1: Out-of-scope check
    if _is_out_of_scope(query):
        print(f"\n[OUT OF SCOPE] {_OUT_OF_SCOPE_RESPONSE}")
        return {
            "approved": False,
            "response": _OUT_OF_SCOPE_RESPONSE,
            "decision": "out_of_scope",
            "reviewer_notes": "",
        }

    # Step 2: Manager classification
    print("\nClassifying query...")
    classification = classify_query(query)
    print(f"Classification: {classification}")

    # Step 3: Route to worker agent
    if classification == "RETROSPECTIVE":
        agent = _get_insight_agent()
        print("Retrieving relevant segments from FAISS index...")
        agent_result = agent.answer(query)
    elif classification == "PROSPECTIVE":
        agent = _get_forecast_agent()
        print("Matching forecast segment...")
        agent_result = agent.answer(query)
        if agent_result["segment"]:
            print(f"Matched segment: {agent_result['segment']}")
        else:
            print("No matching forecast segment found.")
    else:
        raise ValueError(f"Unexpected classification: {classification}")

    # Step 4: Human review checkpoint (MANDATORY)
    review_result = review(agent_result)

    return review_result


# ── Interactive loop ──────────────────────────────────────────────────

def run_interactive():
    """Interactive REPL with graceful error handling."""
    print("=" * 70)
    print("Membership Churn Analysis Insight Agent")
    print("=" * 70)
    print("Ask questions about membership churn patterns.")
    print("Type 'quit' or 'exit' to end the session.")
    print("=" * 70)

    while True:
        try:
            print()
            query = input("Query> ").strip()

            if not query:
                continue
            if query.lower() in ("quit", "exit", "q"):
                print("Session ended.")
                break

            result = process_query(query)

            if result["decision"] == "out_of_scope":
                continue

            if result["approved"]:
                print("\n[FINAL APPROVED RESPONSE]")
                print(result["response"])
            else:
                print("\n[RESPONSE REJECTED — not forwarded to stakeholder]")

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing your query:")
            print(f"  {type(e).__name__}: {e}")
            print("Please try rephrasing your query or check the system configuration.")
            # Log full traceback to stderr for debugging
            traceback.print_exc(file=sys.stderr)


def run_single(query: str):
    """Single query mode for scripted execution."""
    result = process_query(query)
    if result["approved"]:
        print("\n[FINAL APPROVED RESPONSE]")
        print(result["response"])
    else:
        print("\n[RESPONSE REJECTED — not forwarded to stakeholder]")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Membership Churn Analysis Insight Agent"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to process (non-interactive mode)"
    )
    args = parser.parse_args()

    if args.query:
        run_single(args.query)
    else:
        run_interactive()
