"""
Forecast Agent — handles PROSPECTIVE queries.

Parses the user query to identify the target forecast segment, filters
the forecast_outputs Parquet for matching rows, computes trend direction,
formats point estimates and confidence intervals, and calls the Claude API
to generate a projection summary.

ADAPTATION NOTE (for dissertation reference):
─────────────────────────────────────────────
The actual forecast_outputs table differs from the PRD specification:
  - No 'split' column: all 96 rows are forward forecasts (Dec 2025 – Nov 2026)
  - Column mapping: ds (not snapshot_date), predicted (not churn_rate),
    ci_lower/ci_upper (not yhat_lower/yhat_upper)
  - Segment names use parenthetical format: "Nurse member (Wales)"
    rather than "Wales Nurse member"
Segment matching uses case-insensitive substring search against the 8
valid segment names, with Claude API fallback for ambiguous queries.
"""

import pandas as pd
import anthropic

from agent.prompts import SYSTEM_PROMPT, FORECAST_PROMPT
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    CLAUDE_MAX_TOKENS_RESPONSE,
    CLAUDE_MAX_TOKENS_CLASSIFY,
    VALID_FORECAST_SEGMENTS,
    GOLD_TABLES,
)

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _match_segment_from_query(query: str) -> str | None:
    """
    Case-insensitive substring match against the 8 valid forecast segments.

    Strategy: check each valid segment name against the query. If exactly
    one matches, return it. If multiple match, return the longest (most
    specific) match. If none match, return None for Claude fallback.
    """
    query_lower = query.lower()
    matches = []
    for seg in VALID_FORECAST_SEGMENTS:
        # Check if key parts of the segment name appear in the query.
        # "Nurse member (Wales)" -> check for "wales" and "nurse member"
        # "Student (National)" -> check for "student"
        seg_lower = seg.lower()

        # Extract the core identifiers from the segment name
        # Format is "Category (Region)" e.g. "Nurse member (Wales)"
        if "(" in seg:
            category = seg.split("(")[0].strip().lower()
            region = seg.split("(")[1].rstrip(")").strip().lower()
        else:
            category = seg_lower
            region = ""

        # Build a set of keyword variants for the category
        # to handle common shorthand (e.g. "nurses" for "nurse member")
        category_variants = [category]
        if "nurse member" in category:
            category_variants.extend(["nurse member", "nurse members", "nurses"])
        if "nurse support worker" in category:
            category_variants.extend(["nurse support worker", "nsw", "support worker"])
        if "student" in category:
            category_variants.extend(["student", "students"])

        cat_match = any(v in query_lower for v in category_variants)

        # Match if the query contains enough identifying terms
        if region == "national":
            # National segments: match on category alone (e.g. "student churn")
            if cat_match:
                matches.append(seg)
        else:
            # Regional segments: must match both category and region
            if cat_match and region in query_lower:
                matches.append(seg)

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        # Prefer regional (non-National) matches over National ones.
        # National segments match on category alone, so they often appear
        # as false positives alongside the correct regional match.
        regional = [m for m in matches if "(National)" not in m]
        if len(regional) == 1:
            return regional[0]
        elif len(regional) > 1:
            return max(regional, key=len)
        # All matches are national — return first
        return matches[0]
    return None


def _extract_segment_with_claude(query: str) -> str | None:
    """
    Fallback: use Claude to extract the target segment from the query.
    Returns the exact segment string or None if unresolvable.
    """
    segments_list = "\n".join(f"- {s}" for s in VALID_FORECAST_SEGMENTS)
    prompt = (
        f"The user asked: \"{query}\"\n\n"
        f"Which ONE of these forecast segments does the query refer to?\n"
        f"{segments_list}\n\n"
        f"Reply with ONLY the exact segment name from the list above, "
        f"or NONE if the query does not match any segment."
    )

    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=CLAUDE_MAX_TOKENS_CLASSIFY,
        temperature=CLAUDE_TEMPERATURE,
        system="You are a segment classifier. Reply with only the segment name or NONE.",
        messages=[{"role": "user", "content": prompt}],
    )

    result = response.content[0].text.strip()
    if result in VALID_FORECAST_SEGMENTS:
        return result
    return None


def _compute_trend(df: pd.DataFrame) -> str:
    """
    Compute trend direction by comparing mean of first 3 months
    to mean of last 3 months of the forecast period.
    """
    df_sorted = df.sort_values("ds")
    first_3 = df_sorted.head(3)["predicted"].mean()
    last_3 = df_sorted.tail(3)["predicted"].mean()

    diff = last_3 - first_3
    pct_change = (diff / first_3) * 100

    if abs(pct_change) < 1.0:
        return f"stable (change of {pct_change:+.2f}% between first and last 3-month averages)"
    elif diff > 0:
        return f"increasing ({pct_change:+.2f}% between first and last 3-month averages)"
    else:
        return f"decreasing ({pct_change:+.2f}% between first and last 3-month averages)"


def _format_forecast_table(df: pd.DataFrame) -> str:
    """Format forecast data as a readable text table for the prompt."""
    df_sorted = df.sort_values("ds")
    lines = ["Month | Predicted Rate | 95% CI Lower | 95% CI Upper | Method"]
    lines.append("-" * 75)
    for _, row in df_sorted.iterrows():
        month = pd.Timestamp(row["ds"]).strftime("%b %Y")
        lines.append(
            f"{month:<10} | {row['predicted']:.4%}       | {row['ci_lower']:.4%}       | "
            f"{row['ci_upper']:.4%}       | {row['method']}"
        )
    return "\n".join(lines)


class ForecastAgent:
    """Forecast agent for prospective churn projection queries."""

    def __init__(self):
        self._df = pd.read_parquet(GOLD_TABLES["forecast_outputs"])

    def match_segment(self, query: str) -> str | None:
        """
        Identify the target segment: case-insensitive keyword match first,
        then Claude API fallback.
        """
        segment = _match_segment_from_query(query)
        if segment is None:
            segment = _extract_segment_with_claude(query)
        return segment

    def get_forecast(self, segment: str) -> pd.DataFrame:
        """Filter forecast data for the matched segment."""
        return self._df[self._df["segment"] == segment].copy()

    def answer(self, query: str) -> dict:
        """End-to-end: match segment, retrieve forecast, generate response."""
        segment = self.match_segment(query)

        if segment is None:
            return {
                "response": (
                    "I could not identify a valid forecast segment from your query. "
                    f"The available segments are:\n"
                    + "\n".join(f"  - {s}" for s in VALID_FORECAST_SEGMENTS)
                ),
                "forecast_data": None,
                "segment": None,
                "query": query,
                "sources": [],
            }

        forecast_df = self.get_forecast(segment)

        if forecast_df.empty:
            return {
                "response": f"No forecast data found for segment '{segment}'.",
                "forecast_data": None,
                "segment": segment,
                "query": query,
                "sources": [],
            }

        trend = _compute_trend(forecast_df)
        forecast_table = _format_forecast_table(forecast_df)
        forecast_context = f"{forecast_table}\n\nTrend direction: {trend}"

        user_message = FORECAST_PROMPT.format(
            segment=segment,
            forecast_data=forecast_context,
            query=query,
        )

        response = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS_RESPONSE,
            temperature=CLAUDE_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Build sources list for review checkpoint compatibility
        sources = []
        for _, row in forecast_df.iterrows():
            month = pd.Timestamp(row["ds"]).strftime("%B %Y")
            sources.append({
                "source_table": "forecast_outputs",
                "summary": (
                    f"{segment}, {month}: predicted={row['predicted']:.4%}, "
                    f"CI=[{row['ci_lower']:.4%}, {row['ci_upper']:.4%}], "
                    f"method={row['method']}"
                ),
                "data": {
                    "ds": str(row["ds"]),
                    "predicted": float(row["predicted"]),
                    "ci_lower": float(row["ci_lower"]),
                    "ci_upper": float(row["ci_upper"]),
                    "segment": segment,
                    "method": row["method"],
                },
            })

        return {
            "response": response.content[0].text,
            "forecast_data": forecast_df.to_dict(orient="records"),
            "segment": segment,
            "query": query,
            "sources": sources,
        }


# ── Verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    agent = ForecastAgent()

    # Step 1: Test segment matching
    print("Segment Matching Tests")
    print("=" * 60)
    test_queries = [
        "What does the forecast show for Nurse member churn in Wales over the next 12 months?",
        "Will student churn increase next year?",
        "Projected churn for nurses in Northern Ireland?",
        "What about London nurse member forecast?",
    ]
    for q in test_queries:
        seg = _match_segment_from_query(q)
        print(f"  Query: \"{q[:70]}...\"")
        print(f"  Match: {seg}")
        print()

    # Step 2: Verification query
    print("=" * 60)
    print("Forecast Agent — Verification Query")
    print("=" * 60)

    query = "What does the forecast show for Nurse member churn in Wales over the next 12 months?"
    print(f"Query: {query}\n")

    result = agent.answer(query)
    print(f"Matched segment: {result['segment']}")
    print(f"Forecast rows: {len(result['sources'])}")
    print()
    print("Response:")
    print("-" * 60)
    print(result["response"])
