"""
Insight Agent — handles RETROSPECTIVE queries.

Embeds the user query using sentence-transformers, retrieves semantically
relevant segment summaries from the FAISS index, formats the retrieved
context into the Insight prompt, and calls the Claude API to generate a
structured response with citations.

RETRIEVAL STRATEGY:
The FAISS index contains 683 vectors from three Gold layer tables with
different analytical granularities (risk_segmentation: cross-sectional
profiles, monthly_churn_trends: time series, churn_risk_summary: executive
summaries with pre/post-surge uplift). A naive top-k search tends to
saturate results from whichever table has the most vectors closest to the
query embedding (usually monthly_churn_trends with 464 of 683 vectors).

To ensure the agent can draw on all three analytical dimensions, retrieval
uses a multi-source strategy: search with a wider k, then select the best
results per source table before merging into a final ranked set. This
guarantees that executive-level summaries (e.g. regional post-surge uplift
from churn_risk_summary) are not drowned out by individual monthly data
points, while still allowing the model to see granular evidence.
"""

import json

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from agent.prompts import SYSTEM_PROMPT, INSIGHT_PROMPT
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_TEMPERATURE,
    CLAUDE_MAX_TOKENS_RESPONSE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
)

# How many results to return per source table in multi-source retrieval,
# and the wide search multiplier to ensure all tables are represented.
_PER_SOURCE_K = 3
# Search the full 683-vector index to ensure churn_risk_summary rows
# (which have higher L2 distances due to shorter, denser summaries)
# are not excluded by a narrow search window.
_WIDE_SEARCH_K = 683


class InsightAgent:
    """Retrieval-augmented insight agent for retrospective churn queries."""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Multi-source retrieval: search broadly, then select top results
        per source table to ensure coverage across all analytical dimensions.

        Returns up to k results, drawing from each source table proportionally.
        """
        q_embedding = self._model.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self._index.search(q_embedding, _WIDE_SEARCH_K)

        # Group results by source table
        by_source: dict[str, list[dict]] = {}
        for dist, idx in zip(distances[0], indices[0]):
            entry = dict(self._metadata[idx])  # copy to avoid mutating cached metadata
            entry["distance"] = float(dist)
            source = entry["source_table"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(entry)

        # Allocation strategy:
        # - churn_risk_summary (31 rows): include ALL entries — these are short
        #   executive summaries spanning every dimension (Region, Category,
        #   Age Band, Sector, Cross-Segment) with pre/post-surge uplift.
        #   Including all 31 ensures the agent can answer any cross-dimensional
        #   question without losing coverage to distance ranking.
        # - risk_segmentation & monthly_churn_trends: take top _PER_SOURCE_K
        #   to provide granular supporting evidence without flooding context.
        selected = []
        for source, entries in by_source.items():
            if source == "churn_risk_summary":
                selected.extend(entries)  # all 31 rows
            else:
                selected.extend(entries[:_PER_SOURCE_K])

        # Sort by distance and trim to final k
        selected.sort(key=lambda x: x["distance"])
        return selected[:k]

    def generate(self, query: str, retrieved: list[dict]) -> dict:
        """Format retrieved context and call Claude API for a response."""
        context_lines = []
        for i, entry in enumerate(retrieved, 1):
            context_lines.append(f"[{i}] (source: {entry['source_table']}) {entry['summary']}")

        retrieved_context = "\n".join(context_lines)
        user_message = INSIGHT_PROMPT.format(
            retrieved_context=retrieved_context,
            query=query,
        )

        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS_RESPONSE,
            temperature=CLAUDE_TEMPERATURE,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return {
            "response": response.content[0].text,
            "sources": retrieved,
            "query": query,
        }

    def answer(self, query: str, k: int = 37) -> dict:
        """End-to-end: retrieve context, generate response."""
        retrieved = self.retrieve(query, k=k)
        return self.generate(query, retrieved)


# ── Verification ──────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = InsightAgent()

    query = (
        "Which regions have seen the biggest increase in Nurse member "
        "churn since the 2022 recruitment surge ended?"
    )

    print("Insight Agent — Verification Query")
    print("=" * 60)
    print(f"Query: {query}")
    print()

    # Show what was retrieved (k=15 to get all churn_risk_summary + top from others)
    retrieved = agent.retrieve(query, k=37)
    print(f"Retrieved {len(retrieved)} segments across source tables:")
    for i, entry in enumerate(retrieved, 1):
        print(f"  [{i}] dist={entry['distance']:.4f} | {entry['source_table']}")
        print(f"      {entry['summary'][:140]}...")
    print()

    # Generate response
    result = agent.generate(query, retrieved)
    print("Generated response:")
    print("-" * 60)
    print(result["response"])
