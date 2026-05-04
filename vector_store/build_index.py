"""
FAISS Vector Index Builder for the Membership Churn Analysis Insight Agent.

ADAPTATION NOTE (for dissertation reference):
─────────────────────────────────────────────
The original PRD specified a single source table (shap_segment_level) with
157,546 segment-month rows containing MemCategory, Region, age_band,
tenure_band, MemSectorType, period, CM_snapshot_date, churn_rate, leavers,
members, and SHAP value columns. The actual shap_segment_level table
produced by the Databricks pipeline contains 64 rows of SHAP feature
importance by segment/covariate — a different analytical artifact.

This adapted implementation constructs the FAISS index from three Gold
layer tables that collectively provide the segment-level churn insights
the agent needs to answer retrospective queries:

  1. risk_segmentation (188 rows)
     Cross-sectional risk profile: Region × MemCategory × age_band with
     churn_rate, risk_tier, impact_score, and estimated annual leavers.

  2. monthly_churn_trends (464 rows)
     Time-series churn rates per segment per month (Jan 2021 – Nov 2025),
     with total leavers and members per snapshot.

  3. churn_risk_summary (31 rows)
     Executive-level summary of churn by dimension (Region, Category,
     Age Band, Sector, Cross-Segment) with pre/post-surge uplift
     percentages and intervention priority ratings.

Total index vectors: 683 (188 + 464 + 31), no deduplication required as
the three tables have entirely distinct schemas and granularity levels.

Each row is converted to a natural language summary string, embedded using
the all-MiniLM-L6-v2 sentence transformer (384-dimensional vectors), and
stored in a flat FAISS L2 index. A companion JSON metadata file maps each
vector position to its source table, row index, and original data fields
for citation in agent responses.
"""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    GOLD_TABLES,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
)


def build_risk_segmentation_summaries(df: pd.DataFrame) -> list[dict]:
    """
    risk_segmentation: 188 rows.
    Region × MemCategory × age_band cross-sectional risk profiles.
    """
    records = []
    for _, row in df.iterrows():
        summary = (
            f"{row['MemCategory']} member in {row['Region']}, "
            f"age band {row['age_band']}. "
            f"Churn rate: {row['churn_rate']:.4%}. "
            f"Risk tier: {row['risk_tier']}. "
            f"Impact score: {row['impact_score']:.2f}. "
            f"Total leavers: {row['total_leavers']:,.0f} over {row['n_months']} months. "
            f"Average monthly members: {row['avg_monthly_members']:,.0f}. "
            f"Estimated annual leavers: {row['est_annual_leavers']:,.0f}."
        )
        records.append({
            "source_table": "risk_segmentation",
            "summary": summary,
            "data": {
                "Region": row["Region"],
                "MemCategory": row["MemCategory"],
                "age_band": row["age_band"],
                "churn_rate": float(row["churn_rate"]),
                "risk_tier": row["risk_tier"],
                "impact_score": float(row["impact_score"]),
                "total_leavers": int(row["total_leavers"]),
                "total_members": int(row["total_members"]),
                "n_months": int(row["n_months"]),
                "avg_monthly_members": float(row["avg_monthly_members"]),
                "est_annual_leavers": float(row["est_annual_leavers"]),
            },
        })
    return records


def build_monthly_churn_summaries(df: pd.DataFrame) -> list[dict]:
    """
    monthly_churn_trends: 464 rows.
    Time-series churn rate per segment per month.
    """
    records = []
    for _, row in df.iterrows():
        date_str = pd.Timestamp(row["snapshot_date"]).strftime("%B %Y")
        summary = (
            f"{row['segment']} segment, {date_str} ({row['split']} period). "
            f"Monthly churn rate: {row['churn_rate']:.4%}. "
            f"Total leavers: {row['total_leavers']:,.0f}. "
            f"Total members: {row['total_members']:,.0f}."
        )
        records.append({
            "source_table": "monthly_churn_trends",
            "summary": summary,
            "data": {
                "snapshot_date": str(row["snapshot_date"]),
                "segment": row["segment"],
                "churn_rate": float(row["churn_rate"]),
                "total_leavers": float(row["total_leavers"]),
                "total_members": float(row["total_members"]),
                "split": row["split"],
            },
        })
    return records


def build_churn_risk_summaries(df: pd.DataFrame) -> list[dict]:
    """
    churn_risk_summary: 31 rows.
    Executive summary by dimension with pre/post-surge uplift.
    """
    records = []
    for _, row in df.iterrows():
        # NOTE: churn_risk_summary values are already in percentage form
        # (e.g. 0.5100 means 0.51%), unlike risk_segmentation/monthly_churn_trends
        # where values are fractions (e.g. 0.0051 means 0.51%). Do NOT use :.4%
        # which would multiply by 100 again.
        parts = [
            f"{row['Dimension']} dimension: {row['Segment']}. "
            f"Overall monthly churn rate: {row['Overall_Churn_Pct']:.4f}%."
        ]
        if pd.notna(row["PreSurge_Churn_Pct"]) and pd.notna(row["PostSurge_Churn_Pct"]):
            parts.append(
                f" Pre-surge churn rate: {row['PreSurge_Churn_Pct']:.4f}%."
                f" Post-surge churn rate: {row['PostSurge_Churn_Pct']:.4f}%."
                f" Post-surge uplift: {row['PostSurge_Uplift_Pct']:+.2f}%."
            )
        parts.append(
            f" Risk type: {row['Risk_Type']}."
            f" Intervention priority: {row['Intervention_Priority']}."
        )
        summary = "".join(parts)

        data = {
            "Dimension": row["Dimension"],
            "Segment": row["Segment"],
            "Overall_Churn_Pct": float(row["Overall_Churn_Pct"]),
            "Risk_Type": row["Risk_Type"],
            "Intervention_Priority": row["Intervention_Priority"],
        }
        if pd.notna(row["PreSurge_Churn_Pct"]):
            data["PreSurge_Churn_Pct"] = float(row["PreSurge_Churn_Pct"])
        if pd.notna(row["PostSurge_Churn_Pct"]):
            data["PostSurge_Churn_Pct"] = float(row["PostSurge_Churn_Pct"])
        if pd.notna(row["PostSurge_Uplift_Pct"]):
            data["PostSurge_Uplift_Pct"] = float(row["PostSurge_Uplift_Pct"])

        records.append({
            "source_table": "churn_risk_summary",
            "summary": summary,
            "data": data,
        })
    return records


def build_index():
    """Build the FAISS index and metadata from three Gold layer tables."""
    print("Loading Gold layer tables...")
    df_risk = pd.read_parquet(GOLD_TABLES["risk_segmentation"])
    df_monthly = pd.read_parquet(GOLD_TABLES["monthly_churn_trends"])
    df_summary = pd.read_parquet(GOLD_TABLES["churn_risk_summary"])
    print(f"  risk_segmentation:   {len(df_risk):>5} rows")
    print(f"  monthly_churn_trends: {len(df_monthly):>4} rows")
    print(f"  churn_risk_summary:  {len(df_summary):>5} rows")

    print("\nGenerating natural language summaries...")
    records = []
    records.extend(build_risk_segmentation_summaries(df_risk))
    records.extend(build_monthly_churn_summaries(df_monthly))
    records.extend(build_churn_risk_summaries(df_summary))
    print(f"  Total summaries: {len(records)}")

    summaries = [r["summary"] for r in records]
    metadata = [{"index": i, "source_table": r["source_table"], "summary": r["summary"], "data": r["data"]}
                for i, r in enumerate(records)]

    print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Embedding summaries...")
    embeddings = model.encode(summaries, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype("float32")
    print(f"  Embedding shape: {embeddings.shape}")

    print("Building FAISS flat L2 index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"  index.ntotal: {index.ntotal}")

    print(f"\nSaving index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    print(f"Saving metadata to {FAISS_META_PATH}...")
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nDone. Index contains {index.ntotal} vectors of dimension {dimension}.")
    return index, metadata, model


def verify_query(index, metadata, model, query: str, k: int = 5):
    """Run a test query against the index and print results."""
    print(f"\nVerification query: \"{query}\"")
    print("-" * 60)
    q_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_embedding, k)
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        entry = metadata[idx]
        print(f"\n  [{rank}] distance={dist:.4f} | source={entry['source_table']}")
        print(f"      {entry['summary'][:120]}...")
    return indices[0]


if __name__ == "__main__":
    index, metadata, model = build_index()

    # Verification: adapted from PRD — query about Northern Ireland post-surge
    verify_query(
        index, metadata, model,
        "Nurse member Northern Ireland post-surge churn rate"
    )

    # Second verification: time-series query
    verify_query(
        index, metadata, model,
        "Student national churn rate January 2023"
    )
