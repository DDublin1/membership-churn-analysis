"""
Configuration and path constants for the Membership Churn Analysis Insight Agent.
Run this file directly to verify all paths and Gold table accessibility.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Project root ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
GOLD_DIR = DATA_DIR / "gold"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# ── Gold layer table paths ────────────────────────────────────────────
GOLD_TABLES = {
    "shap_segment_level":  GOLD_DIR / "shap_segment_level",
    "forecast_outputs":    GOLD_DIR / "forecast_outputs",
    "risk_segmentation":   GOLD_DIR / "risk_segmentation",
    "hazard_ratios":       GOLD_DIR / "hazard_ratios",
    "shap_importance":     GOLD_DIR / "shap_importance",
    "revenue_at_risk":     GOLD_DIR / "revenue_at_risk",
    "monthly_churn_trends": GOLD_DIR / "monthly_churn_trends",
    "churn_risk_summary":  GOLD_DIR / "churn_risk_summary",
    "region_churn_clean":  GOLD_DIR / "region_churn_clean",
    "predicted_retention":  GOLD_DIR / "predicted_retention",
    "validation_metrics":   GOLD_DIR / "validation_metrics",
    "model_metadata":       GOLD_DIR / "model_metadata",
}

# ── FAISS index paths ─────────────────────────────────────────────────
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss_index.bin"
FAISS_META_PATH = VECTOR_STORE_DIR / "faiss_meta.json"

# ── Embedding model ──────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Claude API configuration ─────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_TEMPERATURE = 0.0
CLAUDE_MAX_TOKENS_RESPONSE = 1000
CLAUDE_MAX_TOKENS_CLASSIFY = 10

# ── Forecast segments (exact strings from the data) ──────────────────
VALID_FORECAST_SEGMENTS = [
    "Nurse member (National)",
    "Nurse Support Worker (National)",
    "Student (National)",
    "Nurse member (Northern Ireland)",
    "Nurse member (Yorkshire & The Humber)",
    "Nurse member (Wales)",
    "Nurse member (North West)",
    "Nurse member (London)",
]

# ── Review log path ──────────────────────────────────────────────────
REVIEW_LOG_PATH = PROJECT_ROOT / "review_log.jsonl"


def verify():
    """Print all path constants and confirm all Gold tables are readable."""
    import pandas as pd

    print("=" * 60)
    print("the organisation Churn Agent — Configuration Verification")
    print("=" * 60)
    print()

    print(f"PROJECT_ROOT:     {PROJECT_ROOT}")
    print(f"DATA_DIR:         {DATA_DIR}")
    print(f"GOLD_DIR:         {GOLD_DIR}")
    print(f"VECTOR_STORE_DIR: {VECTOR_STORE_DIR}")
    print(f"FAISS_INDEX_PATH: {FAISS_INDEX_PATH}")
    print(f"FAISS_META_PATH:  {FAISS_META_PATH}")
    print(f"REVIEW_LOG_PATH:  {REVIEW_LOG_PATH}")
    print()

    print(f"CLAUDE_MODEL:      {CLAUDE_MODEL}")
    print(f"CLAUDE_TEMPERATURE: {CLAUDE_TEMPERATURE}")
    print(f"EMBEDDING_MODEL:   {EMBEDDING_MODEL}")
    api_status = "SET" if ANTHROPIC_API_KEY else "NOT SET"
    print(f"ANTHROPIC_API_KEY: {api_status}")
    print()

    print("Gold Layer Tables:")
    print("-" * 60)
    all_ok = True
    for name, path in GOLD_TABLES.items():
        try:
            df = pd.read_parquet(path)
            print(f"  OK  {name:<25} {len(df):>10,} rows  {len(df.columns):>3} cols")
        except Exception as e:
            print(f"  FAIL {name:<25} {e}")
            all_ok = False

    print()
    if all_ok:
        print("All Gold tables are readable.")
    else:
        print("WARNING: Some Gold tables could not be read.")

    return all_ok


if __name__ == "__main__":
    verify()
