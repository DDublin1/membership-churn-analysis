# Membership Churn Analysis

**End-to-end churn prediction and AI-powered insight generation on 18.4M membership records**

A production-ready data science pipeline combining scalable PySpark processing, statistical hypothesis testing, survival analysis, machine learning classification, and multi-agent LLM systems to predict and explain membership churn at scale.


> **Data Confidentiality Notice**
> The dataset used in this project is subject to a Non-Disclosure Agreement (NDA) and has been removed from this repository. The `data/` directory contains schema documentation and a data card describing the dataset's structure, scale, and statistical properties without exposing confidential records. All methodology, code, model architecture, and agentic system design are fully documented and reproducible given equivalent data.

## Overview

This repository contains a complete analytical workflow for membership churn analysis built on 2.1GB of membership data (18.4M records). The pipeline progresses from raw data ingestion through exploratory analysis, statistical validation, predictive modeling, and finally AI-powered insight generation via a specialized multi-agent system.

**Key capabilities:**
- Scalable data processing with Apache Spark (PySpark) handling 18.4M+ records
- Comprehensive data quality validation and missingness analysis
- Statistical hypothesis testing to identify significant churn drivers
- Survival analysis using actuarial methods (Kaplan-Meier, Cox proportional hazards)
- Predictive modeling with multiple algorithms (XGBoost, Random Forest, Logistic Regression)
- SHAP-based feature importance and model explainability
- Churn segmentation and behavioral profiling
- Multi-agent AI system for prospective and retrospective churn insights
- Interactive stakeholder dashboard for reporting

## Architecture

```
membership-churn-analysis/
├── notebooks/              # 10 Jupyter notebooks progressing from data to insights
├── agent/                  # Multi-agent churn insight generation system
├── vector_store/           # FAISS index builder for LLM context retrieval
├── dashboard/              # Interactive HTML dashboard for stakeholders
├── data/                   # (not included) Dataset location placeholder
├── config.py               # Agent configuration and settings
├── run_agent.py            # Entry point for churn agent pipeline
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Notebooks

The analysis is structured in 10 sequential Jupyter notebooks:

1. **01_data_ingestion.ipynb** - Environment setup, library imports, Spark session initialization, and raw data loading from CSV
2. **02_data_cleaning.ipynb** - Data transformation, standardization, null handling, and feature engineering
3. **03_data_quality.ipynb** - Missingness analysis, data validation, duplicates detection, and quality metrics
4. **04_eda_part1.ipynb** - Univariate analysis, distributions, summary statistics, and initial patterns
5. **05_eda_part2.ipynb** - Bivariate analysis, correlations, churn patterns, and behavioral segmentation
6. **06_churn_metrics.ipynb** - Churn rate calculations, cohort analysis, retention curves, and segment profiling
7. **07_statistical_analysis.ipynb** - Hypothesis testing (chi-square, t-tests), effect sizes, and statistical validation
8. **08_predictive_modelling.ipynb** - Feature engineering, model training (XGBoost, Random Forest, Logistic Regression), evaluation, and SHAP analysis
9. **09_visualisation_portfolio.ipynb** - Publication-quality visualizations for dashboards and reports
10. **11_project_summary.ipynb** - Executive summary, key findings, methodology documentation, and recommendations

## AI Agent System

The `agent/` module provides a multi-agent system for automated churn insight generation:

- **forecast_agent.py** - Generates prospective insights predicting future churn patterns
- **insight_agent.py** - Performs retrospective analysis explaining historical churn
- **manager.py** - Orchestrates agent workflows and coordinates execution
- **prompts.py** - LLM system prompts and instruction templates
- **review_checkpoint.py** - Validation and checkpoint management
- **__init__.py** - Package initialization

### Running the Agent

```bash
# Configure your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the churn analysis agent
python run_agent.py
```

The agent uses FAISS vector search (`vector_store/build_index.py`) to retrieve relevant analytical context from completed notebooks before generating insights via OpenAI's API.

## Dashboard

An interactive HTML dashboard (`dashboard/index.html`) provides stakeholder-facing visualizations including:
- Churn trends and KPIs
- Segment performance
- Model prediction results
- Key insights and recommendations

**Usage:** Open `dashboard/index.html` in any web browser.

## Tech Stack

- **Data Processing**: PySpark, pandas, NumPy
- **Statistical Analysis**: SciPy, statsmodels, lifelines (survival analysis)
- **Machine Learning**: scikit-learn, XGBoost
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **LLM Integration**: OpenAI API, python-dotenv
- **Visualization**: Matplotlib, Seaborn
- **Notebooks**: Jupyter

## Setup

### Requirements

- Python 3.8+
- Java (for Spark)
- OpenAI API key (for agent system)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/membership-churn-analysis.git
   cd membership-churn-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare data**
   - Place your membership dataset at `data/churn_t_db.csv` (see `data/README.md` for format details)
   - Ensure at least 2.1GB disk space and sufficient RAM for Spark processing

5. **Configure agent (optional)**
   - Copy `.env.example` to `.env` and add your OpenAI API key
   - Customize `config.py` for your environment

### Running Notebooks

Start Jupyter and navigate to the `notebooks/` directory:

```bash
jupyter notebook
```

Run notebooks in sequence (01 → 11) for the complete pipeline, or jump to specific notebooks for targeted analysis.

### Building Vector Index

Pre-build the FAISS index for faster agent startup:

```bash
python vector_store/build_index.py
```

## Data

The dataset is not included in this repository due to size constraints (2.1GB, 18.4M rows). Expected format:
- **File**: `data/churn_t_db.csv`
- **Rows**: 18.4 million membership records
- **Columns**: Membership metadata, behavioral features, temporal indicators, churn labels
- **Format**: CSV with headers

See individual notebook cells for specific column requirements and schema definitions.

## Key Findings (Example Structure)

Your analysis will reveal:
- Churn rate by segment and temporal cohort
- Statistical drivers of churn (confirmed via hypothesis tests)
- Predictive model performance and feature importance
- Survival curves and retention dynamics
- Automated AI-generated insights from agent system

## Future Enhancements

- Real-time churn prediction pipeline
- MLOps deployment (model monitoring, retraining)
- Advanced segmentation (clustering, RFM analysis)
- Causal inference analysis
- Prophet-based time-series forecasting

## License

MIT License - see LICENSE file for details

## Contact

For questions or contributions, please open an issue or contact the repository maintainer.

---

**Built with**: PySpark · scikit-learn · SHAP · OpenAI · FAISS
