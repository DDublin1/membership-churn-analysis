# Dataset Documentation

## Proprietary Membership Churn Dataset

> **Confidentiality Notice**
> This dataset is proprietary and subject to a Non-Disclosure Agreement (NDA).
> The data, the organisation it originates from, and any identifying characteristics
> of the data source are strictly confidential. The dataset is **not included**
> in this repository and cannot be shared, redistributed, or made available
> in any form. All analysis in this project was conducted under formal data
> access agreement. Any reproduction or distribution of this dataset is
> prohibited without explicit written authorisation from the data owner.

---

## Overview

| Attribute | Detail |
|-----------|--------|
| **Type** | Aggregated membership panel — monthly snapshots |
| **Scale** | 18,461,480 records (60 monthly snapshots) |
| **File size** | 2.06 GB (CSV) |
| **Time span** | January 2021 — December 2025 (5 years) |
| **Granularity** | Member cohort segments (aggregated) |
| **Access** | Restricted — NDA required |

---

## Dataset Structure

The dataset represents **aggregated membership data**, not raw member-level records. Each row represents a membership cohort segment defined by a specific combination of temporal, geographic, categorical, and demographic dimensions. The value in `q_members_t` represents the aggregated membership count for that cohort, not a single member record.

**Observation Period:** 60 consecutive monthly snapshots from 2021-01-01 to 2025-12-01, with complete coverage (no missing months).

**Data Origin:** Snapshot-based panel structure capturing membership state at each monthly interval, suitable for longitudinal trend analysis and cohort-based research.

---

## Dataset Schema

### Raw Data Columns (12 original columns from CSV)

| Column Name | Data Type | Nulls | Cardinality | Description |
|-------------|-----------|-------|-------------|-------------|
| `_c0` | integer | 0 | 18,461,480 | Sequential row identifier (0 to 18,461,479) |
| `CM_snapshot_date` | date | 0 | 60 | Monthly snapshot date (1st of each month, 2021-01-01 to 2025-12-01) |
| `Int_nurse` | string | 0 | 4 | International classification (UK, Other, Overseas, EEU) |
| `Region` | string | 1 | 14 | Geographic region (13 UK regions + H Q (Overseas); 1 null replaced with "Unknown") |
| `MemCategory` | string | 0 | 4 | Membership category (Student, Practising, Honorary, Other) |
| `CatName` | string | 0 | 102 | Category detail/name field |
| `Branch` | string | 0 | 104 | Local branch identifier (sub-unit of region) |
| `YoB` | double* | 4,406 | 105 | Year of birth (originally double, cast to integer in cleaning) |
| `MemSectorType` | string | 1,436,695 | 5 | Employment sector (NHS, Independent, Education, Other Public Sector, null) |
| `YoJ` | double* | 39 | 85 | Year of join (originally double, cast to integer in cleaning) |
| `q_members_t` | double | 0 | 99 | Aggregated membership count for cohort (base value: 2.97, range: 2.97–296.91) |
| `q_leavers_t` | double | 18,259,005 | 11 | Aggregated leaver count for cohort (nulls expected, 98.90%; range: 2.97–50.48 when present) |

*Note: `YoB` and `YoJ` are stored as double in raw CSV due to Spark schema inference, but contain only integer values. Cleaning notebook (02) casts these to integer type in the Silver table.

---

### Processed Data Columns (14 columns in Silver table after Notebook 02)

After data cleaning (Notebook 02), two additional columns are added:

| Column Name | Data Type | Purpose | Description |
|-------------|-----------|---------|-------------|
| `cleaning_flag` | string | Data quality flagging | Signals data quality issues: `YOB_CONFIRMED_ERROR`, `YOB_YOUNG_MEMBER`, `YOB_DEFAULT_PLACEHOLDER`, `YOB_ELDERLY_MEMBER`, `AGE_TOO_YOUNG`, `AGE_NEGATIVE`. Null = clean record. |
| `geo_flag` | string | Geographic anomaly flagging | Signals regional data issues: `REGION_NON_MEMBERS` (1 record), `REGION_UNKNOWN` (null replaced). Null = standard region. |

---

## Data Quality & Data Characteristics

### Completeness

| Column | Missing Count | Missing % | Status |
|--------|---------------|-----------|--------|
| `_c0`, `CM_snapshot_date`, `Int_nurse`, `MemCategory`, `CatName`, `Branch`, `q_members_t` | 0 | 0.00% | Complete |
| `Region` | 1 | 0.000005% | Replaced with "Unknown" |
| `YoJ` | 39 | 0.0002% | Minor |
| `YoB` | 4,406 | 0.0239% | Minor |
| `MemSectorType` | 1,436,695 | 7.78% | Substantial but expected |
| `q_leavers_t` | 18,259,005 | 98.90% | **Expected** — represents active members without departures |

**Overall Completeness:** 98.90% (when excluding expected nulls in `q_leavers_t`)

### Data Quality Issues Identified (from Notebook 01-03)

**1. Year of Birth (YoB) Outliers: 18 records (0.01% of dataset)**
- 17 records with implausible future years (2020–2966) — data entry errors
- 1 record with YoB = 1895 (pre-1900) — historical outlier
- **Remediation (Notebook 02):** Future years nulled and flagged `YOB_CONFIRMED_ERROR`; pre-1900 and young members flagged `YOB_ELDERLY_MEMBER`/`YOB_YOUNG_MEMBER`

**2. Year of Birth Default Placeholder: 270,062 records (1.46% of dataset)**
- YoB = 1900 is a legacy system default for unknown/missing birth years at registration
- Paired with plausible `YoJ` values (1991–2025), indicating legitimate members with missing birth data
- **Remediation (Notebook 02):** Flagged `YOB_DEFAULT_PLACEHOLDER` for transparency; values retained for downstream filtering

**3. Join Age Violations: 120,606 records (0.65% of dataset)**
- **TOO_OLD:** 119,684 records with derived join ages 91–126 years (mostly YoB = 1900 default)
- **TOO_YOUNG:** 900 records with join ages 0–13 years
- **NEGATIVE_AGE:** 22 records where YoJ < YoB (temporal inconsistencies)
- **Remediation (Notebook 02):** Flagged appropriately; YoB = 1900 records distinguished from genuine elderly/lifetime members

**4. MemSectorType Casing Inconsistency: 307,221 records (1.66% of dataset)**
- Same sector label appears in multiple forms: "nhs", "NHS", "Nhs", "other public sector", "Other Public Sector"
- **Remediation (Notebook 02):** Standardised to canonical form (NHS, Independent, Education, Other Public Sector)

**5. Geographic Anomalies: 2 records (0.000011% of dataset)**
- 1 record with null Region (replaced with "Unknown", flagged `REGION_UNKNOWN`)
- 1 record with Region = "Non Members" (flagged `REGION_NON_MEMBERS`)
- **Remediation (Notebook 02):** Retained with flags for transparency

**6. q_leavers_t Nullness: 18,259,005 records (98.90%)**
- **This is expected and valid.** The high null rate represents active members without departures during observation windows.
- Non-null values present only where departures occurred, with 11 distinct values.
- Mathematical pattern confirmed: `q_leavers_t` derived from same base value as `q_members_t` (base = 2.97)

### Overall Quality Rating

- **Clean records:** 18,153,056 (98.33%)
- **Records with flagged issues:** 308,424 (1.67%)
- **Overall data quality score:** 98.33% — EXCELLENT
- **Suitable for analysis:** Yes, with targeted handling of flagged records

---

## Key Analytical Notes

### Aggregated Data Structure

**Critical:** Each row does NOT represent a single member. Instead:
- Each row represents a **cohort segment** defined by: `CM_snapshot_date`, `Int_nurse`, `Region`, `MemCategory`, `CatName`, `Branch`, `YoB`, `MemSectorType`, `YoJ`
- The value `q_members_t` in each row is the **aggregated membership count** for that cohort
- **Membership calculation:** Use `SUM(q_members_t)` grouped by snapshot date, not row counts
- Example: Total membership in January 2021 = SUM of all `q_members_t` values where `CM_snapshot_date` = 2021-01-01

### Temporal Coverage

- **60 monthly snapshots:** Complete coverage from 2021-01-01 to 2025-12-01
- **No missing months:** All dates are the 1st of each calendar month
- **Panel structure:** Same cohort segments repeat across all 60 snapshots, enabling cohort tracking over time

### Geographic Hierarchy

**Two-tier hierarchy:**
1. **Region** (14 areas): South East (14.49% of members), London (13.03%), North West (9.78%), etc.
2. **Branch** (104 units): Each branch belongs to exactly one region; no multi-region branches detected

### Membership Classification

**Int_nurse (International Qualification Classification):**
- 4 categories: UK (53.23%), Other (31.09%), Overseas (12.00%), EEU (3.68%)
- Operates independently of geographic region — all categories appear in all regions
- Likely represents registration/qualification origin rather than current work location

**MemCategory (Membership Category):**
- 4 distinct values with 102 detailed category names (CatName field)
- Examples: Student, Practising, Honorary, Other

**MemSectorType (Employment Sector):**
- 5 distinct values: NHS, Independent, Education, Other Public Sector, null
- Null values (7.78%) represent members with unknown/unreported sector

### Quantitative Measures

**q_members_t (Aggregated Membership Count):**
- Base multiplier value: 2.97
- Range: 2.97 to 296.91 (approximately 100x range multiplier)
- 99 distinct values, suggesting weighted or calculated cohort counts
- Always non-null

**q_leavers_t (Aggregated Leaver Count):**
- Base multiplier value: 2.97 (identical to q_members_t)
- Range: 2.97 to 50.48 (approximately 17x range multiplier)
- 11 distinct values, sparse and heavily concentrated
- 98.90% null (expected for active members without departures)
- **Constraint:** q_leavers_t never exceeds q_members_t — validated in Notebook 01

---

## Data Processing Pipeline

### Notebook 01: Data Ingestion
- Loaded 2.06 GB CSV file with 18,461,480 rows and 12 columns
- Validated file integrity and schema inference
- Identified data quality issues (YoB outliers, nullness patterns, logical inconsistencies)
- Persisted Bronze table in Delta format

### Notebook 02: Data Cleaning & Transformation
- Cast YoB and YoJ from double to integer type
- Applied tiered remediation to 18 YoB outliers (future years nulled; young/elderly members flagged)
- Applied tiered remediation to 120,606 join age violations (flagged by sub-type)
- Identified and handled 270,062 YoB = 1900 default placeholders (flagged separately)
- Standardised MemSectorType casing across 307,221 inconsistent records
- Handled geographic anomalies: null Region → "Unknown"; "Non Members" → flagged
- Generated Silver table with 14 columns (original 12 + `cleaning_flag` + `geo_flag`)
- All 18,461,480 rows preserved; no membership counts modified

### Notebook 03: Data Quality & Missingness Validation
- Validated completeness patterns and confirmed expected nullness in q_leavers_t
- Confirmed temporal consistency across 60 monthly snapshots
- Validated branch-region hierarchy (all 104 branches map to exactly 1 region)
- Confirmed regional proportions remain stable across 5-year period
- Assessed outliers and anomaly distribution

### Notebook 04: Exploratory Data Analysis Part 1
- Temporal trend analysis of membership by snapshot date
- Distributional analysis of categorical variables (region, category, sector)
- Geographic and taxonomic breakdowns

### Notebook 05: Exploratory Data Analysis Part 2
- Segmentation and cohort analysis
- Regional analysis and age band distributions
- Advanced cohort and temporal pattern investigation

---

## Data Access & Storage

- **Raw data:** Stored at `/Volumes/workspace/rcn_churn/raw_data/churn_t_db.csv` (2.06 GB)
- **Bronze table (Notebook 01 output):** `/Volumes/workspace/rcn_churn/raw_data/delta_raw/`
- **Silver table (Notebook 02 output):** `/Volumes/workspace/rcn_churn/silver/churn_cleaned/`
- **Format:** Delta Lake (Parquet-based, optimised for Spark processing)
- **Access:** Restricted to authorised analytical environment only

---

## Data Limitations & Caveats

1. **Aggregated structure:** Analysis operates on cohort-level data, not individual members. Member-level insights are not possible without member-to-cohort mapping.

2. **Missing birth years:** 270,062 records (1.46%) carry YoB = 1900 (default placeholder). Age-based analysis should exclude or flag these records.

3. **Limited sector data:** 7.78% of records lack MemSectorType values. Sector-based segmentation may undercount membership in unknown categories.

4. **Active member bias:** The 98.90% null rate in q_leavers_t reflects active membership focus; historical churn records may be underrepresented.

5. **Regional standardisation:** Region values may have been standardised or recoded post-hoc; original organisation-specific naming is not recoverable.

6. **Data entry errors:** YoB outliers (future years up to 2966) and join age violations indicate legacy data quality issues from multi-decade membership system history.

---

## Analytical Use Cases

This dataset was used exclusively for the following purposes under data access agreement terms:

- Membership trend analysis across 60 monthly snapshots
- Cohort segmentation by region, category, sector, and age band
- Temporal stability analysis of membership demographics
- Data quality assessment and validation
- Exploratory analysis of membership composition and distributions

---

## Data Governance

- **All personally identifiable information (PII):** Removed prior to analysis
- **Anonymisation:** Members represented as cohort aggregates; no individual-level re-identification possible
- **Proprietary designation:** Dataset remains exclusive to the originating organisation
- **Reproduction:** Synthetic datasets with equivalent schema may be substituted for analysis replication

---

*This document describes dataset structure, schema, and quality characteristics as evidenced by the exploratory data analysis notebooks (01–05) conducted February–March 2026. It is maintained for analytical transparency and internal documentation only. It does not constitute disclosure of confidential or proprietary information.*
