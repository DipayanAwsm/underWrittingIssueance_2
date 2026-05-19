# Shared Output Pipeline (Streamlit + Power BI)

This project now supports a shared processing layer so both Streamlit and Power BI can consume the same output data.

## 1) Run the pipeline

```bash
python build_output_data.py \
  --input data/auto_issuance_synthetic_1year_10000rows.csv \
  --output output
```

## 2) What gets generated in `output/`

- `fact_cases.csv` (main curated fact table for filters and KPIs)
- `fact_hold_events.csv` (hold-event grain table)
- `agg_monthly_tat_bucket.csv`
- `agg_monthly_holds_by_tat_bucket.csv`
- `agg_hold_reason_impact.csv`
- `agg_top_brokers.csv`
- `agg_top_account_analyst_7plus.csv`
- `agg_top_underwriter_7plus.csv`
- `agg_top_rater_7plus.csv`
- `agg_top_7plus_onholdreason.csv`
- `agg_top_7plus_bgi.csv`
- `agg_top_7plus_lob.csv`
- `agg_top_7plus_state.csv`
- `manifest.json`

## 3) Streamlit usage

Run:

```bash
streamlit run app_intelligent_view.py
```

In sidebar `Choose data source`, select:

- `Processed Output Folder`

This reads `output/fact_cases.csv` directly.

## 4) Power BI usage

In Power BI Desktop:

1. Get Data → Text/CSV
2. Load one or more files from `output/`
3. Build visuals from:
   - `fact_cases.csv` for flexible slicing/filtering
   - `agg_*.csv` for fast pre-aggregated visuals

## 5) Recommended production flow

1. Drop latest source file in `data/`.
2. Run `build_output_data.py`.
3. Refresh Streamlit and Power BI.

This guarantees both tools use a single, consistent data model.
