"""Build reusable underwriting output datasets for Streamlit and Power BI.

Usage:
  python build_output_data.py --input data/auto_issuance_synthetic_1year_10000rows.csv --output output
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "[]"}
DATE_TOKEN_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")
REASON_SKIP_TOKENS = {"unknown", "unspecified"}
RECOMMENDATION_COLUMNS = [
    "Reasons for Hold",
    "Next Steps for Customer",
    "Next Steps for Underwriter",
    "Next Steps for Ops Team",
    "Next Steps for Customer Service",
]


def load_data(input_path: str | Path) -> pd.DataFrame:
    input_path = str(input_path)
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path, low_memory=False)
    if input_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(input_path, engine="openpyxl")
    raise ValueError(f"Unsupported input format: {input_path}")


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().strip("[]")
    if not text or text.lower() in MISSING_TOKENS:
        return ""
    return text


def clean_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for token in tokens:
        item = str(token).strip()
        if not item or item.lower() in MISSING_TOKENS:
            continue
        out.append(item)
    return out


def parse_date_history_cell(value: object) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if "|" in text:
        return clean_tokens(text.split("|"))
    if "," in text:
        matches = DATE_TOKEN_PATTERN.findall(text)
        if matches:
            return clean_tokens(matches)
        return clean_tokens(re.split(r"\s*,\s*", text))
    return clean_tokens([text])


def parse_reason_history_cell(value: object, expected_count: int | None = None) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    if "|" in text:
        return clean_tokens(text.split("|"))
    if "," in text:
        if expected_count is not None and expected_count <= 1:
            return clean_tokens([text])
        parts = clean_tokens(re.split(r"\s*,\s*", text))
        if expected_count is not None and expected_count > 1 and len(parts) != expected_count:
            return clean_tokens([text])
        return parts
    return clean_tokens([text])


def normalize_reason_text(value: object) -> str:
    text = clean_text(value).lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def reason_tokens(text: str) -> set[str]:
    stop = {"for", "and", "the", "to", "of", "in", "on", "a", "an"}
    tokens = []
    for tok in text.split():
        t = tok.strip()
        if not t or t in stop:
            continue
        if len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        tokens.append(t)
    return set(tokens)


def best_recommendation_key(reason_norm: str, rec_norm_values: list[str]) -> str | None:
    if not reason_norm or not rec_norm_values:
        return None

    # 1) Exact
    if reason_norm in rec_norm_values:
        return reason_norm

    # 2) Contains
    contains_candidates = [rn for rn in rec_norm_values if (reason_norm in rn) or (rn in reason_norm)]
    if contains_candidates:
        return max(contains_candidates, key=len)

    # 3) Token overlap
    rt = reason_tokens(reason_norm)
    best_key = None
    best_overlap = 0.0
    for rn in rec_norm_values:
        tt = reason_tokens(rn)
        if not rt or not tt:
            continue
        overlap = len(rt & tt) / max(len(rt), 1)
        if overlap > best_overlap:
            best_overlap = overlap
            best_key = rn
    if best_key is not None and best_overlap > 0:
        return best_key

    # 4) Sequence similarity
    scored = [(SequenceMatcher(None, reason_norm, rn).ratio(), rn) for rn in rec_norm_values]
    scored.sort(reverse=True)
    if scored and scored[0][0] >= 0.58:
        return scored[0][1]
    return None


def load_recommendation_reference(path: Path | str = "recommendation.csv") -> pd.DataFrame:
    rec_path = Path(path)
    if not rec_path.exists():
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    rec_df = pd.read_csv(rec_path)
    missing_cols = [c for c in RECOMMENDATION_COLUMNS if c not in rec_df.columns]
    for col in missing_cols:
        rec_df[col] = np.nan

    rec_df = rec_df[RECOMMENDATION_COLUMNS].copy()
    rec_df["__reason_norm"] = rec_df["Reasons for Hold"].apply(normalize_reason_text)
    rec_df = rec_df[rec_df["__reason_norm"].ne("")].drop_duplicates("__reason_norm", keep="first")
    return rec_df


def map_recommendations(reason_counts_df: pd.DataFrame, recommendation_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "onHoldReasonDescriptionsHistory",
        "Count",
        "Reasons for Hold",
        "Next Steps for Customer",
        "Next Steps for Underwriter",
        "Next Steps for Ops Team",
        "Next Steps for Customer Service",
    ]
    if reason_counts_df.empty:
        return pd.DataFrame(columns=base_cols)

    rec_by_norm = {}
    rec_norm_values = []
    if not recommendation_df.empty and "__reason_norm" in recommendation_df.columns:
        rec_by_norm = {row["__reason_norm"]: row for _, row in recommendation_df.iterrows()}
        rec_norm_values = list(rec_by_norm.keys())

    out_rows = []
    for _, row in reason_counts_df.iterrows():
        reason_text = row["Value"]
        reason_norm = normalize_reason_text(reason_text)
        matched = None

        best_key = best_recommendation_key(reason_norm, rec_norm_values)
        if best_key is not None:
            matched = rec_by_norm[best_key]

        out_rows.append(
            {
                "onHoldReasonDescriptionsHistory": reason_text,
                "Count": row["Count"],
                "Reasons for Hold": matched["Reasons for Hold"] if matched is not None else np.nan,
                "Next Steps for Customer": matched["Next Steps for Customer"] if matched is not None else np.nan,
                "Next Steps for Underwriter": matched["Next Steps for Underwriter"] if matched is not None else np.nan,
                "Next Steps for Ops Team": matched["Next Steps for Ops Team"] if matched is not None else np.nan,
                "Next Steps for Customer Service": matched["Next Steps for Customer Service"] if matched is not None else np.nan,
            }
        )

    return pd.DataFrame(out_rows, columns=base_cols)


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_case_type(hold_count: float) -> str:
    if hold_count <= 0:
        return "Straight Through"
    if hold_count == 1:
        return "One Touch"
    return f"Multi Hold ({int(hold_count)} touches)"


def create_tat_bucket(tat_days: float | int | None) -> str | None:
    if pd.isna(tat_days):
        return None
    if tat_days <= 4:
        return "1-4 days"
    if tat_days <= 7:
        return "5-7 days"
    return "7+ days"


def to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_dayfirst = pd.to_datetime(series, errors="coerce", dayfirst=True)
        parsed_default = pd.to_datetime(series, errors="coerce")

    if parsed_dayfirst.notna().sum() > parsed_default.notna().sum():
        return parsed_dayfirst
    return parsed_default


def parse_datetime_value(value):
    text = clean_text(value)
    if not text:
        return pd.NaT

    # Parse strict ISO first to avoid day-first misreads (e.g., 2025-03-01).
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        iso_dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if pd.notna(iso_dt):
            return iso_dt

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        dt_dayfirst = pd.to_datetime(text, errors="coerce", dayfirst=True)
        dt_default = pd.to_datetime(text, errors="coerce")
    if pd.notna(dt_dayfirst):
        return dt_dayfirst
    return dt_default


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    create_col = resolve_column(data, ["createDateTime"])
    complete_col = resolve_column(data, ["completedDateTime"])
    history_reason_col = resolve_column(data, ["onHoldReasonDescriptionsHistory"])
    on_hold_dates_col = resolve_column(data, ["onHoldDatesHistory"])
    off_hold_dates_col = resolve_column(data, ["offHoldDatesHistory"])

    if create_col:
        data["createDateTime"] = parse_datetime_series(data[create_col])
    else:
        data["createDateTime"] = pd.NaT

    if complete_col:
        data["completedDateTime"] = parse_datetime_series(data[complete_col])
    else:
        data["completedDateTime"] = pd.NaT

    tat_from_dates = (data["completedDateTime"] - data["createDateTime"]).dt.total_seconds() / 86400.0
    data["TAT_Days"] = tat_from_dates

    data.loc[data["TAT_Days"] < 0, "TAT_Days"] = np.nan
    data["TAT_Bucket"] = data["TAT_Days"].apply(create_tat_bucket)

    hold_reasons_list = []
    all_hold_reasons_list = []
    hold_days_list = []
    hold_counts_list = []
    global_last_completed_dt = data["completedDateTime"].dropna().max()

    for _, row in data.iterrows():
        history_text = row.get(history_reason_col, "") if history_reason_col else ""
        on_hold_raw = parse_date_history_cell(row.get(on_hold_dates_col, "")) if on_hold_dates_col else []
        off_hold_raw = parse_date_history_cell(row.get(off_hold_dates_col, "")) if off_hold_dates_col else []
        reasons = parse_reason_history_cell(history_text, expected_count=len(on_hold_raw) if on_hold_raw else None)

        reasons_for_top = parse_reason_history_cell(history_text, expected_count=None)
        all_hold_reasons_list.append(reasons_for_top)

        on_hold_dates = [parse_datetime_value(v) for v in on_hold_raw]
        off_hold_dates = [parse_datetime_value(v) for v in off_hold_raw]

        aligned_reasons = []
        aligned_days = []
        event_count = max(len(on_hold_dates), len(reasons))

        for idx in range(event_count):
            on_dt = on_hold_dates[idx] if idx < len(on_hold_dates) else pd.NaT
            if pd.isna(on_dt):
                continue

            reason_val = reasons[idx] if idx < len(reasons) else (reasons[-1] if reasons else "Unknown")
            reason_val = str(reason_val).strip() if str(reason_val).strip() else "Unknown"

            off_dt = off_hold_dates[idx] if idx < len(off_hold_dates) else pd.NaT
            if pd.notna(off_dt):
                end_dt = off_dt
            elif pd.notna(global_last_completed_dt):
                end_dt = global_last_completed_dt
            else:
                continue

            days = (end_dt - on_dt).total_seconds() / 86400.0
            if pd.notna(days) and days >= 0:
                aligned_reasons.append(reason_val)
                aligned_days.append(float(days))

        hold_reasons_list.append(aligned_reasons)
        hold_days_list.append(aligned_days)
        hold_counts_list.append(float(len(aligned_days)) if aligned_days else 1.0)

    data["Hold_Reasons_List"] = hold_reasons_list
    data["All_Hold_Reasons_List"] = all_hold_reasons_list
    data["Hold_Days_List"] = hold_days_list
    data["Total_Hold_Days"] = data["Hold_Days_List"].apply(lambda v: float(np.sum(v)) if v else 0.0)
    data["Hold_Count"] = pd.Series(hold_counts_list, index=data.index, dtype="float64").fillna(0).clip(lower=0)
    data["CaseType"] = data["Hold_Count"].apply(infer_case_type)

    data["Month"] = data["createDateTime"].dt.to_period("M")
    data["Month_Str"] = data["Month"].astype(str)
    data.loc[data["createDateTime"].isna(), "Month_Str"] = np.nan

    data["PrimaryHoldReason"] = data["All_Hold_Reasons_List"].apply(lambda v: v[0] if isinstance(v, list) and v else "Unknown")

    return data


def build_hold_events(prepared_df: pd.DataFrame) -> pd.DataFrame:
    request_col = resolve_column(prepared_df, ["requestId"])
    process_col = resolve_column(prepared_df, ["processId"])

    records = []
    for idx, row in prepared_df.iterrows():
        reasons = row.get("Hold_Reasons_List", [])
        days_list = row.get("Hold_Days_List", [])

        if not isinstance(reasons, list) or not isinstance(days_list, list):
            continue

        for i, reason in enumerate(reasons):
            if i >= len(days_list):
                continue
            hold_days = days_list[i]
            if pd.isna(hold_days):
                continue

            records.append(
                {
                    "row_index": int(idx),
                    "requestId": row.get(request_col) if request_col else None,
                    "processId": row.get(process_col) if process_col else None,
                    "Month_Str": row.get("Month_Str"),
                    "TAT_Days": row.get("TAT_Days"),
                    "TAT_Bucket": row.get("TAT_Bucket"),
                    "Hold_Count": row.get("Hold_Count"),
                    "onHoldReasonDescriptionsHistory": reason,
                    "Hold_Days": float(hold_days),
                }
            )

    return pd.DataFrame(records)


def build_reason_events(prepared_df: pd.DataFrame) -> pd.DataFrame:
    request_col = resolve_column(prepared_df, ["requestId"])
    process_col = resolve_column(prepared_df, ["processId"])

    records = []
    for idx, row in prepared_df.iterrows():
        reasons = row.get("All_Hold_Reasons_List", [])
        if not isinstance(reasons, list) or not reasons:
            continue

        for reason in reasons:
            reason_text = str(reason).strip()
            if not reason_text:
                continue
            if reason_text.lower() in REASON_SKIP_TOKENS:
                continue
            records.append(
                {
                    "row_index": int(idx),
                    "requestId": row.get(request_col) if request_col else None,
                    "processId": row.get(process_col) if process_col else None,
                    "Month_Str": row.get("Month_Str"),
                    "TAT_Days": row.get("TAT_Days"),
                    "TAT_Bucket": row.get("TAT_Bucket"),
                    "onHoldReasonDescriptionsHistory": reason_text,
                }
            )

    return pd.DataFrame(records)


def build_output_tables(prepared_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    completed_df = prepared_df[prepared_df["TAT_Days"].notna()].copy()
    high_tat_df = completed_df[completed_df["TAT_Bucket"] == "7+ days"].copy()
    on_hold_df = prepared_df[prepared_df["Hold_Count"] > 0].copy()

    hold_events_df = build_hold_events(prepared_df)
    reason_events_df = build_reason_events(prepared_df)
    recommendation_df = load_recommendation_reference()

    monthly_tat_bucket = (
        completed_df.dropna(subset=["Month_Str", "TAT_Bucket"])
        .groupby(["Month_Str", "TAT_Bucket"]) 
        .size()
        .reset_index(name="Count")
        .sort_values("Month_Str")
    )

    monthly_holds_by_tat_bucket = (
        completed_df.dropna(subset=["Month_Str", "TAT_Bucket"])
        .groupby(["Month_Str", "TAT_Bucket"]) 
        .agg(Avg_Holds=("Hold_Count", "mean"), Cases=("TAT_Days", "size"))
        .reset_index()
        .sort_values("Month_Str")
    )

    hold_reason_impact = pd.DataFrame(columns=["onHoldReasonDescriptionsHistory", "Count", "Total_Hold_Days", "Avg_Hold_Days"])
    if not hold_events_df.empty:
        hold_reason_impact = (
            hold_events_df.groupby("onHoldReasonDescriptionsHistory")
            .agg(Count=("Hold_Days", "size"), Total_Hold_Days=("Hold_Days", "sum"), Avg_Hold_Days=("Hold_Days", "mean"))
            .reset_index()
            .sort_values("Total_Hold_Days", ascending=False)
        )

    def top_with_avg_tat(df: pd.DataFrame, col_candidates: list[str], top_n: int = 10) -> pd.DataFrame:
        col = resolve_column(df, col_candidates)
        if col is None or col not in df.columns:
            return pd.DataFrame(columns=["Value", "Number_of_Cases", "Avg_TAT_Days"])
        out = (
            df.assign(Value=df[col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown"))
            .groupby("Value")
            .agg(Number_of_Cases=("TAT_Days", "size"), Avg_TAT_Days=("TAT_Days", "mean"))
            .reset_index()
            .sort_values("Number_of_Cases", ascending=False)
            .head(top_n)
        )
        return out

    top_brokers = top_with_avg_tat(prepared_df, ["AgentBrokerName", "AgentBrokerName__2", "agentBrokerNum"], top_n=10)
    top_account_analyst_7plus = top_with_avg_tat(high_tat_df, ["accountAnalyst", "accountAnalystName"], top_n=10)
    top_underwriter_7plus = top_with_avg_tat(high_tat_df, ["underwriterName", "underwriter"], top_n=10)
    top_rater_7plus = top_with_avg_tat(high_tat_df, ["raterFullName"], top_n=10)

    top_7plus_hold_reason = pd.DataFrame(columns=["Value", "Number_of_Cases", "Avg_TAT_Days"])
    if not reason_events_df.empty:
        reason_7plus = reason_events_df[reason_events_df["TAT_Bucket"] == "7+ days"].copy()
        if not reason_7plus.empty:
            top_7plus_hold_reason = (
                reason_7plus.assign(
                    Value=reason_7plus["onHoldReasonDescriptionsHistory"]
                    .fillna("Unknown")
                    .astype(str)
                    .str.strip()
                    .replace("", "Unknown")
                )
                .groupby("Value")
                .agg(Number_of_Cases=("TAT_Days", "size"), Avg_TAT_Days=("TAT_Days", "mean"))
                .reset_index()
                .sort_values("Number_of_Cases", ascending=False)
                .head(5)
            )
    top_7plus_bgi = top_with_avg_tat(high_tat_df, ["bgiDescription"], top_n=5)
    top_7plus_lob = top_with_avg_tat(high_tat_df, ["lineOfBusinessDescription"], top_n=5)
    top_7plus_state = top_with_avg_tat(high_tat_df, ["AgentBrokerStateCode"], top_n=5)

    top_hold_reason_counts = pd.DataFrame(columns=["Value", "Count"])
    reason_tokens: list[str] = []
    for _, row in on_hold_df.iterrows():
        reasons = row.get("All_Hold_Reasons_List", [])
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            reason_text = str(reason).strip()
            if reason_text and reason_text.lower() not in REASON_SKIP_TOKENS:
                reason_tokens.append(reason_text)

    if reason_tokens:
        reason_series = pd.Series(reason_tokens, dtype="object")
        top_hold_reason_counts = reason_series.value_counts().head(10).reset_index()
        top_hold_reason_counts.columns = ["Value", "Count"]

    prescriptive_actions = map_recommendations(top_hold_reason_counts, recommendation_df)

    tables = {
        "fact_cases": prepared_df,
        "fact_hold_events": hold_events_df,
        "fact_reason_events": reason_events_df,
        "agg_monthly_tat_bucket": monthly_tat_bucket,
        "agg_monthly_holds_by_tat_bucket": monthly_holds_by_tat_bucket,
        "agg_hold_reason_impact": hold_reason_impact,
        "agg_top_brokers": top_brokers,
        "agg_top_account_analyst_7plus": top_account_analyst_7plus,
        "agg_top_underwriter_7plus": top_underwriter_7plus,
        "agg_top_rater_7plus": top_rater_7plus,
        "agg_top_7plus_onholdreason": top_7plus_hold_reason,
        "agg_top_7plus_bgi": top_7plus_bgi,
        "agg_top_7plus_lob": top_7plus_lob,
        "agg_top_7plus_state": top_7plus_state,
        "agg_prescriptive_actions_top_hold_reasons": prescriptive_actions,
    }
    return tables


def write_outputs(tables: dict[str, pd.DataFrame], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    written_files = []
    for name, df in tables.items():
        out_df = df.copy()

        # Serialize list columns to pipe-delimited strings for cross-tool compatibility.
        for list_col in ["Hold_Reasons_List", "All_Hold_Reasons_List", "Hold_Days_List"]:
            if list_col in out_df.columns:
                out_df[list_col] = out_df[list_col].apply(
                    lambda v: "|".join(map(str, v)) if isinstance(v, list) else v
                )

        out_path = output_dir / f"{name}.csv"
        out_df.to_csv(out_path, index=False)
        written_files.append({"name": name, "file": out_path.name, "rows": int(len(out_df)), "columns": int(len(out_df.columns))})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir.resolve()),
        "tables": written_files,
    }

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def run_pipeline(input_path: Path, output_dir: Path) -> dict:
    source_df = load_data(input_path)
    prepared_df = prepare_dataframe(source_df)
    tables = build_output_tables(prepared_df)
    return write_outputs(tables, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Build underwriting output data model for Streamlit and Power BI.")
    parser.add_argument("--input", required=True, help="Input CSV/XLSX file path")
    parser.add_argument("--output", default="output", help="Output folder path (default: output)")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = run_pipeline(Path(args.input), Path(args.output))
    print("Output created successfully")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
