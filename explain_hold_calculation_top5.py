"""Create an Excel audit report for hold-time calculation on top 5 hold-count cases.

Logic implemented (as requested):
1) Parse onHoldDatesHistory into a list of on-hold timestamps.
2) Parse offHoldDatesHistory into a list of off-hold timestamps.
3) Parse onHoldReasonDescriptionsHistory into a reason list.
4) For each hold index i:
   - start_dt = on_hold_dates[i] (if missing/invalid, skip)
   - end_dt priority:
     a) off_hold_dates[i] if present/valid
     b) else last completedDateTime in the data (global max completed datetime)
   - hold_days = (end_dt - start_dt).total_seconds() / 86400
   - keep only if hold_days >= 0
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a", "-", "[]"}
DATE_TOKEN_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}")


def load_data(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path, low_memory=False)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path, engine="openpyxl")
    raise ValueError(f"Unsupported input format: {input_path}")


def resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


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


def parse_datetime_value(value: object) -> pd.Timestamp:
    text = clean_text(value)
    if not text:
        return pd.NaT

    # Prefer strict ISO parsing when token starts with YYYY-MM-DD.
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        iso_dt = pd.to_datetime(text, errors="coerce", dayfirst=False)
        if pd.notna(iso_dt):
            return iso_dt

    dayfirst_dt = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(dayfirst_dt):
        return dayfirst_dt
    return pd.to_datetime(text, errors="coerce", dayfirst=False)


def parse_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_datetime_value)


def build_hold_audit_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    request_col = resolve_column(df, ["requestId"])
    process_col = resolve_column(df, ["processId"])
    create_col = resolve_column(df, ["createDateTime"])
    complete_col = resolve_column(df, ["completedDateTime"])
    on_col = resolve_column(df, ["onHoldDatesHistory"])
    off_col = resolve_column(df, ["offHoldDatesHistory"])
    reason_col = resolve_column(df, ["onHoldReasonDescriptionsHistory"])

    if on_col is None or reason_col is None:
        raise ValueError("Required columns missing: onHoldDatesHistory and/or onHoldReasonDescriptionsHistory")

    work_df = df.copy()
    if complete_col:
        work_df["_completed_dt"] = parse_datetime_series(work_df[complete_col])
    else:
        work_df["_completed_dt"] = pd.NaT

    if create_col:
        work_df["_create_dt"] = parse_datetime_series(work_df[create_col])
    else:
        work_df["_create_dt"] = pd.NaT

    global_last_completed_dt = work_df["_completed_dt"].dropna().max()

    case_rows: list[dict] = []
    event_rows: list[dict] = []

    for row_index, row in work_df.iterrows():
        on_tokens = parse_date_history_cell(row.get(on_col, ""))
        off_tokens = parse_date_history_cell(row.get(off_col, "")) if off_col else []
        reason_tokens = parse_reason_history_cell(
            row.get(reason_col, ""),
            expected_count=len(on_tokens) if on_tokens else None,
        )

        derived_hold_count = int(max(len(on_tokens), len(reason_tokens)))
        valid_event_count = 0
        total_hold_days = 0.0

        for i in range(derived_hold_count):
            start_token = on_tokens[i] if i < len(on_tokens) else ""
            start_dt = parse_datetime_value(start_token) if start_token else pd.NaT

            reason_text = reason_tokens[i] if i < len(reason_tokens) else (reason_tokens[-1] if reason_tokens else "Unknown")
            reason_text = str(reason_text).strip() if str(reason_text).strip() else "Unknown"

            off_token = off_tokens[i] if i < len(off_tokens) else ""
            off_dt = parse_datetime_value(off_token) if off_token else pd.NaT

            end_dt = pd.NaT
            end_source = "missing"
            if pd.notna(off_dt):
                end_dt = off_dt
                end_source = "offHoldDatesHistory"
            elif pd.notna(global_last_completed_dt):
                end_dt = global_last_completed_dt
                end_source = "global_last_completedDateTime"

            hold_days = np.nan
            keep_event = False
            drop_reason = ""

            if pd.isna(start_dt):
                drop_reason = "Skipped: start_dt missing/invalid"
            elif pd.isna(end_dt):
                drop_reason = "Skipped: end_dt missing"
            else:
                hold_days = (end_dt - start_dt).total_seconds() / 86400.0
                if pd.notna(hold_days) and hold_days >= 0:
                    keep_event = True
                    valid_event_count += 1
                    total_hold_days += float(hold_days)
                else:
                    drop_reason = "Dropped: hold_days < 0"

            event_rows.append(
                {
                    "row_index": int(row_index),
                    "requestId": row.get(request_col) if request_col else None,
                    "processId": row.get(process_col) if process_col else None,
                    "event_index_1based": int(i + 1),
                    "reason_from_history": reason_text,
                    "start_token": start_token,
                    "start_dt": start_dt,
                    "off_token": off_token,
                    "off_dt": off_dt,
                    "global_last_completed_dt": global_last_completed_dt,
                    "end_dt_used": end_dt,
                    "end_source": end_source,
                    "hold_days": hold_days,
                    "keep_event": keep_event,
                    "drop_reason": drop_reason,
                    "formula": "(end_dt_used - start_dt).total_seconds() / 86400",
                }
            )

        case_rows.append(
            {
                "row_index": int(row_index),
                "requestId": row.get(request_col) if request_col else None,
                "processId": row.get(process_col) if process_col else None,
                "createDateTime": row.get(create_col) if create_col else None,
                "completedDateTime": row.get(complete_col) if complete_col else None,
                "onHoldDatesHistory_raw": row.get(on_col),
                "offHoldDatesHistory_raw": row.get(off_col) if off_col else None,
                "onHoldReasonDescriptionsHistory_raw": row.get(reason_col),
                "parsed_on_count": len(on_tokens),
                "parsed_off_count": len(off_tokens),
                "parsed_reason_count": len(reason_tokens),
                "derived_hold_count": derived_hold_count,
                "valid_hold_events": valid_event_count,
                "total_hold_days_kept": float(total_hold_days),
            }
        )

    case_df = pd.DataFrame(case_rows)
    event_df = pd.DataFrame(event_rows)

    # Top 5 by hold count, then valid events, then total hold days.
    top5_case_df = case_df.sort_values(
        ["derived_hold_count", "valid_hold_events", "total_hold_days_kept"],
        ascending=[False, False, False],
    ).head(5)

    top5_idx = set(top5_case_df["row_index"].tolist())
    top5_events_df = event_df[event_df["row_index"].isin(top5_idx)].copy()
    top5_events_df = top5_events_df.sort_values(["derived_hold_count"] if "derived_hold_count" in top5_events_df.columns else ["row_index", "event_index_1based"])
    top5_events_df = top5_events_df.sort_values(["row_index", "event_index_1based"])

    source_cols = [c for c in [request_col, process_col, create_col, complete_col, on_col, off_col, reason_col] if c]
    source_snapshot_df = work_df.loc[top5_case_df["row_index"].tolist(), source_cols].copy()
    source_snapshot_df.insert(0, "row_index", source_snapshot_df.index.astype(int))

    return top5_case_df, top5_events_df, source_snapshot_df


def build_logic_sheet() -> pd.DataFrame:
    lines = [
        "1. Parse onHoldDatesHistory into list of on-hold timestamps.",
        "2. Parse offHoldDatesHistory into list of off-hold timestamps.",
        "3. Parse onHoldReasonDescriptionsHistory into reason list.",
        "4. For each hold index i:",
        "   a) start_dt = on_hold_dates[i] (if missing/invalid => skip event).",
        "   b) end_dt priority:",
        "      i) off_hold_dates[i] if present/valid",
        "      ii) else global last completedDateTime in data",
        "   c) hold_days = (end_dt - start_dt).total_seconds() / 86400",
        "   d) keep only if hold_days >= 0",
    ]
    return pd.DataFrame({"Logic_Steps": lines})


def write_excel_report(
    output_path: Path,
    top5_case_df: pd.DataFrame,
    top5_events_df: pd.DataFrame,
    source_snapshot_df: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logic_df = build_logic_sheet()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        logic_df.to_excel(writer, index=False, sheet_name="Logic")
        top5_case_df.to_excel(writer, index=False, sheet_name="Top5_Case_Summary")
        top5_events_df.to_excel(writer, index=False, sheet_name="Top5_Event_Calcs")
        source_snapshot_df.to_excel(writer, index=False, sheet_name="Top5_Source_Snapshot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build top-5 hold calculation audit Excel.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/auto_issuance_synthetic_1year_10000rows.csv"),
        help="Input CSV/XLSX path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/hold_calculation_top5_audit.xlsx"),
        help="Output Excel path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.input)
    top5_case_df, top5_events_df, source_snapshot_df = build_hold_audit_frames(df)
    write_excel_report(args.output, top5_case_df, top5_events_df, source_snapshot_df)

    print("Hold calculation audit report created")
    print(f"Input:  {args.input.resolve()}")
    print(f"Output: {args.output.resolve()}")
    print(f"Top-5 cases: {len(top5_case_df)}")
    print(f"Event rows:  {len(top5_events_df)}")


if __name__ == "__main__":
    main()

