# -*- coding: utf-8 -*-
"""
Streamlit Time Series Exploration & Visualization (single file)
Features:
1) Upload and read a CSV (tries common encodings/separators automatically).
2) Show column names so you can choose: feature columns, prediction target, and columns to ignore.
3) Each row is treated as one time point by default; you can optionally specify a time column.
4) Time-series visualization for selected columns (optional resampling and scaling).

Run:
  pip install streamlit pandas altair numpy
  streamlit run app.py
"""

import io
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Page config
st.set_page_config(page_title="Time Series Explorer", layout="wide")
alt.data_transformers.disable_max_rows()

# -------------------------
# Utilities
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv_file(content: bytes) -> Tuple[pd.DataFrame, str]:
    """Read CSV robustly: try multiple encodings; use sep=None to sniff delimiters.
    Returns (DataFrame, used_encoding).
    """
    last_err = None
    for enc in ["utf-8", "gb18030", "latin1"]:
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=None, engine="python")
            return df, enc
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"CSV read failed. Please check the file format. Last error: {last_err}")


def normalize_columns(df: pd.DataFrame, cols: List[str], method: str) -> Tuple[pd.DataFrame, List[str]]:
    """Apply standardization/normalization to specified columns; return new df & new column name list."""
    if method == "None" or not cols:
        return df, cols

    df = df.copy()
    new_cols = []

    if method == "Z-score standardization":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                new_c = f"{c} (z)"
                mu, sigma = df[c].mean(), df[c].std()
                if sigma and not np.isnan(sigma):
                    df[new_c] = (df[c] - mu) / sigma
                else:
                    df[new_c] = 0.0
                new_cols.append(new_c)
    elif method == "Min-Max normalization":
        for c in cols:
            if pd.api.types.is_numeric_dtype(df[c]):
                new_c = f"{c} (minmax)"
                mn, mx = df[c].min(), df[c].max()
                if mx != mn:
                    df[new_c] = (df[c] - mn) / (mx - mn)
                else:
                    df[new_c] = 0.0
                new_cols.append(new_c)

    # Keep non-numeric columns as-is if they were selected
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            new_cols.append(c)

    return df, new_cols


def build_line_chart(df_long: pd.DataFrame, time_kind: str):
    """Plot multi-line chart with Altair. time_kind: 'datetime' or 'step'."""
    x_encode = alt.X("_time:T", title="Time") if time_kind == "datetime" else alt.X("_time:Q", title="Step")

    chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=x_encode,
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Variable:N", legend=alt.Legend(title="Variable")),
            tooltip=["Variable:N", "_time:T" if time_kind == "datetime" else "_time:Q", "Value:Q"],
        )
        .properties(height=420)
        .interactive()
    )
    return chart


# -------------------------
# Sidebar: upload & config
# -------------------------
st.sidebar.header("① Upload CSV")
file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if file is None:
    st.title("Time Series Exploration & Visualization")
    st.info("Please upload a CSV file in the sidebar. Each row is treated as one time point by default.")
    st.stop()

# Read data
content_bytes = file.getvalue()
df, used_encoding = load_csv_file(content_bytes)

st.sidebar.success(f"Loaded successfully (encoding: {used_encoding}, shape: {df.shape[0]}×{df.shape[1]})")

# Basic data info
st.subheader("Data Preview")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.dataframe(df.head(50), use_container_width=True)
with col2:
    st.metric("Rows", f"{df.shape[0]:,}")
with col3:
    st.metric("Columns", f"{df.shape[1]:,}")

all_cols = list(df.columns)
if not all_cols:
    st.error("No columns detected. Please check your file.")
    st.stop()

st.sidebar.header("② Column Selection")
time_col_choice = st.sidebar.selectbox(
    "Time column (optional; index order by default)",
    options=["<Index order>"] + all_cols,
    index=0,
)

exclude_cols = st.sidebar.multiselect("Columns to ignore", options=all_cols)

avail_cols = [c for c in all_cols if c not in exclude_cols]
if not avail_cols:
    st.warning("All columns are excluded. Please keep at least one column.")

target_choice = st.sidebar.selectbox(
    "Prediction target (target)",
    options=["<Not selected>"] + avail_cols,
    index=0,
)

# Default features: all available columns except target
default_feats = [c for c in avail_cols if c != target_choice]
feature_cols = st.sidebar.multiselect(
    "Feature columns (features)",
    options=avail_cols,
    default=default_feats[: min(len(default_feats), 8)],  # Avoid too many by default; at most 8
)

# Clean mutual exclusions
feature_cols = [c for c in feature_cols if c != target_choice and c not in exclude_cols]

# Scaling settings
st.sidebar.header("③ Visualization Settings")
scale_method = st.sidebar.selectbox("Scaling", ["None", "Z-score standardization", "Min-Max normalization"], index=0)

# Resampling (only if there's a real datetime column)
freq_map = {"No resampling": None, "Day (D)": "D", "Week (W)": "W", "Month (M)": "M", "Year (A)": "A"}
resample_choice = st.sidebar.selectbox("Resampling frequency", list(freq_map.keys()), index=0)

# Time range filter toggle
range_filter_on = st.sidebar.checkbox("Enable time range filter", value=True)

# -------------------------
# Create _time for plotting
# -------------------------
df_plot = df.copy()
time_kind = "step"  # or "datetime"

if time_col_choice != "<Index order>":
    # Try to parse the selected time column as datetime
    df_plot["_time"] = pd.to_datetime(df_plot[time_col_choice], errors="coerce")
    time_kind = "datetime"
    lost = df_plot["_time"].isna().sum()
    if lost > 0:
        st.warning(f'In the time column "{time_col_choice}", {lost} rows could not be parsed as datetime and were temporarily dropped.')
        df_plot = df_plot.dropna(subset=["_time"]).copy()
    df_plot = df_plot.sort_values("_time").reset_index(drop=True)
else:
    df_plot["_time"] = np.arange(len(df_plot))
    time_kind = "step"

# Select columns for plotting
plot_candidates = []
if target_choice != "<Not selected>" and target_choice in df_plot.columns:
    plot_candidates.append(target_choice)
plot_candidates.extend([c for c in feature_cols if c in df_plot.columns])

# Keep only numeric columns for plotting (non-numeric will be ignored with a note)
numeric_plot_cols = [c for c in plot_candidates if pd.api.types.is_numeric_dtype(df_plot[c])]
non_numeric = [c for c in plot_candidates if c not in numeric_plot_cols]
if non_numeric:
    st.info("The following columns are non-numeric and were excluded from the line chart: " + ", ".join(non_numeric))

if not numeric_plot_cols:
    st.error("No numeric columns to plot. Please choose at least one numeric column in the sidebar.")
    st.stop()

# Optional resampling (only when using a datetime time column)
if time_kind == "datetime" and freq_map[resample_choice]:
    df_plot = (
        df_plot.set_index("_time")[numeric_plot_cols]
        .resample(freq_map[resample_choice])
        .mean()
        .reset_index()
    )
else:
    # Keep only columns needed for plotting to reduce melt overhead
    df_plot = df_plot[["_time"] + numeric_plot_cols]

# Time range filtering
if range_filter_on:
    if time_kind == "datetime":
        t_min, t_max = df_plot["_time"].min(), df_plot["_time"].max()
        start, end = st.slider("Select time range", min_value=t_min, max_value=t_max, value=(t_min, t_max))
        df_plot = df_plot[(df_plot["_time"] >= start) & (df_plot["_time"] <= end)]
    else:
        i_min, i_max = int(df_plot["_time"].min()), int(df_plot["_time"].max())
        start, end = st.slider("Select step range", min_value=i_min, max_value=i_max, value=(i_min, i_max))
        df_plot = df_plot[(df_plot["_time"] >= start) & (df_plot["_time"] <= end)]

# Scaling/normalization (only affects numeric_plot_cols)
df_plot, plotted_cols = normalize_columns(df_plot, numeric_plot_cols, scale_method)

# Long format
df_long = df_plot.melt(id_vars=["_time"], value_vars=plotted_cols, var_name="Variable", value_name="Value")

# -------------------------
# Selection summary + chart
# -------------------------
with st.expander("Selection Summary", expanded=False):
    st.write(
        {
            "Time column": None if time_col_choice == "<Index order>" else time_col_choice,
            "Ignored columns": exclude_cols,
            "Target": None if target_choice == "<Not selected>" else target_choice,
            "Features": feature_cols,
            "Plotted columns": plotted_cols,
            "Resampling": resample_choice,
            "Scaling": scale_method,
        }
    )

st.subheader("Time-Series Line Chart")
st.altair_chart(build_line_chart(df_long, time_kind), use_container_width=True)

# -------------------------
# Extras: missingness & simple stats
# -------------------------
st.subheader("Simple Data Quality Overview")
q_cols = list(dict.fromkeys((target_choice if target_choice != "<Not selected>" else [], *feature_cols)))
q_cols = [c for c in q_cols if c in df.columns]
if q_cols:
    miss = df[q_cols].isna().sum().rename("Missing count")
    dtype = df[q_cols].dtypes.rename("Dtype")
    st.dataframe(pd.concat([dtype, miss], axis=1))
else:
    st.write("Please select columns in the sidebar first.")
