# -*- coding: utf-8 -*-
"""
Streamlit Time Series Explorer + Optional Dynamic Graph (single file)

What this app does
------------------
1) **Upload the time‑series CSV** (required). Used for column selection, stats, and line charts.
2) **Optionally upload a graph CSV** (edge list). Format can be either:
   - `time,source,target`  (dynamic graph over time), or
   - `source,target`       (no time column -> the app auto‑adds a constant `time = 1`, i.e., a single‑snapshot dynamic graph).
3) If a graph CSV is provided, the app renders an **animated undirected network** (Plotly) with Play/Pause and a time slider.

Run:
  pip install streamlit pandas altair numpy plotly networkx
  streamlit run app.py
"""

import io
from typing import List, Tuple, Optional, Sequence, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Graph deps
import plotly.graph_objects as go
import networkx as nx

# Page config
st.set_page_config(page_title="Time Series + Dynamic Graph", layout="wide")
alt.data_transformers.disable_max_rows()

# =========================
# Utilities
# =========================
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


# =========================
# Graph animation helpers
# =========================

def _coerce_time(series: pd.Series) -> Tuple[pd.Series, str]:
    """Try to coerce to datetime. If <50% succeed, try numeric; otherwise treat as category.
    Returns (coerced_series, kind), where kind in {"datetime", "category", "numeric"}.
    """
    s = series.copy()
    s_dt = pd.to_datetime(s, errors="coerce")
    success_ratio = s_dt.notna().mean() if len(s_dt) else 0
    if success_ratio >= 0.5:
        return s_dt, "datetime"
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.notna().mean() >= 0.8:
        return s_num, "numeric"
    return s, "category"


def _sorted_unique(values: pd.Series, kind: str) -> List[Any]:
    if kind in ("datetime", "numeric"):
        u = pd.Series(values.unique())
        return sorted(u.dropna().tolist())
    # category: preserve first appearance order
    seen = set()
    order = []
    for v in values:
        if pd.isna(v):
            continue
        if v not in seen:
            seen.add(v)
            order.append(v)
    return order


def _edge_list_from_df(df_edges: pd.DataFrame, source_col: str, target_col: str) -> List[Tuple[str, str]]:
    """Return undirected edge list with (u,v) sorted and without self-loops."""
    edges = []
    for u, v in df_edges[[source_col, target_col]].itertuples(index=False):
        if pd.isna(u) or pd.isna(v):
            continue
        u = str(u)
        v = str(v)
        if u == v:
            continue
        a, b = (u, v) if u <= v else (v, u)
        edges.append((a, b))
    return edges


def _build_positions(all_nodes: Sequence[str], all_edges: Sequence[Tuple[str, str]]) -> Dict[str, Tuple[float, float]]:
    G = nx.Graph()
    G.add_nodes_from(all_nodes)
    G.add_edges_from(all_edges)
    if len(G) == 0:
        return {}
    pos = nx.spring_layout(G, seed=42)
    return pos


def _edge_trace_from_edges(edges: Sequence[Tuple[str, str]], pos: Dict[str, Tuple[float, float]]) -> go.Scatter:
    xs, ys = [], []
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(width=1), hoverinfo="none", name="edges")


def _node_trace_for_frame(nodes: Sequence[str], edges: Sequence[Tuple[str, str]], pos: Dict[str, Tuple[float, float]], show_labels: bool) -> go.Scatter:
    deg = {n: 0 for n in nodes}
    for u, v in edges:
        if u in deg:
            deg[u] += 1
        if v in deg:
            deg[v] += 1
    x = [pos[n][0] for n in nodes]
    y = [pos[n][1] for n in nodes]
    sizes = [10 + 4 * np.sqrt(deg.get(n, 0)) for n in nodes]
    texts = nodes if show_labels else [f"deg={deg.get(n, 0)}" for n in nodes]
    opacities = [1.0 if deg.get(n, 0) > 0 else 0.25 for n in nodes]
    return go.Scatter(
        x=x,
        y=y,
        mode="markers+text" if show_labels else "markers",
        text=nodes if show_labels else None,
        textposition="top center",
        marker=dict(size=sizes, opacity=opacities),
        hovertext=texts,
        hoverinfo="text",
        name="nodes",
    )


def build_network_animation(
    df_edges: pd.DataFrame,
    source_col: str,
    target_col: str,
    time_col: Optional[str],  # None => create constant time
    cumulative: bool,
    show_labels: bool,
    max_frames: int = 150,
    height: int = 650,
) -> Optional[go.Figure]:
    """Create an animated Plotly network from an edge stream (undirected).
    If time_col is None, a constant time value (1) will be added.
    Returns a go.Figure or None if cannot build.
    """
    for c in [source_col, target_col]:
        if c not in df_edges.columns:
            return None

    edges_df = df_edges[[source_col, target_col]].copy()
    if edges_df.empty:
        return None

    if time_col is not None and time_col in df_edges.columns:
        edges_df["_time_raw"] = df_edges[time_col]
    else:
        edges_df["_time_raw"] = 1  # single snapshot

    # Coerce time & determine ordering
    time_series, time_kind = _coerce_time(edges_df["_time_raw"])
    edges_df = edges_df.assign(_time=time_series)

    # Sort by time if numeric/datetime
    if time_kind in ("datetime", "numeric"):
        edges_df = edges_df.sort_values(["_time"]).reset_index(drop=True)

    # List of time points for frames
    unique_times = _sorted_unique(edges_df["_time"], time_kind)
    if len(unique_times) == 0:
        return None

    # Downsample frames if too many
    if len(unique_times) > max_frames:
        idxs = np.linspace(0, len(unique_times) - 1, num=max_frames, dtype=int)
        unique_times = [unique_times[i] for i in idxs]

    # Nodes & full edge set (for layout)
    all_edges_full = _edge_list_from_df(edges_df.assign(_t=1), source_col, target_col)
    all_nodes: List[str] = sorted(list({s for e in all_edges_full for s in e}))
    pos = _build_positions(all_nodes, all_edges_full)
    if not pos:
        return None

    # Helper to compute edges per time value
    def edges_for_t(t_val):
        dft = edges_df[edges_df["_time"] == t_val]
        ed = _edge_list_from_df(dft, source_col, target_col)
        return sorted(set(ed))

    frames = []
    cumulative_edges: List[Tuple[str, str]] = []

    # Base frame
    first_time = unique_times[0]
    edges_t = edges_for_t(first_time)
    if cumulative:
        cumulative_edges = edges_t.copy()
        edges_display = cumulative_edges
    else:
        edges_display = edges_t

    base_edge_trace = _edge_trace_from_edges(edges_display, pos)
    base_node_trace = _node_trace_for_frame(all_nodes, edges_display, pos, show_labels)

    for t in unique_times:
        et = edges_for_t(t)
        if cumulative:
            merged = {e: None for e in cumulative_edges}
            for e in et:
                merged[e] = None
            cumulative_edges = list(merged.keys())
            show_edges = cumulative_edges
        else:
            show_edges = et
        e_trace = _edge_trace_from_edges(show_edges, pos)
        n_trace = _node_trace_for_frame(all_nodes, show_edges, pos, show_labels)
        frames.append(go.Frame(data=[e_trace, n_trace], name=str(t)))

    layout = go.Layout(
        title=dict(text=f"Dynamic Graph — {'Cumulative' if cumulative else 'Instant'}", x=0.5, xanchor="center"),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.08,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"r": 10, "t": 0},
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 400, "redraw": True}, "transition": {"duration": 0}}]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "x": 0.05,
                "y": 1.04,
                "currentvalue": {"prefix": "Time: ", "visible": True, "xanchor": "left"},
                "pad": {"t": 0},
                "len": 0.9,
                "steps": [
                    {"args": [[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}], "label": str(f.name), "method": "animate"}
                    for f in frames
                ],
            }
        ],
    )

    fig = go.Figure(data=[base_edge_trace, base_node_trace], layout=layout, frames=frames)
    return fig


# =========================
# Sidebar: uploads & config
# =========================
st.sidebar.header("① Upload Time Series CSV (Required)")
file_ts = st.sidebar.file_uploader("Select time series CSV (for trend charts/column selection)", type=["csv"], key="ts")

st.sidebar.header("② Upload Graph Structure CSV (Optional)")
st.sidebar.caption("Supports two formats: `time,source,target` or `source,target`. If no time column, will auto-add constant time=1.")
file_graph = st.sidebar.file_uploader("Select graph structure CSV (optional)", type=["csv"], key="graph")

if file_ts is None:
    st.title("Time Series + Dynamic Graph Explorer")
    st.info("Please first upload **Time Series CSV** in the sidebar. After uploading, you can optionally upload **Graph Structure CSV** to generate dynamic graphs.")
    st.stop()

# Read time-series CSV
content_ts = file_ts.getvalue()
df_ts, enc_ts = load_csv_file(content_ts)

st.sidebar.success(f"Time series loaded (encoding: {enc_ts}, shape: {df_ts.shape[0]}×{df_ts.shape[1]})")

# Read optional graph CSV
graph_info = None
if file_graph is not None:
    try:
        content_graph = file_graph.getvalue()
        df_graph, enc_graph = load_csv_file(content_graph)
        st.sidebar.success(f"Graph structure loaded (encoding: {enc_graph}, shape: {df_graph.shape[0]}×{df_graph.shape[1]})")
        graph_info = (df_graph, enc_graph)
    except Exception as e:
        st.sidebar.error(f"Graph structure file loading failed: {e}")
        graph_info = None

all_cols_ts = list(df_ts.columns)
if not all_cols_ts:
    st.error("No columns detected, please check the file.")
    st.stop()

# =========================
# Column selection for time-series
# =========================
st.sidebar.header("③ Column Selection (Time Series)")
time_col_choice = st.sidebar.selectbox(
    "Time column (optional; defaults to index order)",
    options=["<Index order>"] + all_cols_ts,
    index=0,
)

exclude_cols = st.sidebar.multiselect("Columns to ignore", options=all_cols_ts)

avail_cols = [c for c in all_cols_ts if c not in exclude_cols]
if not avail_cols:
    st.warning("All columns have been ignored, please keep at least one column.")

target_choice = st.sidebar.selectbox(
    "Prediction target",
    options=["<Not selected>"] + avail_cols,
    index=0,
)

# Default features: all available columns except target
default_feats = [c for c in avail_cols if c != target_choice]
feature_cols = st.sidebar.multiselect(
    "Feature columns",
    options=avail_cols,
    default=default_feats[: min(len(default_feats), 8)],
)

# Clean mutual exclusions
feature_cols = [c for c in feature_cols if c != target_choice and c not in exclude_cols]

# Visualization settings
st.sidebar.header("④ Trend Chart Settings")
scale_method = st.sidebar.selectbox("Scaling/Normalization", ["None", "Z-score standardization", "Min-Max normalization"], index=0)

freq_map = {"No resampling": None, "Daily (D)": "D", "Weekly (W)": "W", "Monthly (M)": "M", "Yearly (A)": "A"}
resample_choice = st.sidebar.selectbox("Resampling frequency", list(freq_map.keys()), index=0)

range_filter_on = st.sidebar.checkbox("Enable time range filtering", value=True)

# =========================
# Build time-series chart (non-blocking if no numeric columns)
# =========================
df_plot = df_ts.copy()
time_kind = "step"

if time_col_choice != "<Index order>":
    df_plot["_time"] = pd.to_datetime(df_plot[time_col_choice], errors="coerce")
    time_kind = "datetime"
    lost = df_plot["_time"].isna().sum()
    if lost > 0:
        st.warning(f'Time column "{time_col_choice}" has {lost} rows that cannot be parsed as datetime, these rows have been temporarily dropped.')
        df_plot = df_plot.dropna(subset=["_time"]).copy()
    df_plot = df_plot.sort_values("_time").reset_index(drop=True)
else:
    df_plot["_time"] = np.arange(len(df_plot))
    time_kind = "step"

plot_candidates = []
if target_choice != "<Not selected>" and target_choice in df_plot.columns:
    plot_candidates.append(target_choice)
plot_candidates.extend([c for c in feature_cols if c in df_plot.columns])

numeric_plot_cols = [c for c in plot_candidates if pd.api.types.is_numeric_dtype(df_plot[c])]
non_numeric = [c for c in plot_candidates if c not in numeric_plot_cols]
if non_numeric:
    st.info("The following non-numeric columns are excluded from line charts: " + ", ".join(non_numeric))

if time_kind == "datetime" and freq_map[resample_choice]:
    if numeric_plot_cols:
        df_plot = (
            df_plot.set_index("_time")[numeric_plot_cols]
            .resample(freq_map[resample_choice])
            .mean()
            .reset_index()
        )
    else:
        df_plot = df_plot[["_time"]]
else:
    df_plot = df_plot[["_time"] + numeric_plot_cols] if numeric_plot_cols else df_plot[["_time"]]

if range_filter_on:
    if time_kind == "datetime" and not df_plot.empty:
        t_min, t_max = df_plot["_time"].min(), df_plot["_time"].max()
        start, end = st.slider("Select time range", min_value=t_min, max_value=t_max, value=(t_min, t_max))
        df_plot = df_plot[(df_plot["_time"] >= start) & (df_plot["_time"] <= end)]
    elif time_kind != "datetime" and not df_plot.empty:
        i_min, i_max = int(df_plot["_time"].min()), int(df_plot["_time"].max())
        start, end = st.slider("Select step range", min_value=i_min, max_value=i_max, value=(i_min, i_max))
        df_plot = df_plot[(df_plot["_time"] >= start) & (df_plot["_time"] <= end)]

# Scaling only if we actually have numeric columns
plotted_cols: List[str] = []
if numeric_plot_cols:
    df_plot, plotted_cols = normalize_columns(df_plot, numeric_plot_cols, scale_method)
    df_long = df_plot.melt(id_vars=["_time"], value_vars=plotted_cols, var_name="Variable", value_name="Value")
else:
    df_long = pd.DataFrame(columns=["_time", "Variable", "Value"])  # empty -> skip chart

with st.expander("Selection Summary", expanded=False):
    st.write(
        {
            "Time column": None if time_col_choice == "<Index order>" else time_col_choice,
            "Ignored columns": exclude_cols,
            "Target": None if target_choice == "<Not selected>" else target_choice,
            "Features": feature_cols,
            "Plot columns": plotted_cols,
            "Resampling": resample_choice,
            "Scaling": scale_method,
        }
    )

# =========================
# Time Series Line Chart
# =========================
st.subheader("Time Series Line Chart")
if not df_long.empty:
    st.altair_chart(build_line_chart(df_long, time_kind), use_container_width=True)
else:
    st.info("No numeric columns selected, trend chart skipped.")

# =========================
# Optional: Dynamic Graph
# =========================
st.markdown("---")
st.subheader("Dynamic Graph Network (Optional Graph Structure CSV)")

if graph_info is None:
    st.info("For dynamic graphs, please upload graph structure CSV in the second upload box in the sidebar. Supports `time,source,target` or `source,target`. If no time column, will auto-add constant `time = 1`.")
else:
    df_graph, _enc = graph_info

    # Auto-detect columns
    graph_cols = list(df_graph.columns)
    cols_lower = [c.lower() for c in graph_cols]

    def guess_col(name: str, fallback_idx: Optional[int]) -> Optional[str]:
        if name in cols_lower:
            return graph_cols[cols_lower.index(name)]
        if fallback_idx is not None and 0 <= fallback_idx < len(graph_cols):
            return graph_cols[fallback_idx]
        return None

    # Prefer conventional names; otherwise positional defaults
    guessed_time = guess_col("time", None)  # None => may create constant later
    guessed_source = guess_col("source", 0)
    guessed_target = guess_col("target", 1 if len(graph_cols) >= 2 else None)

    with st.expander("Configure Network Animation", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            # Allow "<None (constant 1)>" when no time col exists
            time_options = ["<None (constant 1)>"] + graph_cols
            default_time_index = 0 if guessed_time is None else (graph_cols.index(guessed_time) + 1)
            g_time_sel = st.selectbox("Time column", options=time_options, index=default_time_index, help="If <None> is selected, all edges will be assigned the same time.")
        with col_b:
            g_source_sel = st.selectbox("Source column", options=graph_cols, index=graph_cols.index(guessed_source) if guessed_source in graph_cols else 0)
        with col_c:
            tgt_default_idx = graph_cols.index(guessed_target) if guessed_target in graph_cols else (1 if len(graph_cols) > 1 else 0)
            g_target_sel = st.selectbox("Target column", options=graph_cols, index=tgt_default_idx)

        col_opts1, col_opts2, col_opts3 = st.columns([1, 1, 1])
        with col_opts1:
            g_cumulative = st.radio("Edge display mode", ["Instant (show current time only)", "Cumulative (from start to current)"], index=1)
            g_cumulative_flag = g_cumulative.startswith("Cumulative")
        with col_opts2:
            g_show_labels = st.checkbox("Show node labels", value=False)
        with col_opts3:
            g_max_frames = st.slider("Max frames", min_value=10, max_value=500, value=150, step=10, help="Too many frames will affect performance, excess will be automatically downsampled.")

    time_col_for_build = None if g_time_sel.startswith("<None") else g_time_sel

    try:
        fig_net = build_network_animation(
            df_edges=df_graph,
            source_col=g_source_sel,
            target_col=g_target_sel,
            time_col=time_col_for_build,
            cumulative=g_cumulative_flag,
            show_labels=g_show_labels,
            max_frames=g_max_frames,
            height=600,
        )
        if fig_net is None:
            st.info("Please confirm Source/Target column selection is correct; if no Time column, you can select \"<None (constant 1)>\".")
        else:
            st.plotly_chart(fig_net, use_container_width=True, config={"displayModeBar": True})
            with st.expander("Example CSV (with time)", expanded=False):
                st.code(
                    """time,source,target\n1,0,0\n1,0,1\n1,0,2\n1,0,3\n1,0,4\n1,0,5\n1,0,6\n1,0,7\n1,0,8\n1,0,9\n1,0,10\n1,46,36\n1,46,37\n1,46,39\n1,46,41\n1,46,42\n1,46,43\n1,46,44\n1,46,45\n1,46,46\n2,0,0\n2,0,1\n2,0,2\n2,0,3\n2,0,4\n2,0,5\n2,0,6\n2,0,7\n2,0,8\n2,0,10\n2,0,11\n""",
                    language="csv",
                )
            with st.expander("Example CSV (without time, will auto-set to 1)", expanded=False):
                st.code("""source,target\nA,B\nA,C\nB,C\n""", language="csv")
    except Exception as e:
        st.warning(f"Dynamic graph generation failed: {e}")

# =========================
# Data Previe
# =========================
st.markdown("---")
st.subheader("Data Preview (Time Series)")
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.dataframe(df_ts.head(50), use_container_width=True)
with col2:
    st.metric("Rows", f"{df_ts.shape[0]:,}")
with col3:
    st.metric("Columns", f"{df_ts.shape[1]:,}")

# =========================
# Simple Data Quality Overview
# =========================
st.subheader("Data Quality Overview")
q_cols = ([target_choice] if target_choice != "<Not selected>" else []) + feature_cols
q_cols = [c for c in q_cols if c in df_ts.columns]
if q_cols:
    miss = df_ts[q_cols].isna().sum().rename("Missing count")
    dtype = df_ts[q_cols].dtypes.rename("Dtype")
    st.dataframe(pd.concat([dtype, miss], axis=1))
else:
    st.write("Please first select columns in the sidebar.")

