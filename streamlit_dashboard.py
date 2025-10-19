# app.py — Streamlit Data Visualization Dashboard
# ---------------------------------------------------------------------------------
# Features
# - Upload CSV/Parquet or load a built‑in demo dataset
# - Interactive filters (numeric range sliders, categorical multiselect)
# - Distribution plots: histograms, KDE, box/violin
# - Correlation heatmap (Pearson/Spearman/Kendall)
# - Pairwise scatter (color/size by columns)
# - Missingness overview
# - Download filtered data
# - Works great with your AutoClean output CSV
# ---------------------------------------------------------------------------------

# ---------------------
# Imports
# ---------------------
import io  # lightweight in‑memory buffers for downloads
from typing import List, Tuple

import numpy as np  # numerical ops
import pandas as pd  # tabular data ops
import streamlit as st  # the dashboard framework
import plotly.express as px  # high‑level interactive plots
import plotly.graph_objects as go  # lower‑level plot primitives
from pathlib import Path  # file extension handling for uploads


# ---------------------
# Page config (title, layout)
# ---------------------
st.set_page_config(
    page_title="DataViz Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Small helper to avoid Streamlit deprecation noise
# Some Streamlit versions removed this option; make it a no-op if unsupported
try:
    st.set_option("deprecation.showPyplotGlobalUse", False)
except Exception:
    pass


# ---------------------
# Utility: demo dataset (synthetic House Prices‑like)
# ---------------------
def make_demo_df(n: int = 400) -> pd.DataFrame:
    """Create a compact synthetic dataset with numeric/categorical mix and a target."""
    rng = np.random.default_rng(42)
    neighborhoods = np.array(["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]) 
    house_styles = np.array(["1Story", "2Story", "1.5Fin", "SLvl"]) 

    df = pd.DataFrame({
        "LotArea": rng.integers(2000, 15000, size=n).astype(float),
        "OverallQual": rng.integers(3, 10, size=n).astype(float),
        "YearBuilt": rng.integers(1950, 2010, size=n).astype(float),
        "GrLivArea": rng.integers(600, 3500, size=n).astype(float),
        "GarageCars": rng.integers(0, 4, size=n).astype(float),
        "Neighborhood": rng.choice(neighborhoods, size=n),
        "HouseStyle": rng.choice(house_styles, size=n),
        "CentralAir": rng.choice(np.array(["Y", "N"]), size=n),
    })
    # inject some missing values
    for col in ["LotArea", "OverallQual", "GrLivArea"]:
        df.loc[rng.choice(n, size=int(0.05*n), replace=False), col] = np.nan
    for col in ["Neighborhood", "HouseStyle"]:
        df.loc[rng.choice(n, size=int(0.03*n), replace=False), col] = np.nan
    # create a target with noise
    price = (
        50000
        + df["GrLivArea"].fillna(df["GrLivArea"].median()) * 120
        + (df["OverallQual"].fillna(df["OverallQual"].median()) - 5) * 9000
        + df["GarageCars"].fillna(0) * 4500
        + (df["CentralAir"] == "Y").astype(float) * 7000
    )
    df["SalePrice"] = (price + rng.normal(0, 20000, n)).round(0)
    return df


# ---------------------
# Caching helpers (avoid recomputation while interacting)
# ---------------------
@st.cache_data(show_spinner=False)
def load_table(upload: bytes, ext: str) -> pd.DataFrame:
    """Read CSV/Parquet from uploaded bytes based on extension."""
    if ext == ".csv":
        return pd.read_csv(io.BytesIO(upload))
    elif ext in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(io.BytesIO(upload))
        except Exception:
            # Fallback: mislabelled CSV uploaded as parquet
            return pd.read_csv(io.BytesIO(upload), engine="python", encoding_errors="replace")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def split_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols)."""
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat


# ---------------------
# Sidebar — data source & global controls
# ---------------------
st.sidebar.header("1) Data source")

uploader = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet", "pq"])
use_demo = st.sidebar.toggle("Use demo dataset", value=not bool(uploader))

if uploader and not use_demo:
    name = uploader.name or "uploaded_file"
    ext = Path(name).suffix.lower() or ".csv"
    df_raw = load_table(uploader.getvalue(), ext)
else:
    df_raw = make_demo_df()

st.sidebar.header("2) Basic settings")
all_columns = df_raw.columns.tolist()
num_cols, cat_cols = split_cols(df_raw)

# choose target for grouping/plots (optional)
target_col = st.sidebar.selectbox("Target column (optional)", ["<none>"] + all_columns, index=len(all_columns) if "SalePrice" in all_columns else 0)
if target_col == "<none>":
    target_col = None

# correlation method
corr_method = st.sidebar.selectbox("Correlation method", ["pearson", "spearman", "kendall"], index=0)

# ---------------------
# Sidebar — interactive filters
# ---------------------
st.sidebar.header("3) Filters")

# Start from a working copy
filtered = df_raw.copy()

# Numeric filters: range sliders per selected column
with st.sidebar.expander("Numeric filters", expanded=False):
    choose_num = st.multiselect("Select numeric columns to filter", num_cols, default=num_cols[: min(3, len(num_cols))])
    for col in choose_num:
        col_min = float(np.nanmin(filtered[col])) if filtered[col].notna().any() else 0.0
        col_max = float(np.nanmax(filtered[col])) if filtered[col].notna().any() else 1.0
        lo, hi = st.slider(f"{col}", min_value=col_min, max_value=col_max, value=(col_min, col_max))
        mask = filtered[col].between(lo, hi) | filtered[col].isna()
        filtered = filtered[mask]

# Categorical filters: multiselect per selected column
with st.sidebar.expander("Categorical filters", expanded=False):
    choose_cat = st.multiselect("Select categorical columns to filter", cat_cols, default=cat_cols[: min(2, len(cat_cols))])
    for col in choose_cat:
        levels = sorted([x for x in filtered[col].dropna().unique()])
        picks = st.multiselect(f"{col}", levels, default=levels)
        if picks:
            filtered = filtered[filtered[col].isin(picks) | filtered[col].isna()]

# Download button for filtered data
st.sidebar.header("4) Export")
if st.sidebar.button("Download filtered CSV"):
    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Save filtered.csv", data=csv_bytes, file_name="filtered.csv", mime="text/csv")


# ---------------------
# Main — KPIs and table peek
# ---------------------
st.title("📊 Data Visualization Dashboard")
st.caption("Interactive EDA with histograms, box/violin, correlation heatmap, and more.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Rows", f"{len(filtered):,}")
with c2:
    st.metric("Columns", f"{filtered.shape[1]:,}")
with c3:
    st.metric("Numeric", f"{len(num_cols):,}")
with c4:
    st.metric("Categorical", f"{len(cat_cols):,}")

st.dataframe(filtered.head(50), use_container_width=True)


# ---------------------
# Distributions — Histograms & Density
# ---------------------
st.subheader("Distributions")
col_left, col_right = st.columns([2, 1])

with col_left:
    x_col = st.selectbox("Numeric column", options=num_cols, index=0 if num_cols else None, key="hist_x")
    hue_col = st.selectbox("Color by (optional)", options=[None] + cat_cols + ([target_col] if target_col and target_col in cat_cols else []), index=0, key="hist_hue")
    nbins = st.slider("Bins", min_value=5, max_value=80, value=30)
    if x_col:
        fig = px.histogram(filtered, x=x_col, color=hue_col, nbins=nbins, marginal="box", opacity=0.8)
        fig.update_layout(height=420, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

with col_right:
    if x_col:
        # Empirical CDF / KDE quick view via histogram with cumulative
        fig2 = px.histogram(filtered, x=x_col, nbins=60, histnorm="probability density")
        fig2.update_traces(cumulative_enabled=False)
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)


# ---------------------
# Box / Violin by Category
# ---------------------
st.subheader("Box / Violin by Category")
col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    y_num = st.selectbox("Numeric (Y)", options=num_cols, index=0 if num_cols else None, key="bx_num")
with col_b:
    x_cat = st.selectbox("Category (X)", options=cat_cols, index=0 if cat_cols else None, key="bx_cat")
with col_c:
    mode = st.radio("Mode", ["box", "violin"], index=0, horizontal=False)

if y_num and x_cat:
    fig = px.box(filtered, x=x_cat, y=y_num, points="outliers") if mode == "box" else px.violin(filtered, x=x_cat, y=y_num, box=True, points="outliers")
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------
# Correlation Heatmap
# ---------------------
st.subheader("Correlation heatmap")
num_only = filtered.select_dtypes(include=[np.number])
if not num_only.empty:
    corr = num_only.corr(method=corr_method)
    heat = px.imshow(corr, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r", origin="lower")
    heat.update_layout(height=600)
    st.plotly_chart(heat, use_container_width=True)
else:
    st.info("No numeric columns available for correlation heatmap.")


# ---------------------
# Pairwise scatter (2D) with color/size encodings
# ---------------------
st.subheader("Pairwise scatter")
sc1, sc2, sc3 = st.columns(3)
with sc1:
    x_sc = st.selectbox("X", options=num_cols, index=0 if num_cols else None, key="sc_x")
with sc2:
    y_sc = st.selectbox("Y", options=num_cols, index=1 if len(num_cols) > 1 else 0, key="sc_y")
with sc3:
    color_sc = st.selectbox("Color by", options=[None] + cat_cols + ([target_col] if target_col and target_col in cat_cols else []), index=0, key="sc_color")
size_sc = st.select_slider("Point size", options=[4, 6, 8, 10, 12, 14], value=8)

if x_sc and y_sc and x_sc != y_sc:
    fig_sc = px.scatter(filtered, x=x_sc, y=y_sc, color=color_sc, opacity=0.75)
    fig_sc.update_traces(marker=dict(size=size_sc))
    fig_sc.update_layout(height=520)
    st.plotly_chart(fig_sc, use_container_width=True)


# ---------------------
# Missingness overview
# ---------------------
st.subheader("Missingness")
miss = filtered.isna().mean().sort_values(ascending=False)
miss_df = miss.reset_index()
miss_df.columns = ["column", "missing_rate"]
if miss_df["missing_rate"].sum() > 0:
    fig_miss = px.bar(miss_df.head(30), x="column", y="missing_rate")
    fig_miss.update_layout(height=420, xaxis_tickangle=-45, yaxis_tickformat=",.0%")
    st.plotly_chart(fig_miss, use_container_width=True)
else:
    st.success("No missing values detected.")


# ---------------------
# Target‑aware quick view (if target numeric)
# ---------------------
if target_col and target_col in filtered.columns and np.issubdtype(filtered[target_col].dtype, np.number):
    st.subheader("Target vs Feature (quick regression view)")
    t1, t2 = st.columns(2)
    with t1:
        feat = st.selectbox("Feature", options=[c for c in num_cols if c != target_col], index=0 if len(num_cols) > 1 else None)
    with t2:
        trend = st.toggle("Show LOESS‑like trend (rolling mean)", value=True)
    if feat:
        fig_t = px.scatter(filtered, x=feat, y=target_col, opacity=0.7)
        if trend:
            # simple rolling mean line for quick trend feel
            tmp = filtered[[feat, target_col]].dropna().sort_values(feat)
            if len(tmp) > 5:
                win = max(5, int(0.05 * len(tmp)))
                tmp["smooth"] = tmp[target_col].rolling(win, min_periods=1).mean()
                fig_t.add_trace(go.Scatter(x=tmp[feat], y=tmp["smooth"], mode="lines", name="trend"))
        fig_t.update_layout(height=520)
        st.plotly_chart(fig_t, use_container_width=True)


# ---------------------
# Footer
# ---------------------
st.caption("Built with Streamlit + Plotly. Drop in your AutoClean processed CSV for instant EDA.")
