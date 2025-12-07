# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="School Performance – Entrant & Subject Visualisations",
    layout="wide",
)

# -------------------------------------------------------------------
# Paths & data loading
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

DATASETS = {
    "Count by Occupation": BASE_DIR / "Count by Occupation.xlsx",
    "Count by Level of Qualification": BASE_DIR / "Count by Level of Qualification.xlsx",
    "HECoS Data": BASE_DIR / "HECoS Data.xlsx",
}


@st.cache_data
def load_excel(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


# -------------------------------------------------------------------
# Helper for safe filters
# -------------------------------------------------------------------
def multiselect_filter(df, col, selected):
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)]


# -------------------------------------------------------------------
# App layout
# -------------------------------------------------------------------
st.title("Entrant & Subject Visualisations Dashboard")

st.markdown(
    """
This dashboard reads the pre-cleaned analysis files and produces visualisations.
The tabs below cover:

- **Count by Occupation** (socio-economic classification, parental education, IMD)
- **Count by Level of Qualification** (time series of entrants by level)
- **HECoS Subject Data** (subject-level entrants by CAH grouping)
- **Custom visualisations file** – upload any CSV/Excel (e.g. `visualizations.csv`)
  and build your own charts.
"""
)

# Load data up front (with basic error handling)
dfs = {}
for name, path in DATASETS.items():
    try:
        dfs[name] = load_excel(path)
    except Exception as e:
        st.sidebar.error(f"Could not load {name}: {e}")

tab_occ, tab_level, tab_hecos, tab_custom = st.tabs(
    [
        "Count by Occupation",
        "Count by Level of Qualification",
        "HECoS Subject Data",
        "Custom visualisations file",
    ]
)

# -------------------------------------------------------------------
# TAB 1 – Count by Occupation
# -------------------------------------------------------------------
with tab_occ:
    st.header("Count by Occupation (socio-economic classification, IMD, etc.)")

    df = dfs.get("Count by Occupation")
    if df is None:
        st.error("Could not load **Count by Occupation.xlsx**.")
    else:
        # Sidebar-like filters within the tab
        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            cat_marker_vals = sorted(df["Category Marker"].dropna().unique())
            cat_marker = col1.multiselect(
                "Category marker", options=cat_marker_vals, default=cat_marker_vals
            )

            he_country_vals = sorted(df["Country of HE provider"].dropna().unique())
            he_country = col2.multiselect(
                "Country of HE provider",
                options=he_country_vals,
                default=["England"],
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col3.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["All"],
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col4.multiselect(
                "Level of study",
                options=level_vals,
                default=["All", "First degree"],
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Category Marker", cat_marker)
        df_f = multiselect_filter(df_f, "Country of HE provider", he_country)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_marker)
        df_f = multiselect_filter(df_f, "Level of study", level_of_study)
        if years:
            df_f = df_f[df_f["Academic Year"].isin(years)]

        metric = st.radio("Metric", ["Number", "Percentage"], horizontal=True)
        if metric == "Percentage":
            df_f = df_f.dropna(subset=["Percentage"])

        st.markdown("### Visualisations")

        view_type = st.radio(
            "View",
            ["By Category (latest year)", "Trend over time (by category)"],
            horizontal=True,
        )

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            # Aggregate
            metric_col = metric
            agg = (
                df_f.groupby(["Academic Year", "Category"], as_index=False)[metric_col]
                .sum()
            )

            if view_type == "By Category (latest year)":
                latest_year = sorted(agg["Academic Year"].unique())[-1]
                latest = agg[agg["Academic Year"] == latest_year]
                latest = latest.sort_values(metric_col, ascending=False)

                st.caption(f"Latest year in filtered data: **{latest_year}**")
                st.bar_chart(
                    data=latest.set_index("Category")[metric_col],
                    use_container_width=True,
                )

            else:  # Trend over time
                pivot = agg.pivot(
                    index="Academic Year", columns="Category", values=metric_col
                ).sort_index()
                st.line_chart(pivot, use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 2 – Count by Level of Qualification
# -------------------------------------------------------------------
with tab_level:
    st.header("Count by Level of Qualification")

    df = dfs.get("Count by Level of Qualification")
    if df is None:
        st.error("Could not load **Count by Level of Qualification.xlsx**.")
    else:
        with st.expander("Filters", expanded=True):
            col1, col2 = st.columns(2)

            level_vals = sorted(df["Level of qualification"].dropna().unique())
            level_sel = col1.multiselect(
                "Level of qualification",
                options=level_vals,
                default=[lv for lv in level_vals if lv != "Total"] or level_vals,
            )

            year_vals = sorted(df["Academic year"].dropna().unique())
            years = col2.multiselect(
                "Academic year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Level of qualification", level_sel)
        if years:
            df_f = df_f[df_f["Academic year"].isin(years)]

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            st.markdown("### Time series of entrants by level of qualification")

            # Pivot for time series: index = year, columns = level, values = Number
            pivot = (
                df_f.pivot(
                    index="Academic year",
                    columns="Level of qualification",
                    values="Number",
                )
                .sort_index()
            )

            st.line_chart(pivot, use_container_width=True)

            st.markdown("### Composition by level (latest year)")

            latest_year = sorted(df_f["Academic year"].unique())[-1]
            latest = df_f[df_f["Academic year"] == latest_year]
            latest = latest.groupby("Level of qualification", as_index=False)["Number"].sum()
            latest = latest.sort_values("Number", ascending=False)

            st.caption(f"Latest year in filtered data: **{latest_year}**")
            st.bar_chart(
                data=latest.set_index("Level of qualification")["Number"],
                use_container_width=True,
            )

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 3 – HECoS Subject Data
# -------------------------------------------------------------------
with tab_hecos:
    st.header("HECoS / CAH subject data")

    df = dfs.get("HECoS Data")
    if df is None:
        st.error("Could not load **HECoS Data.xlsx**.")
    else:
        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            cah_marker_vals = sorted(df["CAH level marker"].dropna().unique())
            cah_marker = col1.multiselect(
                "CAH level marker",
                options=cah_marker_vals,
                default=cah_marker_vals,
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col2.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["Entrant", "All"],
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col3.multiselect(
                "Level of study",
                options=level_vals,
                default=["All", "First degree"],
            )

            mode_vals = sorted(df["Mode of study"].dropna().unique())
            mode_of_study = col4.multiselect(
                "Mode of study", options=mode_vals, default=["All", "Full-time"]
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year", options=year_vals, default=year_vals
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "CAH level marker", cah_marker)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_marker)
        df_f = multiselect_filter(df_f, "Level of study", level_of_study)
        df_f = multiselect_filter(df_f, "Mode of study", mode_of_study)
        if years:
            df_f = df_f[df_f["Academic Year"].isin(years)]

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            st.markdown("### Entrants by subject (latest year)")

            latest_year = sorted(df_f["Academic Year"].unique())[-1]
            latest = df_f[df_f["Academic Year"] == latest_year]
            latest = (
                latest.groupby("CAH level subject", as_index=False)["Number"]
                .sum()
                .sort_values("Number", ascending=False)
            )

            st.caption(f"Latest year in filtered data: **{latest_year}**")

            # Limit to top N for readability
            top_n = st.slider("Top N subjects", min_value=5, max_value=40, value=20)
            latest_top = latest.head(top_n)

            st.bar_chart(
                latest_top.set_index("CAH level subject")["Number"],
                use_container_width=True,
            )

            st.markdown("### Trend for a selected subject over time")
            subject_sel = st.selectbox(
                "Subject", options=sorted(df_f["CAH level subject"].unique())
            )

            subj_ts = (
                df_f[df_f["CAH level subject"] == subject_sel]
                .groupby("Academic Year", as_index=False)["Number"]
                .sum()
                .sort_values("Academic Year")
            )

            subj_ts = subj_ts.set_index("Academic Year")
            st.line_chart(subj_ts["Number"], use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 4 – Custom visualisations (e.g. visualizations.csv)
# -------------------------------------------------------------------
with tab_custom:
    st.header("Custom visualisations from another file")

    st.markdown(
        """
Upload any **CSV** or **Excel** file (for example, your `visualizations.csv`
that comes out of the regression script). Then choose the X/Y axes and chart type.
"""
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel", type=["csv", "xlsx", "xls"], accept_multiple_files=False
    )

    if uploaded is not None:
        # Auto-detect format
        if uploaded.name.lower().endswith(".csv"):
            df_custom = pd.read_csv(uploaded)
        else:
            df_custom = pd.read_excel(uploaded)

        st.subheader("Data preview")
        st.dataframe(df_custom.head(), use_container_width=True, height=250)

        numeric_cols = df_custom.select_dtypes(include="number").columns.tolist()
        all_cols = df_custom.columns.tolist()
        non_numeric_cols = [c for c in all_cols if c not in numeric_cols]

        if not numeric_cols:
            st.warning("No numeric columns found – need at least one for Y-axis.")
        else:
            st.markdown("### Build a chart")

            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox(
                "X-axis",
                options=all_cols,
                index=0,
            )
            y_col = col2.selectbox(
                "Y-axis (numeric)",
                options=numeric_cols,
                index=0,
            )
            chart_type = col3.selectbox(
                "Chart type", options=["Bar", "Line", "Scatter"], index=0
            )

            # Optional group-by for aggregation
            group = st.checkbox(
                "Aggregate by X (sum of Y)", value=True, help="Useful for bar/line charts."
            )

            df_plot = df_custom.copy()

            if group:
                df_plot = (
                    df_plot.groupby(x_col, as_index=False)[y_col]
                    .sum()
                    .sort_values(x_col)
                )

            st.write("#### Chart")

            if chart_type == "Bar":
                st.bar_chart(
                    df_plot.set_index(x_col)[y_col],
                    use_container_width=True,
                )
            elif chart_type == "Line":
                st.line_chart(
                    df_plot.set_index(x_col)[y_col],
                    use_container_width=True,
                )
            else:  # Scatter
                # For scatter we’ll use st.pyplot via pandas plot
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                ax.scatter(df_plot[x_col], df_plot[y_col])
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"{y_col} vs {x_col}")
                plt.xticks(rotation=45, ha="right")
                st.pyplot(fig, use_container_width=True)

    else:
        st.info("Upload a CSV/Excel file to start building custom visualisations.")
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import List, Optional, Tuple, Dict
from math import erf, sqrt


# ============================================================
# Generic helpers
# ============================================================

def extract_closing_year(year_value) -> Optional[int]:
    """Convert codes like 201516 or 202122 into closing year (2016, 2022)."""
    if pd.isna(year_value):
        return None
    s = str(year_value).strip()

    if len(s) == 4 and s.isdigit():
        return int(s)

    if len(s) == 6 and s.isdigit():
        try:
            start_year = int(s[:4])
            return start_year + 1
        except ValueError:
            return None

    try:
        val = int(float(s))
        if 1900 <= val <= 2100:
            return val
    except ValueError:
        pass

    return None


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def fix_year_column(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    """Add 'year' column from time_period / academic_year if present."""
    df = df.copy()
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return df
    col = existing[0]
    df["year"] = df[col].apply(extract_closing_year)
    return df


def filter_local_authority(df: pd.DataFrame) -> pd.DataFrame:
    """Keep geographic_level == 'Local authority' if column exists."""
    df = df.copy()
    if "geographic_level" in df.columns:
        mask = df["geographic_level"].str.lower().str.replace(" ", "_") == "local_authority"
        df = df[mask]
    return df


def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_spc_from_upload(upload) -> pd.DataFrame:
    """Read an SPC CSV from upload and apply generic cleaning."""
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period"])
    df = filter_local_authority(df)
    return df


def common_key_cols(df: pd.DataFrame) -> List[str]:
    """Common LA-level keys."""
    candidates = [
        "year",
        "country_code", "country_name",
        "region_code", "region_name",
        "la_code", "la_name",
        "school_type",
    ]
    return [c for c in candidates if c in df.columns]


def safe_left_merge(left: pd.DataFrame, right: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Left join using intersection of common key columns."""
    if right is None or right.empty:
        return left

    left_keys = common_key_cols(left)
    right_keys = common_key_cols(right)
    keys = [k for k in left_keys if k in right_keys]
    if not keys:
        return left
    return left.merge(right, on=keys, how="left")


# ============================================================
# SPC cleaning functions
# ============================================================

def clean_spc_school_characteristics(upload) -> pd.DataFrame:
    """Build LA × school_type cohorts from spc_school_characteristics."""
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period"])
    df = filter_local_authority(df)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    cols_keep = common_key_cols(df) + ["headcount_of_pupils"]
    cols_keep = [c for c in cols_keep if c in df.columns]
    df = df[cols_keep]

    if "headcount_of_pupils" in df.columns:
        group_cols = [c for c in common_key_cols(df) if c != "country_code"]
        df = (
            df.groupby(group_cols, as_index=False)["headcount_of_pupils"]
            .sum()
            .rename(columns={"headcount_of_pupils": "cohort_size"})
        )

    return df


def clean_spc_pupils_fsm(upload) -> pd.DataFrame:
    """FSM% at LA × school_type from spc_pupils_fsm."""
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm" in df.columns:
        eligible = df["fsm"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm_percent"})
    )
    return out


def clean_spc_fsm6(upload) -> pd.DataFrame:
    """FSM6% at LA × school_type from spc_fsm6."""
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm6_eligibility" in df.columns:
        eligible = df["fsm6_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "fsm6_percent"})
    )
    return out


def clean_spc_pupils_fsm_ethnicity_yrgp(upload) -> pd.DataFrame:
    """Build disadvantaged_percent from spc_pupils_fsm_ethnicity_yrgp."""
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])

    if "fsm_eligibility" in df.columns:
        eligible = df["fsm_eligibility"].astype(str).str.contains("eligible", case=False, na=False)
        df = df[eligible]

    if "characteristic" in df.columns:
        total = df["characteristic"].astype(str).str.lower() == "total"
        if total.any():
            df = df[total]

    keys = common_key_cols(df)
    if "percent_of_pupils" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percent_of_pupils"]
        .mean()
        .rename(columns={"percent_of_pupils": "disadvantaged_percent"})
    )
    return out


def clean_spc_pupils_ethnicity_and_language(upload) -> pd.DataFrame:
    """Ethnicity proportions (eth_*) + eal_percent from spc_pupils_ethnicity_and_language."""
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percent_of_pupils"])
    keys = common_key_cols(df)

    # Ethnicity wide
    if "ethnicity_minor" in df.columns:
        eth = (
            df.groupby(keys + ["ethnicity_minor"], as_index=False)["percent_of_pupils"]
            .mean()
        )
        eth_wide = eth.pivot_table(
            index=keys,
            columns="ethnicity_minor",
            values="percent_of_pupils",
            aggfunc="first",
        )
        eth_wide.columns = [
            "eth_" + str(c).lower().replace(" ", "_").replace("/", "_")
            for c in eth_wide.columns
        ]
        eth_wide = eth_wide.reset_index()
    else:
        eth_wide = pd.DataFrame(columns=keys)

    # EAL
    if "language" in df.columns:
        lang = (
            df.groupby(keys + ["language"], as_index=False)["percent_of_pupils"]
            .mean()
        )
        mask_eal = lang["language"].astype(str).str.contains("other than english", case=False, na=False)
        eal = (
            lang[mask_eal]
            .groupby(keys, as_index=False)["percent_of_pupils"]
            .sum()
            .rename(columns={"percent_of_pupils": "eal_percent"})
        )
    else:
        eal = pd.DataFrame(columns=keys + ["eal_percent"])

    merged = pd.merge(eth_wide, eal, on=keys, how="outer")
    return merged


def clean_spc_pupils_age_and_sex(upload) -> pd.DataFrame:
    """pct_male / pct_female at LA × school_type from spc_pupils_age_and_sex."""
    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["headcount"])
    keys = common_key_cols(df)
    if "sex" not in df.columns or "headcount" not in df.columns:
        return df.iloc[0:0][keys]

    g = (
        df.groupby(keys + ["sex"], as_index=False)["headcount"]
        .sum()
    )
    pivot = g.pivot_table(
        index=keys,
        columns="sex",
        values="headcount",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    cols = list(pivot.columns)
    male_col = next((c for c in cols if str(c).lower().startswith("male")), None)
    female_col = next((c for c in cols if str(c).lower().startswith("female")), None)

    if male_col is None or female_col is None:
        pivot["pct_male"] = np.nan
        pivot["pct_female"] = np.nan
    else:
        total = pivot[male_col] + pivot[female_col]
        pivot["pct_male"] = np.where(total > 0, 100 * pivot[male_col] / total, np.nan)
        pivot["pct_female"] = np.where(total > 0, 100 * pivot[female_col] / total, np.nan)

    return pivot[keys + ["pct_male", "pct_female"]]


def clean_spc_uifsm(upload) -> Optional[pd.DataFrame]:
    """UIFSM% at LA × school_type from spc_uifsm (optional)."""
    if upload is None:
        return None

    df = load_spc_from_upload(upload)

    if "phase_type_grouping" in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    df = ensure_numeric(df, ["percentage"])

    if "characteristic" in df.columns:
        mask_total = df["characteristic"].astype(str).str.lower() == "total"
        if mask_total.any():
            df = df[mask_total]

    keys = common_key_cols(df)
    if "percentage" not in df.columns:
        return df.iloc[0:0][keys]

    out = (
        df.groupby(keys, as_index=False)["percentage"]
        .mean()
        .rename(columns={"percentage": "uifsm_percent_total"})
    )
    return out


def clean_performance_data(upload) -> pd.DataFrame:
    """
    Clean KS4 performance data.

    Expects columns something like:
      la_code, la_name, region_name, school_type / phase_type_grouping,
      time_period or academic_year,
      attainment8 / att8score / progress8_score / p8_score / performance_score
    """
    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df = fix_year_column(df, ["time_period", "academic_year"])

    # Normalise performance column names
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "attainment8" in lc or lc == "att8score" or lc == "att8_score":
            col_map[c] = "attainment8"
        if "progress8" in lc or lc == "p8_score":
            col_map[c] = "progress8"
    df = df.rename(columns=col_map)

    perf_cols = ["attainment8", "progress8", "performance_score"]
    df = ensure_numeric(df, perf_cols)

    # Harmonise school_type if needed
    if "phase_type_grouping" in df.columns and "school_type" not in df.columns:
        df = df.rename(columns={"phase_type_grouping": "school_type"})

    return df


def clean_uk_hpi(upload) -> Optional[pd.DataFrame]:
    """Clean UK-HPI-cleaned.csv from upload, to region-year avg house price."""
    if upload is None:
        return None

    df = pd.read_csv(upload)
    df = standardise_columns(df)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    df = df.rename(
        columns={
            "regionname": "region_name",
            "areacode": "region_code",
            "averageprice": "avg_house_price",
        }
    )

    group_cols = [c for c in ["year", "region_code", "region_name"] if c in df.columns]
    out = (
        df.groupby(group_cols, as_index=False)["avg_house_price"]
        .mean()
    )
    return out


# ============================================================
# Modelling (NumPy OLS)
# ============================================================

def normal_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def prepare_modelling_data(
    df: pd.DataFrame,
    dependent_var: str,
    continuous_vars: List[str],
    categorical_vars: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build X (with intercept 'const') and y."""
    cols_needed = [dependent_var] + continuous_vars + categorical_vars
    df_model = df[cols_needed].dropna(subset=cols_needed).copy()

    X_cont = df_model[continuous_vars].astype(float) if continuous_vars else pd.DataFrame(index=df_model.index)
    X_cat = pd.get_dummies(df_model[categorical_vars], drop_first=True, dummy_na=False) if categorical_vars else pd.DataFrame(index=df_model.index)
    X_const = pd.Series(1.0, index=df_model.index, name="const")

    X = pd.concat([X_const, X_cont, X_cat], axis=1)
    y = df_model[dependent_var].astype(float)

    return X, y


def run_ols_numpy(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    """Run OLS using NumPy; return dict of results."""
    X_mat = X.values
    y_vec = y.values.reshape(-1, 1)
    n, k = X_mat.shape

    XtX = X_mat.T @ X_mat
    XtX_inv = np.linalg.inv(XtX)
    XtY = X_mat.T @ y_vec
    beta = XtX_inv @ XtY

    y_hat = X_mat @ beta
    residuals = y_vec - y_hat

    ssr = float((residuals.T @ residuals))
    sst = float(((y_vec - y_vec.mean()).T @ (y_vec - y_vec.mean())))
    r2 = 1.0 - ssr / sst if sst > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else np.nan

    sigma2 = ssr / (n - k) if n > k else np.nan
    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.diag(var_beta)).reshape(-1, 1)

    t_stats = beta / se
    t_flat = t_stats.flatten()
    p_vals = np.array([2 * (1 - normal_cdf(abs(t))) for t in t_flat]).reshape(-1, 1)

    beta_s = pd.Series(beta.flatten(), index=X.columns, name="coef")
    se_s = pd.Series(se.flatten(), index=X.columns, name="std_err")
    t_s = pd.Series(t_flat, index=X.columns, name="t_stat")
    p_s = pd.Series(p_vals.flatten(), index=X.columns, name="p_value")

    y_hat_s = pd.Series(y_hat.flatten(), index=X.index, name="fitted")
    resid_s = pd.Series(residuals.flatten(), index=X.index, name="residual")

    return {
        "beta": beta_s,
        "se": se_s,
        "t": t_s,
        "p": p_s,
        "y_hat": y_hat_s,
        "residuals": resid_s,
        "r2": r2,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "sigma2": sigma2,
        "XtX_inv": XtX_inv,
        "X": X,
    }


def cooks_like_influence(results: Dict[str, object]) -> pd.Series:
    """Compute a Cook's-distance-like influence measure."""
    X = results["X"]
    residuals = results["residuals"]
    sigma2 = results["sigma2"]
    k = results["k"]

    X_mat = X.values
    XtX_inv = results["XtX_inv"]
    hat_matrix = X_mat @ XtX_inv @ X_mat.T
    h = np.diag(hat_matrix)

    resid_vals = residuals.values
    cooks = (resid_vals**2 / (k * sigma2)) * (h / (1 - h)**2)
    return pd.Series(cooks, index=X.index, name="cooks_distance")


# ============================================================
# Visualisation helpers (Altair)
# ============================================================

def scatter_with_line(df, x, y, color=None, title="", xlab=None, ylab=None):
    """Altair scatter with global OLS line."""
    data = df.dropna(subset=[x, y]).copy()
    if data.empty:
        return alt.Chart().mark_text(text="No data").properties(height=200)

    base = alt.Chart(data).mark_point(opacity=0.6).encode(
        x=alt.X(x, title=xlab or x),
        y=alt.Y(y, title=ylab or y),
        tooltip=list(data.columns)
    )

    if color and color in data.columns:
        base = base.encode(color=color)

    # Regression line
    x_vals = data[x].values.astype(float)
    y_vals = data[y].values.astype(float)
    if len(x_vals) > 1:
        slope, intercept = np.polyfit(x_vals, y_vals, 1)
        line_df = pd.DataFrame({
            x: np.linspace(x_vals.min(), x_vals.max(), 50)
        })
        line_df[y] = slope * line_df[x] + intercept

        line = alt.Chart(line_df).mark_line(color="red").encode(
            x=x,
            y=y
        )
        chart = (base + line).properties(title=title, height=350)
    else:
        chart = base.properties(title=title, height=350)

    return chart.interactive()


def heatmap_region_fsm_perf(df, region_col, fsm_col, perf_col):
    data = df.dropna(subset=[region_col, fsm_col, perf_col]).copy()
    if data.empty:
        return alt.Chart().mark_text(text="No data").properties(height=200)

    data["fsm_bin"] = pd.qcut(data[fsm_col], q=4, duplicates="drop")
    data["fsm_bin"] = data["fsm_bin"].astype(str)

    chart = alt.Chart(data).mark_rect().encode(
        x=alt.X("fsm_bin:N", title="FSM quantile group"),
        y=alt.Y(f"{region_col}:N", title="Region"),
        color=alt.Color(f"mean({perf_col}):Q", title=f"Mean {perf_col}"),
        tooltip=[region_col, "fsm_bin", alt.Tooltip(f"mean({perf_col}):Q", title="Mean performance")]
    ).properties(
        title="Region × FSM × performance",
        height=350
    )

    return chart


def boxplot_perf_by_school_type(df, perf_col, school_type_col):
    data = df.dropna(subset=[perf_col, school_type_col]).copy()
    if data.empty:
        return alt.Chart().mark_text(text="No data").properties(height=200)

    chart = alt.Chart(data).mark_boxplot().encode(
        x=alt.X(f"{school_type_col}:N", title="School type / phase"),
        y=alt.Y(f"{perf_col}:Q", title=perf_col),
        tooltip=[school_type_col, perf_col]
    ).properties(
        title="Performance by school type",
        height=350
    )
    return chart


def residual_chart(fitted, residuals):
    data = pd.DataFrame({"fitted": fitted, "residual": residuals})
    chart = alt.Chart(data).mark_point(opacity=0.6).encode(
        x="fitted:Q",
        y="residual:Q",
        tooltip=["fitted", "residual"]
    ).properties(
        title="Residuals vs fitted values",
        height=350
    ) + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red").encode(y="y:Q")
    return chart.interactive()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="KS4 Socioeconomic Regression", layout="wide")
st.title("Data Mining in School Performance – KS4 & SPC Regression (Single-file app)")

st.markdown(
    """
Upload the SPC characteristic files, a KS4 performance file, and optionally UK-HPI data.  
The app will clean and merge them, then fit an OLS regression (NumPy-based) and show diagnostics.
"""
)

# Upload widgets
col1, col2 = st.columns(2)

with col1:
    fsm_file = st.file_uploader("spc_pupils_fsm.csv", type="csv")
    fsm6_file = st.file_uploader("spc_fsm6.csv", type="csv")
    fsm_eth_file = st.file_uploader("spc_pupils_fsm_ethnicity_yrgp.csv", type="csv")
    age_sex_file = st.file_uploader("spc_pupils_age_and_sex.csv", type="csv")

with col2:
    eth_lang_file = st.file_uploader("spc_pupils_ethnicity_and_language.csv", type="csv")
    school_char_file = st.file_uploader("spc_school_characteristics.csv", type="csv")
    uifsm_file = st.file_uploader("spc_uifsm.csv (optional)", type="csv")
    hpi_file = st.file_uploader("UK-HPI-cleaned.csv (optional)", type="csv")

perf_file = st.file_uploader("KS4 performance file (school or LA level)", type="csv")

required = [fsm_file, fsm6_file, fsm_eth_file, age_sex_file, eth_lang_file, school_char_file, perf_file]

if not all(required):
    st.info("Upload all required SPC files and the KS4 performance file to proceed.")
    st.stop()

if st.button("Run cleaning & regression"):
    with st.spinner("Cleaning and merging uploaded files..."):
        school_char_df = clean_spc_school_characteristics(school_char_file)
        fsm_df = clean_spc_pupils_fsm(fsm_file)
        fsm6_df = clean_spc_fsm6(fsm6_file)
        fsm_eth_df = clean_spc_pupils_fsm_ethnicity_yrgp(fsm_eth_file)
        age_sex_df = clean_spc_pupils_age_and_sex(age_sex_file)
        eth_lang_df = clean_spc_pupils_ethnicity_and_language(eth_lang_file)
        uifsm_df = clean_spc_uifsm(uifsm_file)
        perf_df = clean_performance_data(perf_file)
        hpi_df = clean_uk_hpi(hpi_file)

        merged = school_char_df.copy()
        merged = safe_left_merge(merged, fsm_df)
        merged = safe_left_merge(merged, fsm6_df)
        merged = safe_left_merge(merged, fsm_eth_df)
        merged = safe_left_merge(merged, age_sex_df)
        merged = safe_left_merge(merged, eth_lang_df)
        merged = safe_left_merge(merged, uifsm_df)
        merged = safe_left_merge(merged, perf_df)
        merged = safe_left_merge(merged, hpi_df)

        if "disadvantaged_percent" not in merged.columns:
            if "fsm6_percent" in merged.columns:
                merged["disadvantaged_percent"] = merged["fsm6_percent"]
            elif "fsm_percent" in merged.columns:
                merged["disadvantaged_percent"] = merged["fsm_percent"]

    st.success(f"Merged dataset built – {merged.shape[0]} rows, {merged.shape[1]} columns.")

    tab_data, tab_viz, tab_reg = st.tabs(["Data overview", "Visualisations", "Regression & diagnostics"])

    # ---------------- Data overview ----------------
    with tab_data:
        st.subheader("Preview")
        st.dataframe(merged.head(200))

        st.subheader("Summary statistics (numeric)")
        st.write(merged.describe().T)

    # ---------------- Visualisations ----------------
    with tab_viz:
        st.subheader("Exploratory visualisations")

        region_col = "region_name" if "region_name" in merged.columns else None
        la_col = "la_name" if "la_name" in merged.columns else None
        school_type_col = "school_type" if "school_type" in merged.columns else None

        perf_candidates = ["attainment8", "progress8", "performance_score"]
        perf_col = next((c for c in perf_candidates if c in merged.columns), None)

        fsm_col = "fsm_percent" if "fsm_percent" in merged.columns else None
        eal_col = "eal_percent" if "eal_percent" in merged.columns else None

        if perf_col is None:
            st.warning("No performance column (e.g. attainment8/progress8) found in KS4 file.")
        else:
            if fsm_col:
                st.markdown("**FSM% vs performance**")
                chart = scatter_with_line(merged, fsm_col, perf_col, color=region_col,
                                          title=f"{fsm_col} vs {perf_col}",
                                          xlab="FSM (%)", ylab=perf_col)
                st.altair_chart(chart, use_container_width=True)

            if eal_col:
                st.markdown("**EAL% vs performance**")
                chart = scatter_with_line(merged, eal_col, perf_col, color=region_col,
                                          title=f"{eal_col} vs {perf_col}",
                                          xlab="EAL (%)", ylab=perf_col)
                st.altair_chart(chart, use_container_width=True)

            if "avg_house_price" in merged.columns:
                st.markdown("**Average house price vs performance**")
                chart = scatter_with_line(merged, "avg_house_price", perf_col, color=region_col,
                                          title=f"Average house price vs {perf_col}",
                                          xlab="Average house price (£)", ylab=perf_col)
                st.altair_chart(chart, use_container_width=True)

            if region_col and fsm_col:
                st.markdown("**Region × FSM × performance heatmap**")
                chart = heatmap_region_fsm_perf(merged, region_col, fsm_col, perf_col)
                st.altair_chart(chart, use_container_width=True)

            if school_type_col:
                st.markdown("**Performance by school type**")
                chart = boxplot_perf_by_school_type(merged, perf_col, school_type_col)
                st.altair_chart(chart, use_container_width=True)

    # ---------------- Regression & diagnostics ----------------
    with tab_reg:
        st.subheader("Regression setup")

        perf_candidates = ["attainment8", "progress8", "performance_score"]
        dep_var = st.selectbox(
            "Dependent variable",
            [c for c in perf_candidates if c in merged.columns],
        )

        default_cont = [
            col for col in [
                "fsm_percent",
                "fsm6_percent",
                "disadvantaged_percent",
                "eal_percent",
                "pct_female",
                "cohort_size",
                "avg_house_price",
            ] if col in merged.columns
        ]

        continuous_vars = st.multiselect(
            "Continuous predictors",
            [c for c in merged.columns if merged[c].dtype != "object" and c != dep_var],
            default=default_cont,
        )

        cat_candidates = [c for c in ["school_type", "region_name"] if c in merged.columns]
        categorical_vars = st.multiselect(
            "Categorical predictors (fixed effects)",
            [c for c in merged.columns if merged[c].dtype == "object"],
            default=cat_candidates,
        )

        if st.button("Run OLS regression"):
            with st.spinner("Fitting NumPy-based OLS model..."):
                X, y = prepare_modelling_data(merged, dep_var, continuous_vars, categorical_vars)
                results = run_ols_numpy(X, y)
                cooks = cooks_like_influence(results)

            st.success("Model fitted.")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("R-squared", f"{results['r2']:.3f}")
            with col_b:
                st.metric("Adjusted R-squared", f"{results['adj_r2']:.3f}")

            coef_df = pd.concat(
                [results["beta"], results["se"], results["t"], results["p"]],
                axis=1
            )
            coef_df.columns = ["coef", "std_err", "t_stat", "p_value"]

            st.subheader("Coefficient table")
            st.dataframe(
                coef_df.style.format(
                    {"coef": "{:.3f}", "std_err": "{:.3f}", "t_stat": "{:.2f}", "p_value": "{:.3f}"}
                )
            )

            st.subheader("Residual diagnostics")
            chart = residual_chart(results["y_hat"], results["residuals"])
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Top 15 observations by influence (Cook's-like distance)")
            diag_df = pd.DataFrame({
                "fitted": results["y_hat"],
                "residual": results["residuals"],
                "cooks_distance": cooks,
            })
            st.dataframe(diag_df.sort_values("cooks_distance", ascending=False).head(15))
