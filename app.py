# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Higher Education Entrant & Subject Visualisations – England",
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


def multiselect_filter(df, col, selected):
    if not selected or "All" in selected:
        return df
    return df[df[col].isin(selected)]


# -------------------------------------------------------------------
# PAGE TITLE & INTRO
# -------------------------------------------------------------------
st.title("Higher Education Entrant & Subject Visualisations (England)")

st.markdown(
    """
This dashboard provides exploratory visualisations of **entrants to higher education in England**, 
based on the pre-cleaned analysis files. It is intended to support the thesis questions on how 
**socio-economic factors, deprivation and subject mix** relate to higher education participation.

Tabs:

- **Count by Occupation** – entrants by socio-economic category, with filters for level of study, entrant marker and year.
- **Count by Level of Qualification** – time series of entrants by level (e.g. first degree vs other).
- **HECoS / CAH Subject Data** – entrants by subject area using HECoS codes aggregated to CAH subject groups.
- **Custom visualisations file** – upload a pre-aggregated file (for example, `visualisations.csv`) 
  from your regression script and build your own charts.

All data here are **aggregated** and restricted to **higher education providers in England**.
"""
)

# Load data
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

    st.markdown(
        """
This tab shows **entrants to higher education in England** by occupation-related and other 
socio-economic characteristics.

Key points:

- **Country of HE provider** is restricted to **England** in this dashboard.
- **Entrant marker** distinguishes **new entrants** from other records (e.g. continuing students).
- The **“% by category (latest year)”** view shows the **share of entrants** in each response 
  category for the most recent academic year in the filtered data (e.g. *Yes*, *No*, *Don't know*).
"""
    )

    df = dfs.get("Count by Occupation")
    if df is None:
        st.error("Could not load **Count by Occupation.xlsx**.")
    else:
        # Restrict to England only for HE provider
        if "Country of HE provider" in df.columns:
            df = df[df["Country of HE provider"] == "England"].copy()
            # Keep the column but fix its value so we can describe it in the UI if needed
            df["Country of HE provider"] = "England"

        with st.expander("Filters", expanded=True):
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            # Category marker (e.g. type of characteristic)
            cat_marker_vals = sorted(df["Category Marker"].dropna().unique())
            cat_marker = col1.multiselect(
                "Category marker",
                options=cat_marker_vals,
                default=cat_marker_vals,
                key="occ_cat_marker",
                help="Select which characteristic or grouping is shown (e.g. socio-economic category).",
            )

            # Country is fixed to England – show as non-editable information
            col2.markdown(
                """
**Country of HE provider**

This dashboard only includes providers in **England**.
"""
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col3.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["Entrant", "All"] if "Entrant" in entrant_vals else entrant_vals,
                key="occ_entrant_marker",
                help="‘Entrant’ usually refers to new entrants in that academic year; ‘All’ may include other records.",
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col4.multiselect(
                "Level of study",
                options=level_vals,
                default=[lv for lv in level_vals if lv in ["All", "First degree"]] or level_vals,
                key="occ_level_study",
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year",
                options=year_vals,
                default=year_vals,
                key="occ_academic_year",
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Category Marker", cat_marker)
        df_f = multiselect_filter(df_f, "Entrant marker", entrant_marker)
        df_f = multiselect_filter(df_f, "Level of study", level_of_study)
        if years:
            df_f = df_f[df_f["Academic Year"].isin(years)]

        metric = st.radio(
            "Metric",
            ["Number", "Percentage"],
            horizontal=True,
            key="occ_metric",
            help="Switch between absolute counts and percentage share of entrants within each category.",
        )

        if metric == "Percentage":
            df_f = df_f.dropna(subset=["Percentage"])

        st.markdown("### Visualisations")

        view_type = st.radio(
            "View",
            ["% by category (latest year)", "Trend over time (by category)"],
            horizontal=True,
            key="occ_view_type",
            help="‘% by category (latest year)’ focuses on the latest academic year; the trend view shows changes over time.",
        )

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            metric_col = metric

            # Group by academic year and specific response category
            agg = (
                df_f.groupby(["Academic Year", "Category"], as_index=False)[metric_col]
                .sum()
            )

            if view_type == "% by category (latest year)":
                latest_year = sorted(agg["Academic Year"].unique())[-1]
                latest = agg[agg["Academic Year"] == latest_year]
                latest = latest.sort_values(metric_col, ascending=False)

                # Rename columns for clearer axis labels
                latest_display = latest.rename(
                    columns={
                        "Category": "Characteristic category (e.g. Yes/No/Don't know)",
                        metric_col: "Value",
                    }
                )

                st.caption(
                    f"Latest academic year in filtered data: **{latest_year}** – "
                    "values show the share or number of entrants in each response category."
                )
                st.bar_chart(
                    data=latest_display.set_index("Characteristic category (e.g. Yes/No/Don't know)")["Value"],
                    use_container_width=True,
                )
            else:
                pivot = (
                    agg.pivot(
                        index="Academic Year",
                        columns="Category",
                        values=metric_col,
                    )
                    .sort_index()
                )

                pivot_display = pivot.copy()
                pivot_display.index.name = "Academic Year"

                st.caption(
                    "Time series of the chosen metric by response category for the selected academic years."
                )
                st.line_chart(pivot_display, use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 2 – Count by Level of Qualification
# -------------------------------------------------------------------
with tab_level:
    st.header("Count by Level of Qualification")

    st.markdown(
        """
This tab shows entrants by **level of qualification** (for example, first degrees versus 
other levels) over time.

Typical use:

- Assess how the mix of qualification levels has changed.
- Compare first degrees to other routes in the most recent year.
"""
    )

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
                key="level_level_qual",
            )

            year_vals = sorted(df["Academic year"].dropna().unique())
            years = col2.multiselect(
                "Academic year",
                options=year_vals,
                default=year_vals,
                key="level_academic_year",
            )

        df_f = df.copy()
        df_f = multiselect_filter(df_f, "Level of qualification", level_sel)
        if years:
            df_f = df_f[df_f["Academic year"].isin(years)]

        if df_f.empty:
            st.warning("No data after filters – adjust filters to see charts.")
        else:
            st.markdown("### Time series of entrants by level of qualification")

            pivot = (
                df_f.pivot(
                    index="Academic year",
                    columns="Level of qualification",
                    values="Number",
                )
                .sort_index()
            )
            pivot.index.name = "Academic year"

            st.line_chart(pivot, use_container_width=True)

            st.markdown("### Composition by level (latest year)")

            latest_year = sorted(df_f["Academic year"].unique())[-1]
            latest = df_f[df_f["Academic year"] == latest_year]
            latest = latest.groupby("Level of qualification", as_index=False)["Number"].sum()
            latest = latest.sort_values("Number", ascending=False)

            latest_display = latest.rename(
                columns={
                    "Level of qualification": "Level of qualification",
                    "Number": "Number of entrants",
                }
            )

            st.caption(
                f"Latest academic year in filtered data: **{latest_year}** – "
                "showing the distribution of entrants by qualification level."
            )
            st.bar_chart(
                data=latest_display.set_index("Level of qualification")["Number of entrants"],
                use_container_width=True,
            )

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 3 – HECoS Subject Data
# -------------------------------------------------------------------
with tab_hecos:
    st.header("HECoS / CAH subject data")

    st.markdown(
        """
This tab focuses on **subject areas** using the **Higher Education Classification of Subjects (HECoS)** 
and the **Common Aggregation Hierarchy (CAH)**:

- **HECoS**: detailed subject codes used by higher education providers (replacing JACS).
- **CAH**: groups HECoS codes into broader subject families (e.g. ‘Computing’, ‘Business’, ‘Psychology’).

In the charts we show **subject names** (CAH level subjects) rather than numeric codes to keep labels readable.
"""
    )

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
                key="hecos_cah_marker",
                help="Choose the CAH level (e.g. CAH1 / CAH2 / CAH3) for aggregation.",
            )

            entrant_vals = sorted(df["Entrant marker"].dropna().unique())
            entrant_marker = col2.multiselect(
                "Entrant marker",
                options=entrant_vals,
                default=["Entrant", "All"] if "Entrant" in entrant_vals else entrant_vals,
                key="hecos_entrant_marker",
                help="‘Entrant’ usually refers to new entrants in that academic year.",
            )

            level_vals = sorted(df["Level of study"].dropna().unique())
            level_of_study = col3.multiselect(
                "Level of study",
                options=level_vals,
                default=[lv for lv in level_vals if lv in ["All", "First degree"]] or level_vals,
                key="hecos_level_study",
            )

            mode_vals = sorted(df["Mode of study"].dropna().unique())
            mode_of_study = col4.multiselect(
                "Mode of study",
                options=mode_vals,
                default=[mv for mv in mode_vals if mv in ["All", "Full-time"]] or mode_vals,
                key="hecos_mode_study",
            )

            year_vals = sorted(df["Academic Year"].dropna().unique())
            years = col5.multiselect(
                "Academic Year",
                options=year_vals,
                default=year_vals,
                key="hecos_academic_year",
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
            st.markdown("### Entrants by CAH subject (latest year)")

            latest_year = sorted(df_f["Academic Year"].unique())[-1]
            latest = df_f[df_f["Academic Year"] == latest_year]
            latest = (
                latest.groupby("CAH level subject", as_index=False)["Number"]
                .sum()
                .sort_values("Number", ascending=False)
            )

            latest_display = latest.rename(
                columns={
                    "CAH level subject": "Subject (CAH)",
                    "Number": "Number of entrants",
                }
            )

            st.caption(
                f"Latest academic year in filtered data: **{latest_year}** – "
                "showing the top subject groups by number of entrants."
            )

            top_n = st.slider(
                "Top N subjects",
                min_value=5,
                max_value=40,
                value=20,
                key="hecos_top_n_subjects",
            )
            latest_top = latest_display.head(top_n)

            st.bar_chart(
                latest_top.set_index("Subject (CAH)")["Number of entrants"],
                use_container_width=True,
            )

            st.markdown("### Trend for a selected subject over time")

            subject_options = sorted(df_f["CAH level subject"].dropna().unique())
            subject_sel = st.selectbox(
                "Subject (CAH level)",
                options=subject_options,
                key="hecos_subject_select",
            )

            subj_ts = (
                df_f[df_f["CAH level subject"] == subject_sel]
                .groupby("Academic Year", as_index=False)["Number"]
                .sum()
                .sort_values("Academic Year")
            )

            subj_ts_display = subj_ts.rename(
                columns={"Academic Year": "Academic Year", "Number": "Number of entrants"}
            ).set_index("Academic Year")

            st.line_chart(subj_ts_display["Number of entrants"], use_container_width=True)

        with st.expander("Raw data preview"):
            st.dataframe(df_f, use_container_width=True, height=300)

# -------------------------------------------------------------------
# TAB 4 – Custom visualisations (e.g. visualisations.csv)
# -------------------------------------------------------------------
with tab_custom:
    st.header("Custom visualisations from another file")

    st.markdown(
        """
Upload any **CSV** or **Excel** file (for example, your `visualisations.csv`
that comes out of the regression script). Then choose the X/Y axes and chart type.

This is designed to link the **regression outputs** from the school performance analysis 
to a simple visual layer – for example, plotting contextual residuals or disadvantage gaps.
"""
    )

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=False,
        key="custom_file_uploader",
    )

    if uploaded is not None:
        if uploaded.name.lower().endswith(".csv"):
            df_custom = pd.read_csv(uploaded)
        else:
            df_custom = pd.read_excel(uploaded)

        st.subheader("Data preview")
        st.dataframe(df_custom.head(), use_container_width=True, height=250)

        numeric_cols = df_custom.select_dtypes(include="number").columns.tolist()
        all_cols = df_custom.columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns found – need at least one for Y-axis.")
        else:
            st.markdown("### Build a chart")

            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox(
                "X-axis",
                options=all_cols,
                index=0,
                key="custom_x_axis",
            )
            y_col = col2.selectbox(
                "Y-axis (numeric)",
                options=numeric_cols,
                index=0,
                key="custom_y_axis",
            )
            chart_type = col3.selectbox(
                "Chart type",
                options=["Bar", "Line", "Scatter"],
                index=0,
                key="custom_chart_type",
            )

            group = st.checkbox(
                "Aggregate by X (sum of Y)",
                value=True,
                help="Useful for bar/line charts – e.g. total entrants by region or residual band.",
                key="custom_group_checkbox",
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
            else:
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
