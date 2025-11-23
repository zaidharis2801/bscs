# app.py
# School Performance Data Mining Dashboard – API version
#
# Uses Explore Education Statistics Public Data API to fetch data
# via /v1/data-sets/{dataSetId}/csv and then performs the same analysis
# as the earlier CSV-based version.

import io
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# ---------------------------------------------------------------------
# CONFIG: API BASE + DATASET IDS
# ---------------------------------------------------------------------

# For local dev, if you are running the API spec locally:
# BASE_URL = "http://localhost:5050"
# For production, replace with the real EES API base URL.
BASE_URL = "http://localhost:5050"

# IMPORTANT:
# Replace these with the *real* dataset IDs from the EES API portal.
# You can get them by:
# 1) Finding the publication
# 2) Listing its datasets
# 3) Copying the datasetId (UUID)
DATASET_IDS = {
    "school_performance": "<KS4_or_KS5_dataset_uuid_here>",
    "pupil_characteristics": "<pupil_characteristics_dataset_uuid_here>",
    "imd": "<imd_or_deprivation_dataset_uuid_here>",   # optional if exposed
    "ofsted": "<ofsted_dataset_uuid_here>",            # optional if exposed
}

# Version pattern –  e.g. "1.*" to always get latest minor in v1 series
DEFAULT_DATASET_VERSION = "1.*"

# ---------------------------------------------------------------------
# CONFIG: column mappings (from API CSV columns -> internal names)
# You will need to inspect the CSV headers once and adjust these.
# ---------------------------------------------------------------------

COLUMN_MAPPINGS = {
    "school_performance": {
        # API CSV column name -> internal standard name
        # Example names; change to match actual CSV headings:
        "time_period": "year",
        "school_urn": "urn",
        "school_name": "school_name",
        "local_authority_name": "la_name",
        "region_name": "region",
        "school_type_group": "school_type",
        "attainment_8_score": "attainment8",   # e.g. "Average Attainment 8 score"
        "progress_8_score": "progress8",
    },
    "pupil_characteristics": {
        "time_period": "year",
        "school_urn": "urn",
        "local_authority_name": "la_name",
        "region_name": "region",
        # FSM variable might be something like "Percent of pupils eligible for FSM"
        "fsm_percent": "fsm_percentage",
    },
    # If IMD/IDACI is available via a dataset:
    "imd": {
        "local_authority_name": "la_name",
        "imd_score": "imd_score",
        "imd_decile": "imd_decile",
    },
    # If Ofsted is available via a dataset:
    "ofsted": {
        "school_urn": "urn",
        "ofsted_overall_effectiveness": "ofsted_rating",
    },
}

# ---------------------------------------------------------------------
# Helpers: call EES API
# ---------------------------------------------------------------------


def fetch_dataset_csv(
    data_set_id: str,
    data_set_version: str = DEFAULT_DATASET_VERSION,
) -> pd.DataFrame:
    """
    Download a dataset as CSV using the /v1/data-sets/{dataSetId}/csv endpoint.

    NOTE:
    - The OpenAPI spec shows this as returning 'text/csv'.
    - We just pass dataSetVersion; all filtering is done client-side.
    """
    if not data_set_id or data_set_id.startswith("<"):
        # Not configured yet
        return pd.DataFrame()

    url = f"{BASE_URL}/v1/data-sets/{data_set_id}/csv"
    params = {"dataSetVersion": data_set_version}

    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.error(
            f"Failed to fetch dataset {data_set_id} from API. "
            f"Status: {resp.status_code}"
        )
        return pd.DataFrame()

    csv_bytes = resp.content
    df = pd.read_csv(io.BytesIO(csv_bytes))
    return df


def standardise_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Rename columns from raw API CSV to our internal standard names.

    mapping: {raw_name: internal_name}
    """
    if df.empty:
        return df

    # Build rename dict by matching case-insensitively
    rename_dict = {}
    raw_cols_upper = {c.upper(): c for c in df.columns}
    for raw_col, std_col in mapping.items():
        raw_upper = raw_col.upper()
        if raw_upper in raw_cols_upper:
            rename_dict[raw_cols_upper[raw_upper]] = std_col

    df = df.rename(columns=rename_dict)
    return df


@st.cache_data(show_spinner=True)
def load_all_data_from_api() -> Dict[str, pd.DataFrame]:
    """Fetch datasets via API, standardise them, and merge into a master table."""

    # 1. Fetch raw CSVs via API
    sp_raw = fetch_dataset_csv(DATASET_IDS["school_performance"])
    pupils_raw = fetch_dataset_csv(DATASET_IDS["pupil_characteristics"])
    imd_raw = fetch_dataset_csv(DATASET_IDS["imd"])
    ofsted_raw = fetch_dataset_csv(DATASET_IDS["ofsted"])

    # 2. Standardise column names
    sp = standardise_columns(sp_raw, COLUMN_MAPPINGS["school_performance"])
    pupils = standardise_columns(pupils_raw, COLUMN_MAPPINGS["pupil_characteristics"])
    imd = standardise_columns(imd_raw, COLUMN_MAPPINGS["imd"])
    ofsted = standardise_columns(ofsted_raw, COLUMN_MAPPINGS["ofsted"])

    # 3. Basic type cleaning
    for df in [sp, pupils]:
        if "year" in df.columns:
            df["year"] = df["year"].astype(str).str.extract(r"(\d{4})").astype(float)

    if "fsm_percentage" in pupils.columns:
        pupils["fsm_percentage"] = pd.to_numeric(
            pupils["fsm_percentage"], errors="coerce"
        )

    if "imd_score" in imd.columns:
        imd["imd_score"] = pd.to_numeric(imd["imd_score"], errors="coerce")

    # 4. Merge: performance + pupils + Ofsted + IMD
    master = sp

    if not pupils.empty:
        master = master.merge(
            pupils[
                [c for c in pupils.columns if c in ("urn", "year", "fsm_percentage")]
            ],
            on=["urn", "year"],
            how="left",
            suffixes=("", "_pupil"),
        )

    if not ofsted.empty:
        master = master.merge(
            ofsted[["urn", "ofsted_rating"]],
            on="urn",
            how="left",
        )

    if not imd.empty and "la_name" in master.columns:
        master = master.merge(
            imd[["la_name", "imd_score", "imd_decile"]],
            on="la_name",
            how="left",
        )

    # 5. Derived fields
    if "imd_decile" not in master.columns and "imd_score" in master.columns:
        master["imd_decile"] = pd.qcut(
            master["imd_score"], 10, labels=False, duplicates="drop"
        ) + 1

    if "imd_decile" in master.columns:
        master["high_deprivation"] = master["imd_decile"].apply(
            lambda x: "High (1–3)" if pd.notna(x) and x <= 3 else "Other"
        )

    return {
        "school_performance": sp,
        "pupils": pupils,
        "imd": imd,
        "ofsted": ofsted,
        "master": master,
    }


# ---------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters and return filtered DF."""
    if df.empty:
        return df

    years = sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else []
    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    school_types = (
        sorted(df["school_type"].dropna().unique().tolist())
        if "school_type" in df.columns
        else []
    )
    ofsted_ratings = (
        sorted(df["ofsted_rating"].dropna().unique().tolist())
        if "ofsted_rating" in df.columns
        else []
    )

    st.sidebar.header("Filters")

    year_sel = st.sidebar.multiselect(
        "Year",
        options=years,
        default=years,
    )

    region_sel = st.sidebar.multiselect(
        "Region",
        options=regions,
        default=regions if regions else [],
    )

    st.sidebar.markdown("---")
    school_type_sel = st.sidebar.multiselect(
        "School type",
        options=school_types,
        default=school_types if school_types else [],
    )

    ofsted_sel = st.sidebar.multiselect(
        "Ofsted rating",
        options=ofsted_ratings,
        default=ofsted_ratings if ofsted_ratings else [],
    )

    df_f = df.copy()
    if year_sel:
        df_f = df_f[df_f["year"].isin(year_sel)]
    if regions and region_sel:
        df_f = df_f[df_f["region"].isin(region_sel)]
    if school_types and school_type_sel:
        df_f = df_f[df_f["school_type"].isin(school_type_sel)]
    if ofsted_ratings and ofsted_sel:
        df_f = df_f[df_f["ofsted_rating"].isin(ofsted_sel)]

    return df_f


# ---------------------------------------------------------------------
# Dashboard views (same as before, but now using API-loaded data)
# ---------------------------------------------------------------------


def view_overview(df: pd.DataFrame):
    st.subheader("Overview: attainment and context")

    if df.empty:
        st.info("No data available with current filters.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Mean Attainment 8",
            f"{df['attainment8'].mean():.1f}" if "attainment8" in df.columns else "–",
        )
    with col2:
        if "fsm_percentage" in df.columns:
            st.metric("Median FSM %", f"{df['fsm_percentage'].median():.1f}%")
        else:
            st.metric("Median FSM %", "–")
    with col3:
        if "imd_decile" in df.columns:
            share_high_dep = (
                (df["imd_decile"] <= 3).mean() * 100
                if df["imd_decile"].notna().any()
                else np.nan
            )
            st.metric(
                "Schools in high deprivation",
                f"{share_high_dep:.1f}%" if not np.isnan(share_high_dep) else "–",
            )
        else:
            st.metric("Schools in high deprivation", "–")
    with col4:
        if "ofsted_rating" in df.columns:
            share_good = (
                (df["ofsted_rating"].isin(["Outstanding", "Good"])).mean() * 100
            )
            st.metric(
                "Outstanding / Good",
                f"{share_good:.1f}%" if not np.isnan(share_good) else "–",
            )
        else:
            st.metric("Outstanding / Good", "–")

    st.markdown("### Attainment 8 by deprivation (IMD decile)")
    if "imd_decile" in df.columns and "attainment8" in df.columns:
        tmp = (
            df.dropna(subset=["imd_decile", "attainment8"])
            .groupby("imd_decile", as_index=False)["attainment8"]
            .mean()
        )
        fig = px.line(
            tmp,
            x="imd_decile",
            y="attainment8",
            markers=True,
            labels={"imd_decile": "IMD decile (1 = most deprived)"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("IMD decile and Attainment 8 required for this chart.")


def view_socioeconomic(df: pd.DataFrame):
    st.subheader("Socioeconomic patterns and performance")

    if df.empty:
        st.info("No data available with current filters.")
        return

    # Attainment vs FSM
    if "fsm_percentage" in df.columns and "attainment8" in df.columns:
        st.markdown("### Relationship between FSM % and Attainment 8")
        fig = px.scatter(
            df,
            x="fsm_percentage",
            y="attainment8",
            color="imd_decile" if "imd_decile" in df.columns else None,
            opacity=0.4,
            labels={"fsm_percentage": "FSM %", "attainment8": "Attainment 8"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Regional comparison
    if "region" in df.columns and "attainment8" in df.columns:
        st.markdown("### Average Attainment 8 by region and deprivation band")
        tmp = df.copy()
        if "imd_decile" in tmp.columns:
            tmp["dep_band"] = np.where(
                tmp["imd_decile"] <= 3, "High deprivation (1–3)", "Other"
            )
        else:
            tmp["dep_band"] = "All"

        grp = (
            tmp.dropna(subset=["region", "attainment8"])
            .groupby(["region", "dep_band"], as_index=False)["attainment8"]
            .mean()
        )
        fig = px.bar(
            grp,
            x="region",
            y="attainment8",
            color="dep_band",
            barmode="group",
            labels={"attainment8": "Mean Attainment 8"},
        )
        st.plotly_chart(fig, use_container_width=True)


def view_ofsted_governance(df: pd.DataFrame):
    st.subheader("Ofsted ratings, governance and performance")

    if df.empty:
        st.info("No data available with current filters.")
        return

    # Boxplot: attainment by Ofsted rating
    if "ofsted_rating" in df.columns and "attainment8" in df.columns:
        st.markdown("### Attainment 8 by Ofsted rating (contextual view)")
        fig = px.box(
            df,
            x="ofsted_rating",
            y="attainment8",
            color="ofsted_rating",
            labels={"attainment8": "Attainment 8", "ofsted_rating": "Ofsted rating"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Governance vs performance
    if "school_type" in df.columns and "attainment8" in df.columns:
        st.markdown("### Average Attainment 8 by school type")
        grp = (
            df.dropna(subset=["school_type", "attainment8"])
            .groupby("school_type", as_index=False)["attainment8"]
            .mean()
        )
        fig = px.bar(
            grp,
            x="school_type",
            y="attainment8",
            labels={"attainment8": "Mean Attainment 8", "school_type": "School type"},
        )
        st.plotly_chart(fig, use_container_width=True)


def compute_disadvantage_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Return a year-by-year disadvantage gap (FSM vs non-FSM) DataFrame."""
    if "fsm_percentage" not in df.columns or "attainment8" not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    median_fsm = tmp["fsm_percentage"].median()
    tmp["fsm_group"] = np.where(tmp["fsm_percentage"] >= median_fsm, "High FSM", "Low FSM")

    gap = (
        tmp.dropna(subset=["year", "attainment8"])
        .groupby(["year", "fsm_group"], as_index=False)["attainment8"]
        .mean()
        .pivot(index="year", columns="fsm_group", values="attainment8")
        .reset_index()
    )

    if "High FSM" in gap.columns and "Low FSM" in gap.columns:
        gap["gap"] = gap["Low FSM"] - gap["High FSM"]

    return gap


def view_disadvantage_gap(df: pd.DataFrame):
    st.subheader("Disadvantage gap over time")

    if df.empty:
        st.info("No data available with current filters.")
        return

    gap = compute_disadvantage_gap(df)
    if gap.empty:
        st.info("FSM % and Attainment 8 are required to compute the disadvantage gap.")
        return

    st.markdown("### Attainment 8 for high- vs low-FSM schools")
    fig = px.line(
        gap,
        x="year",
        y=["High FSM", "Low FSM"],
        markers=True,
        labels={"value": "Attainment 8", "variable": "FSM group"},
    )
    st.plotly_chart(fig, use_container_width=True)

    if "gap" in gap.columns:
        st.markdown("### Gap (Low FSM minus High FSM)")
        fig2 = px.line(
            gap,
            x="year",
            y="gap",
            markers=True,
            labels={"gap": "Attainment 8 gap"},
        )
        st.plotly_chart(fig2, use_container_width=True)


def view_modelling_outliers(df: pd.DataFrame):
    st.subheader("Predictive modelling and contextual outliers")

    if df.empty:
        st.info("No data available with current filters.")
        return

    features = []
    if "fsm_percentage" in df.columns:
        features.append("fsm_percentage")
    if "imd_score" in df.columns:
        features.append("imd_score")
    if "imd_decile" in df.columns:
        features.append("imd_decile")
    if "school_type" in df.columns:
        features.append("school_type")
    if "ofsted_rating" in df.columns:
        features.append("ofsted_rating")
    if "region" in df.columns:
        features.append("region")

    target = st.selectbox(
        "Target variable", options=[c for c in ["attainment8", "progress8"] if c in df.columns]
    )

    if not target or not features:
        st.info("Need at least one feature and a target to build a model.")
        return

    df_m = df.dropna(subset=[target] + features).copy()
    if df_m.empty:
        st.info("Not enough complete rows for modelling.")
        return

    st.markdown("### Train a simple model to predict school performance")

    numeric_feats = [c for c in features if df_m[c].dtype != "object"]
    cat_feats = [c for c in features if df_m[c].dtype == "object"]

    X = df_m[features]
    y = df_m[target]

    preproc = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
        ]
    )

    model_type = st.selectbox("Model type", options=["Linear regression", "Decision tree"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if model_type == "Linear regression":
        model = LinearRegression()
    else:
        model = DecisionTreeRegressor(max_depth=5, random_state=42)

    X_train_t = preproc.fit_transform(X_train)
    X_test_t = preproc.transform(X_test)

    model.fit(X_train_t, y_train)
    y_pred = model.predict(X_test_t)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("R² (test)", f"{r2_score(y_test, y_pred):.3f}")
    with col2:
        st.metric("RMSE (test)", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

    df_m["predicted"] = model.predict(preproc.transform(X))
    df_m["residual"] = df_m[target] - df_m["predicted"]

    st.markdown("### Contextual residuals by region (actual minus predicted)")
    if "region" in df_m.columns:
        grp = (
            df_m.groupby("region", as_index=False)["residual"]
            .mean()
            .sort_values("residual", ascending=False)
        )
        fig = px.bar(
            grp,
            x="region",
            y="residual",
            labels={"residual": "Mean residual", "region": "Region"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Positive residuals indicate regions performing above model expectations "
            "given their socioeconomic context; negative residuals indicate below-expected performance."
        )
    else:
        st.info("Region column is required for aggregated residual analysis.")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    st.set_page_config(page_title="School Performance Explorer (API)", layout="wide")

    st.title("School Performance Data Mining Dashboard – API version")
    st.markdown(
        "This dashboard fetches data directly from the Explore Education Statistics "
        "Public Data API and explores how socioeconomic factors (FSM, deprivation, "
        "region) and accountability measures (Ofsted ratings) relate to GCSE/A-level outcomes. "
        "All views use public, anonymised data and focus on structural patterns rather "
        "than ranking individual schools."
    )

    data_dict = load_all_data_from_api()
    df_master = data_dict["master"]

    if df_master.empty:
        st.error(
            "Master dataset is empty. Check BASE_URL, DATASET_IDS, and COLUMN_MAPPINGS."
        )
        return

    df_filtered = apply_filters(df_master)

    st.markdown("---")
    view = st.radio(
        "Select view",
        options=[
            "Overview",
            "Socioeconomic patterns",
            "Ofsted & governance",
            "Disadvantage gap",
            "Modelling & contextual outliers",
        ],
        horizontal=True,
    )

    if view == "Overview":
        view_overview(df_filtered)
    elif view == "Socioeconomic patterns":
        view_socioeconomic(df_filtered)
    elif view == "Ofsted & governance":
        view_ofsted_governance(df_filtered)
    elif view == "Disadvantage gap":
        view_disadvantage_gap(df_filtered)
    else:
        view_modelling_outliers(df_filtered)

    st.markdown("---")
    st.caption(
        "Note: All results are descriptive and rely on observational data from the "
        "Public Data API. They should be interpreted as patterns and associations, "
        "not causal claims."
    )


if __name__ == "__main__":
    main()
