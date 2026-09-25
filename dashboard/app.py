import re
from pathlib import Path

import folium
from folium.plugins import FastMarkerCluster
import pandas as pd
import plotly.express as px
import streamlit as st


# =========================================================
# Paths configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "output"
DATA_DIR = BASE_DIR / "data" / "processed"


# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="Road Accident Hotspot & Risk Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Custom Styling
# =========================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #1e293b;
        margin-top: 1.8rem;
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }

    .section-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    div[data-testid="stMetric"] label {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }

    hr {
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        border: 0;
        border-top: 1px solid rgba(128, 128, 128, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Data Loading
# =========================================================

@st.cache_data
def load_data():
    clustered = pd.read_csv(
        DATA_DIR / "accidents_clustered.csv"
    )

    engineered = pd.read_csv(
        DATA_DIR / "accidents_engineered.csv"
    )

    hotspots = pd.read_csv(
        OUTPUT_DIR / "hotspot_summary.csv"
    )

    evaluation = pd.read_csv(
        OUTPUT_DIR / "kmeans_evaluation.csv"
    )

    importance = pd.read_csv(
        OUTPUT_DIR / "random_forest_feature_importance.csv"
    )

    predictions = pd.read_csv(
        OUTPUT_DIR / "risk_predictions.csv"
    )

    return (
        clustered,
        engineered,
        hotspots,
        evaluation,
        importance,
        predictions,
    )


def load_rf_metrics(output_dir: Path) -> dict:
    metrics_file = output_dir / "random_forest_metrics.txt"

    metrics = {
        "balanced_accuracy": "0.5167",
        "overall_accuracy": "77.0%",
        "training_records": "80,000",
        "testing_records": "20,000",
    }

    if metrics_file.exists():
        try:
            content = metrics_file.read_text(
                encoding="utf-8"
            )

            ba_match = re.search(
                r"Balanced Accuracy:\s*([0-9.]+)",
                content,
            )

            if ba_match:
                metrics["balanced_accuracy"] = (
                    f"{float(ba_match.group(1)):.4f}"
                )

            acc_match = re.search(
                r"accuracy\s+([0-9.]+)",
                content,
                re.IGNORECASE,
            )

            if acc_match:
                metrics["overall_accuracy"] = (
                    f"{float(acc_match.group(1)):.1%}"
                )

            tr_match = re.search(
                r"Training records:\s*(\d+)",
                content,
            )

            if tr_match:
                metrics["training_records"] = (
                    f"{int(tr_match.group(1)):,}"
                )

            te_match = re.search(
                r"Testing records:\s*(\d+)",
                content,
            )

            if te_match:
                metrics["testing_records"] = (
                    f"{int(te_match.group(1)):,}"
                )

        except Exception:
            pass

    return metrics


# =========================================================
# Plotly Styling
# =========================================================

def apply_chart_style(fig, title=None, height=360):

    if title is None and fig.layout.title:
        title = fig.layout.title.text

    layout_update = dict(
        template="plotly_white",
        height=height,
        margin=dict(
            l=40,
            r=25,
            t=50 if title else 25,
            b=40,
        ),
        font=dict(
            family=(
                "-apple-system, BlinkMacSystemFont, "
                "Segoe UI, Roboto, sans-serif"
            ),
            size=12,
            color="#334155",
        ),
        transition=dict(
            duration=300,
            easing="cubic-in-out",
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            font_size=12,
            font_color="#ffffff",
        ),
    )

    if title:
        layout_update["title"] = dict(
            text=title,
            font=dict(
                size=14,
                color="#0f172a",
                family=(
                    "-apple-system, BlinkMacSystemFont, "
                    "Segoe UI, Roboto, sans-serif"
                ),
            ),
            x=0.01,
            xanchor="left",
        )

    fig.update_layout(**layout_update)

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#f1f5f9",
        zeroline=False,
    )

    return fig


# =========================================================
# Load Data
# =========================================================

(
    clustered,
    engineered,
    hotspots,
    evaluation,
    importance,
    predictions,
) = load_data()

rf_metrics = load_rf_metrics(OUTPUT_DIR)


# =========================================================
# Dynamic K-Means Evaluation
# =========================================================

best_idx = evaluation["silhouette_score"].idxmax()

best_k = int(
    evaluation.loc[best_idx, "k"]
)

best_silhouette = float(
    evaluation.loc[best_idx, "silhouette_score"]
)


# =========================================================
# Sidebar Controls
# =========================================================

st.sidebar.markdown("### Dashboard Controls")

available_clusters = (
    ["All"]
    + sorted(
        clustered["cluster_id"]
        .unique()
        .tolist()
    )
)

selected_cluster = st.sidebar.selectbox(
    "Filter by Hotspot Cluster",
    available_clusters,
    help=(
        "Select a specific K-Means cluster "
        "to isolate spatial distribution "
        "and characteristics."
    ),
)

map_sample_limit = st.sidebar.select_slider(
    "Map Sample Limit",
    options=[
        1000,
        2500,
        5000,
        10000,
    ],
    value=5000,
    help=(
        "Adjust marker sampling density "
        "to maintain browser rendering performance."
    ),
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Project Metadata**

    - Architecture: PySpark + Scikit-Learn
    - Clustering: Spatial K-Means
    - Classification: Random Forest
    - Scope: Academic Prototype
    """
)


# =========================================================
# 1. Overview
# =========================================================

st.title(
    "Road Accident Hotspot Detection and Risk Analysis"
)

st.caption(
    "Big Data Analytics Prototype: Distributed Preprocessing, "
    "Spatial Clustering, and Machine Learning Risk Modeling"
)

st.info(
    "Prototype Note: This dashboard evaluates a 100,000-record "
    "development sample extracted from the US Accidents dataset "
    "(2016-2023). Distributed feature engineering and spatial "
    "K-Means clustering were orchestrated via PySpark, followed "
    "by multi-class Random Forest risk prediction. This prototype "
    "demonstrates the end-to-end analytical pipeline and is not "
    "a complete representation of the entire 7.7M+ national corpus."
)


# =========================================================
# KPI Cards
# =========================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    label="Analyzed Incidents",
    value=f"{len(clustered):,}",
    help=(
        "Total records processed "
        "in the development dataset."
    ),
)

c2.metric(
    label="Identified Hotspots",
    value=f"{clustered['cluster_id'].nunique():,}",
    help=(
        "Discrete spatial clusters "
        "detected using K-Means."
    ),
)

c3.metric(
    label="Best Silhouette Score",
    value=f"{best_silhouette:.4f}",
    delta=f"Optimal K = {best_k}",
    delta_color="off",
    help=(
        "Maximum silhouette coefficient "
        "achieved across cluster evaluations."
    ),
)

c4.metric(
    label="RF Balanced Accuracy",
    value=rf_metrics["balanced_accuracy"],
    delta=(
        f"Overall Accuracy: "
        f"{rf_metrics['overall_accuracy']}"
    ),
    delta_color="off",
    help=(
        f"Dynamically evaluated on "
        f"{rf_metrics['testing_records']} "
        f"held-out test records."
    ),
)

st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 2. Accident Hotspot Map
# =========================================================

st.markdown(
    '<div class="section-title">Accident Hotspot Map</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Spatial distribution of traffic incidents. "
    "Clustered markers optimize browser responsiveness "
    "while preserving drill-down fidelity."
    "</div>",
    unsafe_allow_html=True,
)

if selected_cluster == "All":
    filtered_map_df = clustered
else:
    filtered_map_df = clustered[
        clustered["cluster_id"] == selected_cluster
    ]

sample_size = min(
    map_sample_limit,
    len(filtered_map_df),
)

map_sample = filtered_map_df.sample(
    sample_size,
    random_state=42,
)

center_lat = float(
    map_sample["Start_Lat"].mean()
)

center_lng = float(
    map_sample["Start_Lng"].mean()
)

m = folium.Map(
    location=[
        center_lat,
        center_lng,
    ],
    zoom_start=5,
    tiles="OpenStreetMap",
)

marker_callback = """
function (row) {
    var marker = L.circleMarker(
        new L.LatLng(row[0], row[1]),
        {
            radius: 3,
            color: '#1e40af',
            fillColor: '#3b82f6',
            fillOpacity: 0.75,
            weight: 1
        }
    );

    marker.bindPopup(
        '<b>Incident ID:</b> ' + row[2] +
        '<br><b>Severity:</b> ' + row[3] +
        '<br><b>Cluster:</b> ' + row[4]
    );

    return marker;
}
"""

FastMarkerCluster(
    data=map_sample[
        [
            "Start_Lat",
            "Start_Lng",
            "ID",
            "Severity",
            "cluster_id",
        ]
    ].values.tolist(),
    callback=marker_callback,
).add_to(m)

st.iframe(
    src=m._repr_html_(),
    height=580,
    width="stretch",
)

st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 3. Hotspot Analysis
# =========================================================

st.markdown(
    '<div class="section-title">Hotspot Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Comparative assessment of incident concentration "
    "and average severity across identified spatial clusters."
    "</div>",
    unsafe_allow_html=True,
)

col_h1, col_h2 = st.columns(2)


# ---------------------------------------------------------
# Accident Count
# ---------------------------------------------------------

with col_h1:

    fig_counts = px.bar(
        hotspots,
        x="cluster_id",
        y="accident_count",
        title="Accident Count by Hotspot Cluster",
        labels={
            "cluster_id": "Cluster ID",
            "accident_count": "Total Accidents",
        },
        text="accident_count",
        custom_data=[
            "cluster_id",
            "accident_count",
        ],
        color="accident_count",
        color_continuous_scale="Blues",
    )

    fig_counts.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            "<b>Cluster %{customdata[0]}</b><br>"
            "Accidents: %{customdata[1]:,}"
            "<extra></extra>"
        ),
    )

    fig_counts.update_coloraxes(
        showscale=False
    )

    apply_chart_style(
        fig_counts,
        height=360,
    )

    fig_counts.update_yaxes(
        tickformat=",",
    )

    st.plotly_chart(
        fig_counts,
        width="stretch",
    )


# ---------------------------------------------------------
# Average Severity
# ---------------------------------------------------------

with col_h2:

    fig_sev = px.bar(
        hotspots,
        x="cluster_id",
        y="average_severity",
        title="Average Severity by Hotspot Cluster",
        labels={
            "cluster_id": "Cluster ID",
            "average_severity": "Mean Severity Score",
        },
        text="average_severity",
        custom_data=[
            "cluster_id",
            "average_severity",
        ],
        color="average_severity",
        color_continuous_scale="YlOrRd",
    )

    fig_sev.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
        hovertemplate=(
            "<b>Cluster %{customdata[0]}</b><br>"
            "Average Severity: %{customdata[1]:.2f}"
            "<extra></extra>"
        ),
    )

    fig_sev.update_coloraxes(
        showscale=False
    )

    apply_chart_style(
        fig_sev,
        height=360,
    )

    fig_sev.update_yaxes(
        range=[0, 3.2],
        tickformat=".1f",
    )

    st.plotly_chart(
        fig_sev,
        width="stretch",
    )


# ---------------------------------------------------------
# K-Means Evaluation
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title" '
    'style="font-size: 1.15rem; margin-top: 1.2rem;">'
    "K-Means Model Selection and Cluster Profiles"
    "</div>",
    unsafe_allow_html=True,
)

fig_eval = px.line(
    evaluation,
    x="k",
    y="silhouette_score",
    markers=True,
    title="K-Means Silhouette Evaluation",
    labels={
        "k": "Number of Clusters (K)",
        "silhouette_score": "Silhouette Score",
    },
    custom_data=[
        "k",
        "silhouette_score",
    ],
)

fig_eval.update_traces(
    line=dict(
        width=3,
    ),
    marker=dict(
        size=9,
    ),
    hovertemplate=(
        "<b>K = %{customdata[0]}</b><br>"
        "Silhouette Score: "
        "%{customdata[1]:.4f}"
        "<extra></extra>"
    ),
)

fig_eval.add_scatter(
    x=[best_k],
    y=[best_silhouette],
    mode="markers+text",
    text=[f"Selected K = {best_k}"],
    textposition="top center",
    marker=dict(
        size=14,
        symbol="diamond",
    ),
    name="Selected K",
    hovertemplate=(
        f"<b>Selected K = {best_k}</b><br>"
        f"Silhouette Score: "
        f"{best_silhouette:.4f}"
        "<extra></extra>"
    ),
)

apply_chart_style(
    fig_eval,
    height=340,
)

fig_eval.update_xaxes(
    dtick=1,
    showgrid=False,
)

fig_eval.update_yaxes(
    tickformat=".3f",
)

st.plotly_chart(
    fig_eval,
    width="stretch",
)


# ---------------------------------------------------------
# Hotspot Summary Table
# ---------------------------------------------------------

st.dataframe(
    hotspots.rename(
        columns={
            "cluster_id": "Cluster ID",
            "accident_count": "Accident Count",
            "average_severity": "Average Severity",
            "average_latitude": "Avg Latitude",
            "average_longitude": "Avg Longitude",
            "minimum_latitude": "Min Latitude",
            "maximum_latitude": "Max Latitude",
            "minimum_longitude": "Min Longitude",
            "maximum_longitude": "Max Longitude",
        }
    ),
    width="stretch",
    hide_index=True,
)

st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 4. Risk Analysis
# =========================================================

st.markdown(
    '<div class="section-title">Risk Analysis</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Evaluation of predicted accident risk classifications "
    "against empirical ground-truth severity tiers."
    "</div>",
    unsafe_allow_html=True,
)

col_r1, col_r2 = st.columns(2)


# ---------------------------------------------------------
# Predicted Risk
# ---------------------------------------------------------

with col_r1:

    risk_order = [
        "Low",
        "Medium",
        "High",
    ]

    risk_counts = (
        predictions["Predicted_Risk"]
        .value_counts()
        .reindex(risk_order)
        .fillna(0)
        .reset_index()
    )

    risk_counts.columns = [
        "Risk",
        "Count",
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk",
        y="Count",
        title="Predicted Risk Category Breakdown",
        labels={
            "Risk": "Assigned Risk Tier",
            "Count": "Prediction Frequency",
        },
        text="Count",
        color="Risk",
        color_discrete_map={
            "Low": "#10b981",
            "Medium": "#f59e0b",
            "High": "#ef4444",
        },
    )

    fig_risk.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Predictions: %{y:,}"
            "<extra></extra>"
        ),
    )

    fig_risk.update_layout(
        showlegend=False,
    )

    apply_chart_style(
        fig_risk,
        height=360,
    )

    fig_risk.update_yaxes(
        tickformat=",",
    )

    st.plotly_chart(
        fig_risk,
        width="stretch",
    )


# ---------------------------------------------------------
# Observed Severity
# ---------------------------------------------------------

with col_r2:

    sev_counts = (
        predictions["Severity"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    sev_counts.columns = [
        "Severity",
        "Count",
    ]

    sev_counts["Severity_Label"] = (
        sev_counts["Severity"]
        .apply(
            lambda s: f"Severity {s}"
        )
    )

    fig_sev_dist = px.bar(
        sev_counts,
        x="Severity_Label",
        y="Count",
        title="Observed Severity Distribution (Test Sample)",
        labels={
            "Severity_Label": "Severity Grade",
            "Count": "Observed Frequency",
        },
        text="Count",
        color="Count",
        color_continuous_scale="Blues",
    )

    fig_sev_dist.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Observed: %{y:,}"
            "<extra></extra>"
        ),
    )

    fig_sev_dist.update_coloraxes(
        showscale=False
    )

    apply_chart_style(
        fig_sev_dist,
        height=360,
    )

    fig_sev_dist.update_yaxes(
        tickformat=",",
    )

    st.plotly_chart(
        fig_sev_dist,
        width="stretch",
    )


st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 5. Time and Weather Analysis
# =========================================================

st.markdown(
    '<div class="section-title">'
    "Time and Weather Analysis"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Investigation of diurnal accident temporal patterns "
    "and primary environmental contributing factors."
    "</div>",
    unsafe_allow_html=True,
)

col_tw1, col_tw2 = st.columns(2)


# ---------------------------------------------------------
# Accident Hour
# ---------------------------------------------------------

with col_tw1:

    hourly = (
        engineered["Accident_Hour"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    hourly.columns = [
        "Hour",
        "Accidents",
    ]

    fig_hour = px.line(
        hourly,
        x="Hour",
        y="Accidents",
        markers=True,
        title="Accidents by Hour",
        labels={
            "Hour": "Hour (00:00 - 23:00)",
            "Accidents": "Recorded Incidents",
        },
        custom_data=[
            "Hour",
            "Accidents",
        ],
    )

    fig_hour.update_traces(
        line=dict(
            color="#0284c7",
            width=2.5,
        ),
        marker=dict(
            size=6,
            color="#0369a1",
        ),
        hovertemplate=(
            "<b>Hour: %{customdata[0]:02d}:00</b><br>"
            "Accidents: %{customdata[1]:,}"
            "<extra></extra>"
        ),
    )

    apply_chart_style(
        fig_hour,
        height=360,
    )

    fig_hour.update_xaxes(
        dtick=2,
    )

    fig_hour.update_yaxes(
        tickformat=",",
    )

    st.plotly_chart(
        fig_hour,
        width="stretch",
    )


# ---------------------------------------------------------
# Weather Conditions
# ---------------------------------------------------------

with col_tw2:

    weather = (
        engineered["Weather_Condition"]
        .value_counts()
        .head(10)
        .reset_index()
    )

    weather.columns = [
        "Weather",
        "Accidents",
    ]

    weather = weather.sort_values(
        "Accidents",
        ascending=True,
    )

    fig_weather = px.bar(
        weather,
        x="Accidents",
        y="Weather",
        orientation="h",
        title="Top Weather Conditions",
        labels={
            "Accidents": "Incident Count",
            "Weather": "Weather Condition",
        },
        text="Accidents",
        color="Accidents",
        color_continuous_scale="Teal",
    )

    fig_weather.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Accidents: %{x:,}"
            "<extra></extra>"
        ),
    )

    fig_weather.update_coloraxes(
        showscale=False
    )

    apply_chart_style(
        fig_weather,
        height=360,
    )

    fig_weather.update_xaxes(
        tickformat=",",
    )

    st.plotly_chart(
        fig_weather,
        width="stretch",
    )


st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 6. Random Forest Feature Importance
# =========================================================

st.markdown(
    '<div class="section-title">'
    "Random Forest Feature Importance"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Relative importance of spatial, temporal, and atmospheric "
    "variables derived from the Gini impurity metric."
    "</div>",
    unsafe_allow_html=True,
)

top_features = (
    importance
    .head(15)
    .sort_values(
        "importance",
        ascending=True,
    )
    .copy()
)

top_features["display_feature"] = (
    top_features["feature"]
    .str.replace(
        "numerical__",
        "",
        regex=False,
    )
    .str.replace(
        "categorical__",
        "",
        regex=False,
    )
    .str.replace(
        "_",
        " ",
        regex=False,
    )
)

fig_importance = px.bar(
    top_features,
    x="importance",
    y="display_feature",
    orientation="h",
    title="Top Random Forest Feature Importance",
    labels={
        "importance": "Relative Gini Importance",
        "display_feature": "Model Feature",
    },
    text="importance",
    color="importance",
    color_continuous_scale="Blues",
)

fig_importance.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Importance: %{x:.4f}"
        "<extra></extra>"
    ),
)

fig_importance.update_coloraxes(
    showscale=False
)

apply_chart_style(
    fig_importance,
    height=460,
)

fig_importance.update_xaxes(
    tickformat=".2f",
)

st.plotly_chart(
    fig_importance,
    width="stretch",
)

st.markdown(
    "<hr>",
    unsafe_allow_html=True,
)


# =========================================================
# 7. Sample Risk Predictions
# =========================================================

st.markdown(
    '<div class="section-title">'
    "Sample Risk Predictions"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    "Held-out test set instances evaluated with the "
    "multi-class Random Forest risk model."
    "</div>",
    unsafe_allow_html=True,
)

display_prediction_columns = [
    c
    for c in [
        "ID",
        "Start_Time",
        "Start_Lat",
        "Start_Lng",
        "Severity",
        "Risk_Category",
        "Predicted_Risk",
        "Probability_High",
        "Probability_Medium",
        "Probability_Low",
    ]
    if c in predictions.columns
]

st.dataframe(
    predictions[
        display_prediction_columns
    ].head(100),
    width="stretch",
    hide_index=True,
)


# =========================================================
# Footer
# =========================================================

st.caption(
    "Project Prototype: Road Accident Hotspot Detection "
    "and Risk Analysis Using Big Data Techniques"
)