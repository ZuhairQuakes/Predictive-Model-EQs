"""Streamlit interface for exploring published megathrust XAI checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from megathrust_xai.data import FEATURE_COLUMNS, StudyData, load_study_data
from megathrust_xai.inference import explain_lrp, predict
from megathrust_xai.model import CheckpointSpec, PublishedNetwork, load_checkpoint

CLASS_LABELS = {
    2: ("Mw < 8.0", "Mw ≥ 8.0"),
    3: ("Mw < 6.4", "6.4 ≤ Mw < 8.3", "Mw ≥ 8.3"),
}


def _root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for candidate in candidates:
        if (candidate / "in-data").is_dir() and (candidate / "models").is_dir():
            return candidate
    raise RuntimeError("Run the app from a repository checkout containing in-data/ and models/.")


@st.cache_resource(show_spinner="Loading the published feature dataset…")
def _study_data(data_directory: str) -> StudyData:
    return load_study_data(data_directory)


@st.cache_resource(show_spinner="Loading checkpoint…")
def _model(path: str) -> tuple[PublishedNetwork, CheckpointSpec]:
    return load_checkpoint(path)


def _checkpoint_label(path: Path) -> str:
    spec = CheckpointSpec.from_path(path)
    return (
        f"{spec.classes} classes · {spec.exclusion_distance_km} km exclusion · "
        f"scenario {spec.scenario} · epoch {spec.epoch}"
    )


def _feature_group(feature: str) -> str:
    if feature.startswith(("CRD", "CRS", "CRM", "INV", "DLT", "SED", "SRO", "IRO", "LRO")):
        return "Physical state"
    if feature.startswith(("V_", "AGE")):
        return "Kinematics"
    return "Dynamics / geometry"


def _format_score(value: float) -> str:
    return f"{value:.3f}"


def _render_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">INTERACTIVE RESEARCH COMPANION</div>
          <h1>What drives a megathrust classification?</h1>
          <p>Explore published subduction-zone features, run a selected neural-network
          checkpoint, and inspect its Layer-wise Relevance Propagation explanation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run() -> None:
    st.set_page_config(
        page_title="Megathrust XAI Explorer",
        page_icon="🌊",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp {background: linear-gradient(145deg, #07131f 0%, #102c3b 55%, #163d43 100%);}
        [data-testid="stSidebar"] {background: #081924; border-right: 1px solid #244556;}
        .hero {padding: 2.2rem 2.4rem; border: 1px solid #31586a; border-radius: 18px;
               background: linear-gradient(120deg, rgba(8,25,36,.95), rgba(15,62,66,.86));
               box-shadow: 0 18px 50px rgba(0,0,0,.22); margin-bottom: 1.2rem;}
        .hero h1 {font-size: clamp(2rem, 5vw, 4.2rem); line-height: 1.02; margin: .3rem 0 .8rem;
                  letter-spacing: -.04em; color: #f4f7f1; max-width: 900px;}
        .hero p {font-size: 1.08rem; color: #b8d0d3; max-width: 850px; margin: 0;}
        .eyebrow {color: #49d6b4; font-weight: 750; letter-spacing: .16em; font-size: .75rem;}
        [data-testid="stMetric"] {background: rgba(7,24,34,.72); border: 1px solid #294d5e;
                                  border-radius: 14px; padding: .75rem 1rem;}
        div[data-testid="stAlert"] {border-radius: 12px;}
        h2, h3 {letter-spacing: -.02em;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    root = _root()
    study = _study_data(str(root / "in-data"))
    checkpoints = sorted((root / "models").glob("ncls*-dex*/*.pt"))
    if not checkpoints:
        st.error("No published checkpoints were found.")
        st.stop()

    _render_header()
    st.warning(
        "Research-use boundary: this explorer reproduces the repository's PowerTransformer "
        "from the bundled 556 complete samples because the fitted transformer was not archived. "
        "Scores are independent sigmoid outputs, not calibrated earthquake probabilities."
    )

    with st.sidebar:
        st.header("Model setup")
        selected_checkpoint = st.selectbox(
            "Published checkpoint",
            checkpoints,
            format_func=_checkpoint_label,
        )
        model, spec = _model(str(selected_checkpoint))
        labels = CLASS_LABELS[spec.classes]
        st.caption(f"Checkpoint: `{selected_checkpoint.relative_to(root)}`")
        st.divider()
        st.header("Segment")
        region_names = sorted(study.frame["REGION_NAME"].unique())
        region = st.selectbox("Subduction region", region_names)
        regional = study.frame.loc[study.frame["REGION_NAME"] == region]
        segment = st.selectbox(
            "Along-trench segment",
            regional.index,
            format_func=lambda index: (
                f"Segment {int(regional.loc[index, 'SEGMENT'])} · "
                f"observed max Mw {regional.loc[index, 'MR_GCMT']:.2f}"
            ),
        )

    source_row = study.frame.loc[segment, list(FEATURE_COLUMNS)].astype(float)
    edited_row = source_row.copy()
    with st.sidebar:
        st.divider()
        st.header("What-if controls")
        editable = st.multiselect(
            "Features to perturb",
            FEATURE_COLUMNS,
            default=["SED_AVE", "CRD_UP_AVE", "INV_UP_AVE"],
            max_selections=6,
        )
        for feature in editable:
            values = study.frame[feature].dropna().astype(float)
            low, high = np.quantile(values, [0.01, 0.99])
            if np.isclose(low, high):
                low, high = float(values.min()), float(values.max())
            current = float(np.clip(source_row[feature], low, high))
            edited_row[feature] = st.slider(
                feature,
                min_value=float(low),
                max_value=float(high),
                value=current,
                format="%.4g",
                key=f"{segment}-{feature}",
                help=f"Bundled-data 1st–99th percentile; original value {source_row[feature]:.4g}",
            )

    transformed = study.transform(edited_row.to_numpy())
    prediction = predict(model, transformed)
    target_class = prediction.predicted_class
    explanation = explain_lrp(model, transformed, target_class)

    overview, explanation_tab, data_tab, method_tab = st.tabs(
        ["Prediction", "LRP explanation", "Study data", "Method & limits"]
    )
    with overview:
        metric_columns = st.columns([1.35, 0.7, 1.1, 1.35])
        metric_columns[0].metric("Region", region)
        metric_columns[1].metric("Segment", int(study.frame.loc[segment, "SEGMENT"]))
        observed_maximum = f"Mw {study.frame.loc[segment, 'MR_GCMT']:.2f}"
        metric_columns[2].metric("Observed maximum", observed_maximum)
        metric_columns[3].metric("Model classification", labels[target_class])
        scores = pd.DataFrame({"Class": labels, "Sigmoid score": prediction.scores})
        figure = px.bar(
            scores,
            x="Class",
            y="Sigmoid score",
            color="Sigmoid score",
            color_continuous_scale=["#173b4a", "#49d6b4", "#ffd166"],
            text=scores["Sigmoid score"].map(_format_score),
            range_y=[0, 1],
        )
        figure.update_layout(coloraxis_showscale=False, template="plotly_dark", height=430)
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "The original training code used one-hot targets with BCEWithLogitsLoss, so each "
            "class receives an independent sigmoid score. Scores do not necessarily sum to one."
        )

    with explanation_tab:
        top_n = st.slider("Features shown", 8, len(FEATURE_COLUMNS), 20)
        relevance = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Relevance": explanation.relevance,
                "Group": [_feature_group(feature) for feature in FEATURE_COLUMNS],
                "Value": edited_row.to_numpy(),
            }
        )
        relevance["Absolute relevance"] = relevance["Relevance"].abs()
        relevance = relevance.nlargest(top_n, "Absolute relevance").sort_values("Relevance")
        relevance["Direction"] = np.where(
            relevance["Relevance"] >= 0,
            "supports selected class",
            "opposes selected class",
        )
        figure = px.bar(
            relevance,
            x="Relevance",
            y="Feature",
            orientation="h",
            color="Direction",
            color_discrete_map={
                "supports selected class": "#49d6b4",
                "opposes selected class": "#ff7b72",
            },
            hover_data=["Value", "Group"],
        )
        figure.update_layout(template="plotly_dark", height=max(480, top_n * 28))
        st.plotly_chart(figure, width="stretch")
        st.caption(
            f"LRP target: {labels[target_class]}. Captum completeness delta: "
            f"{explanation.convergence_delta:.3g}. Relevance is local to this checkpoint, "
            "sample, preprocessing reconstruction, and target class."
        )

    with data_tab:
        feature = st.selectbox("Feature distribution", FEATURE_COLUMNS, index=9)
        distribution = study.frame[["REGION_NAME", feature]].rename(
            columns={"REGION_NAME": "Region", feature: "Value"}
        )
        figure = px.box(
            distribution,
            x="Region",
            y="Value",
            color="Region",
            points="outliers",
        )
        figure.add_hline(
            y=float(edited_row[feature]),
            line_dash="dash",
            line_color="#ffd166",
            annotation_text="selected / edited segment",
        )
        figure.update_layout(template="plotly_dark", showlegend=False, height=500)
        st.plotly_chart(figure, width="stretch")
        st.dataframe(
            regional[["SEGMENT", "MR_GCMT", *editable]].rename(
                columns={"MR_GCMT": "Observed maximum Mw"}
            ),
            hide_index=True,
            width="stretch",
        )

    with method_tab:
        st.markdown(
            """
            ### What this page does

            1. Combines the eight bundled regional CSV files and retains the 556 complete rows.
            2. Fits the same Yeo–Johnson `PowerTransformer` class used by the research code.
            3. Reconstructs the two-hidden-layer network directly from the selected
               weights-only checkpoint.
            4. Reports independent sigmoid class scores and explains the winning
               score with Captum LRP.

            ### What it does not establish

            Feature relevance is not causal influence, physical sensitivity,
            uncertainty, or earthquake probability. The app does not reproduce
            spatial cross-validation metrics, propagate input uncertainty, or
            replace the article and archived workflow. What-if controls can create
            feature combinations outside the joint training distribution even when
            individual sliders remain within their marginal ranges.
            """
        )
        st.link_button(
            "Read the associated article",
            "https://doi.org/10.1029/2024JB028774",
        )


if __name__ == "__main__":
    run()
