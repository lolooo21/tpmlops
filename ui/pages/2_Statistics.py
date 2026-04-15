from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
UI_DIR = CURRENT_DIR.parents[0]
PROJECT_ROOT = CURRENT_DIR.parents[1]
for path in (str(UI_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ui.bootstrap import ensure_project_root_on_path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ensure_project_root_on_path()
from ui.data_access import load_predicted_trip_durations, load_training_trip_durations


def build_histogram_figure(
    training_durations: pd.Series,
    predicted_durations: pd.Series,
    bins: int,
) -> plt.Figure:
    # Both distributions share the same axes so users can compare spread and
    # central tendency directly without switching charts.
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        training_durations,
        bins=bins,
        alpha=0.55,
        color="#2563eb",
        label="Train",
    )
    axis.hist(
        predicted_durations,
        bins=bins,
        alpha=0.55,
        color="#dc2626",
        label="Predictions",
    )
    axis.set_title("Distribution des durees de trajet")
    axis.set_xlabel("Duree (secondes)")
    axis.set_ylabel("Frequence")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    return figure


def render_summary_metrics(training_durations: pd.Series, predicted_durations: pd.Series) -> None:
    # Summary cards complement the histograms with a few quick aggregate checks.
    col1, col2, col3 = st.columns(3)
    col1.metric("Nb trajets train", f"{len(training_durations):,}".replace(",", " "))
    col2.metric("Nb predictions", f"{len(predicted_durations):,}".replace(",", " "))
    col3.metric(
        "Moyenne predictions",
        f"{predicted_durations.mean():.0f} sec" if not predicted_durations.empty else "N/A",
    )


def main() -> None:
    st.set_page_config(page_title="NYC Taxi UI - Statistiques", page_icon=":bar_chart:", layout="wide")

    st.title("Statistiques de distribution")
    st.write(
        "Cette page compare la distribution des durees du jeu d'entrainement "
        "avec la distribution des valeurs predites enregistrees par l'API."
    )

    bins = st.slider("Nombre de classes de l'histogramme", min_value=10, max_value=80, value=30)

    try:
        training_durations = load_training_trip_durations()
        predicted_durations = load_predicted_trip_durations()
    except Exception as exc:
        st.error(f"Impossible de charger les statistiques depuis SQLite: {exc}")
        return

    if training_durations.empty:
        st.error("La table `train` est vide ou inaccessible. Recharge les donnees avant d'afficher les statistiques.")
        return

    if predicted_durations.empty:
        st.warning(
            "Aucune prediction enregistree pour le moment. Lance au moins une prediction "
            "dans la page principale pour alimenter l'histogramme des valeurs predites."
        )

    render_summary_metrics(training_durations, predicted_durations)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jeu d'entrainement")
        st.write(training_durations.describe().to_frame(name="trip_duration"))
    with col2:
        st.subheader("Predictions")
        if predicted_durations.empty:
            st.info("Pas encore de valeurs predites a resumer.")
        else:
            st.write(predicted_durations.describe().to_frame(name="prediction"))

    st.subheader("Histogrammes")
    if predicted_durations.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        axis.hist(training_durations, bins=bins, color="#2563eb", alpha=0.7)
        axis.set_title("Distribution des durees du jeu d'entrainement")
        axis.set_xlabel("Duree (secondes)")
        axis.set_ylabel("Frequence")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        st.pyplot(figure)
        return

    st.pyplot(build_histogram_figure(training_durations, predicted_durations, bins))


if __name__ == "__main__":
    main()
