import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


SUNCOLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]


class Plotter:
    """Unified plotting utilities with a consistent SunCulture visual theme using Plotly."""

    @staticmethod
    def bar(
        data: pd.Series,
        title: str = "",
        xlabel: str = "",
        ylabel: str = "Count",
        rotate_x: int = 45,
    ):
        fig = go.Figure()
        fig.add_bar(
            x=data.index.tolist(),
            y=data.values.tolist(),
            marker_color=SUNCOLORS[: len(data)],
        )
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            xaxis_tickangle=rotate_x,
            template="simple_white",
        )
        return fig

    @staticmethod
    def radar(
        data: pd.DataFrame,
        categories: list[str],
        title: str = "Segment Profiles",
    ):
        fig = go.Figure()
        for i, row in data.iterrows():
            values = row[categories].tolist()
            values += values[:1]
            theta = categories + [categories[0]]
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                name=str(i),
            ))
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True)),
            template="simple_white",
        )
        return fig

    @staticmethod
    def heatmap_corr(
        df: pd.DataFrame,
        title: str = "Correlation Matrix",
    ):
        corr = df.select_dtypes(include=np.number).corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title=title,
            aspect="auto",
            zmin=-1,
            zmax=1,
        )
        fig.update_layout(template="simple_white")
        return fig

    @staticmethod
    def elbow_silhouette(
        elbow_df: pd.DataFrame,
        sil_df: pd.DataFrame,
    ):
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method", "Silhouette Score"))
        fig.add_trace(
            go.Scatter(x=elbow_df["k"], y=elbow_df["inertia"], mode="lines+markers",
                       marker_color=SUNCOLORS[0]),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=sil_df["k"], y=sil_df["silhouette_score"], mode="lines+markers",
                       marker_color=SUNCOLORS[1]),
            row=1, col=2,
        )
        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=1)
        fig.update_yaxes(title_text="Inertia", row=1, col=1)
        fig.update_xaxes(title_text="Number of clusters (k)", row=1, col=2)
        fig.update_yaxes(title_text="Score", row=1, col=2)
        fig.update_layout(template="simple_white", showlegend=False)
        return fig
