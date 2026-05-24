import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class CustomerSegmentation:
    """Performs customer segmentation using unsupervised learning.

    Primary algorithm is KMeans for its scalability and interpretability.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: int = 42,
        algorithm: str = "kmeans",
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.model = self._init_model()

    def _init_model(self):
        if self.algorithm == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init="auto",
            )
        elif self.algorithm == "gmm":
            return GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
            )
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def fit_predict(self, df: pd.DataFrame, feature_cols: list[str], wandb_run=None) -> pd.DataFrame:
        X = df[feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X)
        labels = self.model.fit_predict(X_scaled)
        result = df.copy()
        result["segment"] = labels

        if wandb_run is not None:
            import wandb
            centers = self.model.cluster_centers_
            center_df = pd.DataFrame(centers, columns=feature_cols)
            center_df.index.name = "segment"
            reset_df = center_df.reset_index()
            wandb_run.log({"cluster_centers": wandb.Table(dataframe=reset_df)})
            wandb_table = wandb.Table(dataframe=reset_df)
            wandb_run.log({"cluster_centers_plot": wandb.plot.line(
                wandb_table, "segment",
                [c for c in feature_cols if c in wandb_table.columns[1:]],
                title="Cluster Centers"
            )})
            for i, center in enumerate(centers):
                for j, col in enumerate(feature_cols):
                    wandb_run.log({f"segment_{i}_{col}": float(center[j])})

        return result

    @staticmethod
    def compute_elbow(X_scaled: np.ndarray, max_k: int = 10, seed: int = 42) -> pd.DataFrame:
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            km.fit(X_scaled)
            inertias.append({"k": k, "inertia": km.inertia_})
        return pd.DataFrame(inertias)

    @staticmethod
    def compute_silhouette_scores(
        X_scaled: np.ndarray, max_k: int = 10, seed: int = 42
    ) -> pd.DataFrame:
        from sklearn.metrics import silhouette_score

        scores = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            scores.append({"k": k, "silhouette_score": score})
        return pd.DataFrame(scores)
