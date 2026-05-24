import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc


class CreditRiskModel:
    """Binary classifier for predicting loan default risk.

    All features passed to ``train()`` MUST be pre-behaviour indicators
    (e.g. region, product type, payment type, tenure).  No status-derived
    columns should be included — those cause target leakage.
    """

    def __init__(self, random_state: int = 42, scale_pos_weight: float | None = None):
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model = self._build_model()
        self.feature_cols = None
        self._train_metrics = {}

    def _build_model(self):
        return GradientBoostingClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
            random_state=self.random_state,
        )

    def train(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "is_default",
        val_split: float = 0.2,
    ):
        self.feature_cols = feature_cols
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        X = X.fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=val_split, random_state=self.random_state, stratify=y
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)

        f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        y_pred_opt = (y_prob >= best_threshold).astype(int)

        report = classification_report(y_test, y_pred_opt, output_dict=True)

        self._train_metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": pr_auc,
            "best_threshold": float(best_threshold),
            "report": report,
            "feature_importances": dict(
                zip(feature_cols, self.model.feature_importances_)
            ),
            "n_features": len(feature_cols),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_default_rate": float(y_test.mean()),
        }
        return self._train_metrics

    def predict_risk(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(df[self.feature_cols])[:, 1]

    @property
    def metrics(self):
        return self._train_metrics
