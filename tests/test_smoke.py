from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import CustomerSegmentation, CreditRiskModel

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Senior_Data_Scientist_Assessment_Data.xlsx"


def test_data_loading():
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    assert "Customers" in raw
    assert "Accounts" in raw
    assert "Products" in raw
    assert len(raw["Customers"]) > 0


def test_preprocessing():
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    prep.clean_all()
    c360 = prep.build_customer_360()
    assert "id" in c360.columns
    assert len(c360) > 0


def test_feature_engineering():
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    prep.clean_all()
    c360 = prep.build_customer_360()
    eng = FeatureEngineer(c360)
    features = eng.engineer()
    required = ["account_tenure_days", "risk_score", "is_refurbished", "is_payg"]
    for col in required:
        assert col in features.columns, f"Missing feature: {col}"


def test_segmentation():
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    prep.clean_all()
    c360 = prep.build_customer_360()
    eng = FeatureEngineer(c360)
    features = eng.engineer().dropna()
    FEATURE_COLS = ["account_tenure_days", "risk_score", "is_refurbished", "is_payg"]
    available = [c for c in FEATURE_COLS if c in features.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features[available])
    model = CustomerSegmentation(n_clusters=4, random_state=42)
    result = model.fit_predict(features, available)
    labels = result["segment"].values
    score = silhouette_score(X_scaled, labels)
    assert 0 <= score <= 1
    assert len(result["segment"].unique()) == 4
    assert "segment" in result.columns


def test_credit_risk_model():
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    prep = DataPreprocessor(raw)
    prep.clean_all()
    c360 = prep.build_customer_360()
    eng = FeatureEngineer(c360)
    features = eng.engineer_credit()
    feature_cols = [c for c in features.columns if c != "is_default"]
    model = CreditRiskModel()
    metrics = model.train(features, feature_cols)
    assert "pr_auc" in metrics
    assert 0 <= metrics["pr_auc"] <= 1


def test_viz_plotter():
    from src.viz import Plotter
    plotter = Plotter()
    accounts = pd.read_excel(DATA_PATH, sheet_name="Accounts")
    status_counts = accounts["status"].value_counts()
    fig = plotter.bar(status_counts, title="Payment Status")
    assert fig is not None
    fig = plotter.heatmap_corr(accounts.select_dtypes(include="number"))
    assert fig is not None
