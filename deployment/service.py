import joblib
import pandas as pd
from pathlib import Path
from typing import List

import bentoml
from pydantic import BaseModel

SEGMENT_DESCRIPTIONS = {
    0: "Cash Defaulters — All-cash, highest default rate (48.9%), no refurbished products",
    1: "PAYG Defaulters — Exclusively PAYG, worst risk profile (52.1% default), highest arrears",
    2: "Refurbished Buyers — All refurbished units, moderate risk (38.8% default), largest segment",
    3: "Healthy Completers — Zero defaults, zero arrears, 53.4% completed payments, ideal customers",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
FEATURE_COLS = ["account_tenure_days", "risk_score", "is_refurbished", "is_payg"]


class SegmentInput(BaseModel):
    account_tenure_days: float
    risk_score: float
    is_refurbished: float
    is_payg: float


class SegmentOutput(BaseModel):
    segment: int
    description: str


class BatchSegmentInput(BaseModel):
    records: List[SegmentInput]


class BatchSegmentOutput(BaseModel):
    predictions: List[SegmentOutput]


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool


@bentoml.service(
    name="segmentation_service",
    traffic={"max_latency_ms": 3000},
)
class SegmentationService:

    def __init__(self):
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.model = joblib.load(MODEL_DIR / "production" / "segmentation_model.pkl")

    @bentoml.api
    def segment_single(self, input_data: SegmentInput) -> SegmentOutput:
        df = pd.DataFrame([input_data.model_dump()])[FEATURE_COLS]
        X_scaled = self.scaler.transform(df)
        label = int(self.model.predict(X_scaled)[0])
        return SegmentOutput(segment=label, description=SEGMENT_DESCRIPTIONS[label])

    @bentoml.api
    def segment_batch(self, input_data: BatchSegmentInput) -> BatchSegmentOutput:
        rows = [r.model_dump() for r in input_data.records]
        df = pd.DataFrame(rows)[FEATURE_COLS]
        X_scaled = self.scaler.transform(df)
        labels = self.model.predict(X_scaled).astype(int).tolist()
        return BatchSegmentOutput(
            predictions=[SegmentOutput(segment=l, description=SEGMENT_DESCRIPTIONS[l]) for l in labels]
        )

    @bentoml.api
    def health(self) -> HealthOutput:
        return HealthOutput(
            status="ok",
            model_loaded=self.model is not None,
            scaler_loaded=self.scaler is not None,
        )

    @bentoml.api
    def profile(self) -> dict:
        return {"segments": {str(k): v for k, v in SEGMENT_DESCRIPTIONS.items()}}
