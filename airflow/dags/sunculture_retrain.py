import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import CustomerSegmentation

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Senior_Data_Scientist_Assessment_Data.xlsx"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MLFLOW_URI = "http://mlflow:5000"
FEATURE_COLS = ["account_tenure_days", "risk_score", "is_refurbished", "is_payg"]
N_CLUSTERS = 4
SEED = 42
WANDB_PROJECT = "sunculture-segmentation"
WANDB_ENTITY = None

def _wandb_init(run_config: dict | None = None):
    import wandb
    api_key = os.environ.get("WANDB_API_KEY", "")
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    if not api_key:
        return None
    try:
        wandb.login(key=api_key, host=base_url, anonymous="never", verify=False)
        kwargs = dict(project=WANDB_PROJECT, config=run_config,
                      settings=wandb.Settings(start_method="thread"))
        if WANDB_ENTITY:
            kwargs["entity"] = WANDB_ENTITY
        return wandb.init(**kwargs)
    except Exception:
        return None

default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _load_and_prepare() -> pd.DataFrame:
    loader = DataLoader(DATA_PATH)
    raw = loader.load_all()
    preprocessor = DataPreprocessor(raw)
    preprocessor.clean_all()
    customer_360 = preprocessor.build_customer_360()
    engineer = FeatureEngineer(customer_360)
    seg_features = engineer.engineer()
    return seg_features.dropna()


def extract_features(**context):
    log = context["task_instance"].log
    df = _load_and_prepare()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MODEL_DIR / "features.parquet")
    log.info("Features extracted: shape=%s, rows=%d", df.shape, len(df))


def train_model(**context):
    import mlflow
    import joblib
    import wandb

    log = context["task_instance"].log
    df = pd.read_parquet(MODEL_DIR / "features.parquet")
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    seg_model = CustomerSegmentation(n_clusters=N_CLUSTERS, random_state=SEED)
    result = seg_model.fit_predict(df, feature_cols)
    labels = result["segment"].values
    score = silhouette_score(X_scaled, labels)

    result.to_parquet(MODEL_DIR / "segments.parquet")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    candidate_path = MODEL_DIR / "segmentation_model_candidate.pkl"
    joblib.dump(seg_model.model, candidate_path)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sunculture-segmentation")
    with mlflow.start_run(run_name=f"retrain_{datetime.now():%Y%m%d_%H%M}") as run:
        mlflow.log_params({
            "n_clusters": N_CLUSTERS,
            "algorithm": seg_model.algorithm,
            "features": feature_cols,
            "seed": SEED,
        })
        mlflow.log_metric("silhouette_score", round(score, 4))
        mlflow.sklearn.log_model(
            seg_model.model,
            "model",
            input_example=X_scaled[:1],
        )
        run_id = run.info.run_id

    wandb_run = _wandb_init({
        "n_clusters": N_CLUSTERS,
        "algorithm": seg_model.algorithm,
        "features": feature_cols,
        "seed": SEED,
    })
    if wandb_run is not None:
        wandb_run.log({"silhouette_score": round(score, 4)})
        profile_df = result.groupby("segment")[feature_cols + ["is_complete", "in_arrears", "is_default"]].mean()
        wandb_run.log({"segment_profile": wandb.Table(dataframe=profile_df.reset_index())})
        for segment_id in sorted(result["segment"].unique()):
            size = int((result["segment"] == segment_id).sum())
            default_rate = float(result[result["segment"] == segment_id]["is_default"].mean())
            wandb_run.log({f"segment_{segment_id}_size": size, f"segment_{segment_id}_default_rate": default_rate})
        wandb_run.log_artifact(str(candidate_path), type="model", name="segmentation_model", aliases=["latest", f"silhouette_{score:.4f}"])
        wandb_run.finish()
        log.info("wandb run logged: %s", wandb_run.url)

    with open(MODEL_DIR / "run_id.txt", "w") as f:
        f.write(run_id)
    log.info("Model trained: silhouette=%.4f, run_id=%s", score, run_id)


def validate_model(**context):
    import mlflow
    import shutil

    log = context["task_instance"].log
    mlflow.set_tracking_uri(MLFLOW_URI)
    with open(MODEL_DIR / "run_id.txt") as f:
        new_run_id = f.read().strip()
    new_score = mlflow.get_run(new_run_id).data.metrics["silhouette_score"]
    experiment = mlflow.get_experiment_by_name("sunculture-segmentation")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=10,
    )
    champion_score = -1.0
    if len(runs) > 1:
        prev_scores = [s for s in runs.iloc[1:]["metrics.silhouette_score"] if s is not None]
        if prev_scores:
            champion_score = max(prev_scores)

    candidate_path = MODEL_DIR / "segmentation_model_candidate.pkl"
    champion_path = MODEL_DIR / "segmentation_model.pkl"

    if new_score >= champion_score:
        shutil.copy2(candidate_path, champion_path)  # promote candidate to champion
        with open(MODEL_DIR / "champion_run_id.txt", "w") as f:
            f.write(new_run_id)
        log.info("Candidate PROMOTED to champion (silhouette=%.4f >= %.4f)", new_score, champion_score)
    else:
        log.info("Candidate REJECTED (silhouette=%.4f < champion=%.4f). Champion unchanged.", new_score, champion_score)


def deploy_model(**context):
    import joblib
    import shutil

    log = context["task_instance"].log
    production_dir = MODEL_DIR / "production"
    production_dir.mkdir(parents=True, exist_ok=True)

    champion_path = MODEL_DIR / "segmentation_model.pkl"
    candidate_path = MODEL_DIR / "segmentation_model_candidate.pkl"
    dst_model = production_dir / "segmentation_model.pkl"

    if champion_path.exists():
        shutil.copy2(champion_path, dst_model)
        champ_run_id = ""
        if (MODEL_DIR / "champion_run_id.txt").exists():
            champ_run_id = (MODEL_DIR / "champion_run_id.txt").read_text().strip()
        log.info("Deployed champion model (run_id=%s) to production", champ_run_id)
    elif candidate_path.exists():
        shutil.copy2(candidate_path, dst_model)
        log.info("No champion yet — deployed candidate model to production (first run)")
    else:
        raise FileNotFoundError("No model file found to deploy")

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    log.info("Deployment complete: model at %s, scaler loaded=%s", dst_model, scaler is not None)


with DAG(
    dag_id="sunculture_retrain",
    default_args=default_args,
    description="Weekly segmentation model retraining pipeline",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "segmentation"],
) as dag:

    t1 = PythonOperator(task_id="extract_features", python_callable=extract_features)
    t2 = PythonOperator(task_id="train_model", python_callable=train_model)
    t3 = PythonOperator(task_id="validate_model", python_callable=validate_model)
    t4 = PythonOperator(task_id="deploy_model", python_callable=deploy_model)

    t1 >> t2 >> t3 >> t4
