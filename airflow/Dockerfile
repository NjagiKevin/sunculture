FROM apache/airflow:2.7.3-python3.11

RUN pip install --no-cache-dir --default-timeout=120 \
    scikit-learn>=1.6 \
    pandas>=2.2 \
    numpy>=2.1 \
    mlflow==2.18.0 \
    python-dotenv>=1.0 \
    openai>=1.55 \
    openpyxl>=3.1 \
    seaborn>=0.13 \
    matplotlib>=3.8 \
    joblib>=1.3 \
    pyarrow>=14.0 \
    wandb>=0.27 \
    plotly>=6.0 \
    'email-validator>=2.0,<3'
