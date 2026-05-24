FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --default-timeout=120 \
    scikit-learn>=1.6 \
    pandas>=2.2 \
    numpy>=2.1 \
    joblib>=1.3 \
    bentoml>=1.4 \
    dill>=0.3

COPY deployment/service.py service.py
COPY models/scaler.pkl models/scaler.pkl
COPY models/production/segmentation_model.pkl models/production/segmentation_model.pkl

EXPOSE 3000

CMD ["bentoml", "serve", "service.py:SegmentationService", "--host", "0.0.0.0", "--port", "3000"]
