FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir pandas scikit-learn numpy joblib pyyaml fastapi uvicorn

COPY src/predict.py src/predict.py
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]