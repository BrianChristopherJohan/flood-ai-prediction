FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# If pre-trained FYP model files are present alongside the source tree (e.g. copied
# in during CI via sync_models.py --skip-train), reuse them; otherwise run the full
# training pipeline to generate flood_model.pkl + scaler.pkl from scratch.
RUN python scripts/train.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
