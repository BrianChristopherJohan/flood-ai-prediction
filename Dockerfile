FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train models during build so the image is self-contained.
# If pre-trained .pkl files are already present they will be reused.
RUN python scripts/train.py

# Railway injects PORT at runtime; default to 8000 for local Docker.
EXPOSE 8080

# Single worker — enough for a stateless prediction service on Hobby plan.
# Use shell form so ${PORT:-8000} is expanded at container startup.
CMD ["/bin/sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --no-access-log"]
