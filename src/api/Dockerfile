FROM python:3.10-slim

WORKDIR /app

# Install system deps (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy project and install python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only necessary parts to keep image small
COPY src /app/src
COPY outputs /app/outputs
COPY README.md /app/README.md

ENV MODEL_PATH=/app/outputs/model.pkl
ENV PP_PATH=/app/outputs/preprocessor.joblib
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "src.api.app:app", "--workers", "2"]
