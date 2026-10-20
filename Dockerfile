# 1. Use an official, lightweight Python base image
FROM python:3.12-slim AS runtime-base

# 2. Set environment variables to prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Allow MLflow to use the bundled local file store.
ENV MLFLOW_ALLOW_FILE_STORE=true
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Install system dependencies (required for pandas/scikit-learn on slim images)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 6. Copy ONLY the production requirements and install them
COPY requirements_prod.txt .
RUN pip install --no-cache-dir -r requirements_prod.txt

# CI image: install dependencies and verify application imports without
# bundling runtime datasets, trained models, or MLflow artifacts.
FROM runtime-base AS ci
COPY src/ ./src/
COPY configs/ ./configs/
RUN python -c "import src.models.model; import src.serve.app"

# Production image: includes the model and data artifacts required for inference.
FROM runtime-base AS production

# 7. Copy the specific assets required for inference
COPY src/ ./src/
COPY configs/ ./configs/
COPY mlruns/ ./mlruns/
COPY datasets/processed/ ./datasets/processed/

# 8. Expose the port FastAPI runs on
EXPOSE 8000

# 9. Boot the Uvicorn server, binding to all interfaces (0.0.0.0) so Docker can route traffic
CMD ["sh", "-c", "uvicorn src.serve.app:app --host $APP_HOST --port $APP_PORT"]