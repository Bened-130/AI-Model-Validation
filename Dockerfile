# ============================================================================
# AI VALIDATION FRAMEWORK - PRODUCTION DOCKERFILE
# ============================================================================
# This Dockerfile creates a production-ready container for the AI Validation
# Framework with multiple service options (API, Streamlit, or Jupyter)
# ============================================================================

# -----------------------------------------------------------------------------
# STAGE 1: Base Image with Dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies required for ML libraries
# - gcc: Required for compiling Python packages with C extensions
# - libgomp1: Required for OpenMP (used by scikit-learn, XGBoost)
# - curl: For health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# -----------------------------------------------------------------------------
# STAGE 2: Python Dependencies
# -----------------------------------------------------------------------------
FROM base as dependencies

# Copy only requirements first (for layer caching)
COPY requirements.txt .
COPY requirements-dev.txt .

# Install production dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# STAGE 3: Application Code
# -----------------------------------------------------------------------------
FROM dependencies as application

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Install the package itself
COPY setup.py pyproject.toml ./
RUN pip install -e .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose common ports (actual usage depends on service)
EXPOSE 8000 8501 8888 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "ai_validation"]

# -----------------------------------------------------------------------------
# STAGE 4: API Service (FastAPI)
# -----------------------------------------------------------------------------
FROM application as api

# Install API-specific dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY services/api/ ./services/api/

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# STAGE 5: Streamlit Demo Service
# -----------------------------------------------------------------------------
FROM application as streamlit

# Streamlit is already in requirements, but ensure it's there
RUN pip install --no-cache-dir streamlit plotly

COPY demo/ ./demo/

EXPOSE 8501

# Set Streamlit configuration
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "demo/app.py"]

# -----------------------------------------------------------------------------
# STAGE 6: Jupyter Notebook Service
# -----------------------------------------------------------------------------
FROM application as jupyter

RUN pip install --no-cache-dir jupyter jupyterlab

COPY notebooks/ ./notebooks/

EXPOSE 8888

# Generate config and set password (change in production!)
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.password = 'sha1:your_hashed_password_here'" >> ~/.jupyter/jupyter_notebook_config.py

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# -----------------------------------------------------------------------------
# STAGE 7: MLflow Tracking Server
# -----------------------------------------------------------------------------
FROM application as mlflow

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

# Create artifacts directory
RUN mkdir -p /app/mlruns

CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:///app/mlruns/mlflow.db", \
     "--default-artifact-root", "/app/mlruns", \
     "--host", "0.0.0.0", \
     "--port", "5000"]