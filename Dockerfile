ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

# Copy dependency files and install
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" \
    -r requirements.txt -r requirements-dev.txt

# Copy source and data
COPY src/ ./src/
COPY data/ ./data/
COPY .streamlit/ ./.streamlit/

# Train the model during image build (so artifacts/ is baked in)
RUN mkdir -p artifacts && python -m src.ml.train

# Remove dev dependencies to keep image lean
RUN pip uninstall -y pytest pytest-cov ruff && \
    rm -f requirements-dev.txt

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8004/_stcore/health || exit 1

CMD ["streamlit", "run", "src/app.py", "--server.port=8004", "--server.address=0.0.0.0"]
