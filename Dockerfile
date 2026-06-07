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

# Train the model during image build (so models/ is baked in)
RUN python -m src.train

# Remove dev dependencies to keep image lean
RUN pip uninstall -y pytest pytest-cov ruff httpx coverage && \
    rm -f requirements-dev.txt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
