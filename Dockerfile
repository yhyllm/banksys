ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

WORKDIR /app

# Copy and install production dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 -i "${PIP_INDEX_URL}" -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY models/ ./models/

# Expose service port
EXPOSE 8000

# Start FastAPI with uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
