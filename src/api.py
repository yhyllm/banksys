"""FastAPI prediction service for Bank Marketing model."""

from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.preprocess import _add_features
from src.train import load_model


# ---- Pydantic models for input validation ----------------------------------


class CustomerFeatures(BaseModel):
    """Input features for a single bank customer."""

    age: int = Field(..., ge=17, le=100, description="Age in years")
    job: str = Field(..., description="Occupation type")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default? (yes/no/unknown)")
    housing: str = Field(..., description="Has housing loan? (yes/no/unknown)")
    loan: str = Field(..., description="Has personal loan? (yes/no/unknown)")
    contact: str = Field(..., description="Contact type (cellular/telephone)")
    month: str = Field(..., description="Last contact month")
    day_of_week: str = Field(..., description="Last contact day of week")
    duration: int = Field(..., ge=0, description="Last contact duration (seconds)")
    campaign: int = Field(..., ge=1, description="Contacts during this campaign")
    pdays: int = Field(
        ..., ge=0, description="Days since last contact (999 = never contacted)"
    )
    previous: int = Field(..., ge=0, description="Contacts before this campaign")
    poutcome: str = Field(..., description="Previous campaign outcome")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_index: float = Field(..., description="Consumer price index")
    cons_conf_index: float = Field(..., description="Consumer confidence index")
    lending_rate3m: float = Field(..., description="3-month lending rate")
    nr_employed: float = Field(..., description="Number of employees (quarterly)")


class BatchRequest(BaseModel):
    """Request body for batch prediction."""

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=1000, description="List of customer records"
    )


class PredictionResult(BaseModel):
    """Response for a single prediction."""

    subscribe: int = Field(..., description="Predicted class (0=no, 1=yes)")
    probability: float = Field(..., description="Probability of class 1 (purchase)")


class BatchResult(BaseModel):
    """Response for batch prediction."""

    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str


# ---- Lifespan: load model once at startup ----------------------------------


_model_data: Optional[dict] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup, release on shutdown."""
    global _model_data
    _model_data = load_model()
    yield
    _model_data = None


# ---- FastAPI app -----------------------------------------------------------

app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict whether a bank customer will subscribe to a term deposit.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow cross-origin requests from the frontend (port 8050)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Helpers ---------------------------------------------------------------


def _input_to_features(customer: CustomerFeatures) -> pd.DataFrame:
    """Convert a validated CustomerFeatures to a DataFrame ready for preprocessing."""
    df = pd.DataFrame([customer.model_dump()])
    return _add_features(df)


def _predict_one(customer: CustomerFeatures) -> PredictionResult:
    """Run a single prediction and return the result."""
    if _model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    df = _input_to_features(customer)
    X = _model_data["preprocessor"].transform(df)
    proba = _model_data["model"].predict_proba(X)[0]  # [P(0), P(1)]
    pred = int(proba.argmax())

    return PredictionResult(subscribe=pred, probability=round(float(proba[1]), 4))


# ---- Routes ----------------------------------------------------------------


@app.get("/")
async def root():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if _model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResult)
async def predict_single(customer: CustomerFeatures):
    """Predict for a single customer.

    Returns the predicted class (0=no purchase, 1=purchase)
    and the probability of purchase.
    """
    return _predict_one(customer)


@app.post("/predict/batch", response_model=BatchResult)
async def predict_batch(request: BatchRequest):
    """Predict for multiple customers at once.

    Accepts up to 1000 records per request.
    """
    results = [_predict_one(c) for c in request.customers]
    return BatchResult(predictions=results)
