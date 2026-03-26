import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn
import pandas as pd
import joblib
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

from train_model import train_and_save_model, cleanup_old_checkpoints

# Load environment variables from the .env file
load_dotenv()

# --- 1. Security & Logging Setup ---
# Securely fetch the API key
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    # Use the same default key as the Android App for easier development
    API_KEY = "internship_secret_key_2026"
    logger.warning("API_KEY environment variable not set! Using default developer key.")

api_key_header = APIKeyHeader(name="X-API-Key")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 2. Database Setup (SQLite) ---
engine = create_engine("sqlite:///sales_logs.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    industry = Column(String)
    location = Column(String)
    predicted_sales = Column(Float)

Base.metadata.create_all(bind=engine)

# --- 3. Authentication Function ---
def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized access. Invalid API Key.")

# --- 4. Enums & Blueprints ---
class IndustryEnum(str, Enum):
    retail = 'Retail'
    tech = 'Tech'
    food = 'Food'
    healthcare = 'Healthcare'
    fashion = 'Fashion'
    electronics = 'Electronics'
    automotive = 'Automotive'
    finance = 'Finance'

class LocationEnum(str, Enum):
    bengaluru = 'Bengaluru'
    mumbai = 'Mumbai'
    delhi = 'Delhi'
    chennai = 'Chennai'
    hyderabad = 'Hyderabad'
    pune = 'Pune'
    kolkata = 'Kolkata'
    ahmedabad = 'Ahmedabad'

class SalesRequest(BaseModel):
    social_ad_spend: float = Field(..., ge=0)
    search_ad_spend: float = Field(..., ge=0)
    discount_percent: float = Field(..., ge=0, le=100)
    is_holiday: int = Field(..., ge=0, le=1)
    industry: IndustryEnum
    location: LocationEnum

# --- 5. App Initialization & Scheduler ---
def scheduled_retraining():
    logger.info("Starting scheduled model retraining...")
    train_and_save_model()
    cleanup_old_checkpoints()
    global model_pipeline
    model_pipeline = joblib.load('sales_pipeline_latest.joblib')

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduled_retraining, trigger='cron', day_of_week='sun', hour=2, minute=0)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="Secure Sales ML API", lifespan=lifespan)
model_pipeline = joblib.load('sales_pipeline_latest.joblib')

# --- 6. Secured Endpoints ---
@app.post("/predict")
async def predict_sales(request: SalesRequest, api_key: str = Depends(verify_api_key)):
    try:
        input_data = pd.DataFrame([request.model_dump()])
        prediction = model_pipeline.predict(input_data)
        final_sales = float(prediction[0])
        
        # Log to Database
        db = SessionLocal()
        new_log = PredictionLog(
            industry=request.industry.value,
            location=request.location.value,
            predicted_sales=final_sales
        )
        db.add(new_log)
        db.commit()
        db.close()
        
        return {"predicted_sales": round(final_sales, 2)}
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error processing the prediction")

@app.get("/logs", tags=["Admin"])
async def get_prediction_logs(limit: int = 10, api_key: str = Depends(verify_api_key)):
    """Fetch the most recent prediction logs from the SQLite database."""
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
        return {"logs": logs}
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
