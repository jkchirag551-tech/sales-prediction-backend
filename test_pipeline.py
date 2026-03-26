import pandas as pd
import joblib
import os

MODEL_PATH = 'd:\\SalesPredictor\\SalesPredictorML\\sales_pipeline_latest.joblib'

def test_model_exists():
    assert os.path.exists(MODEL_PATH), f"Model file {MODEL_PATH} not found."

def test_pipeline_serialization_and_prediction():
    pipeline = joblib.load(MODEL_PATH)

    mock_data = pd.DataFrame([{
        'social_ad_spend': 5000.0,
        'search_ad_spend': 3000.0,
        'discount_percent': 10.0,
        'is_holiday': 0,
        'industry': 'Tech',
        'location': 'Bengaluru'
    }])

    prediction = pipeline.predict(mock_data)
    assert len(prediction) == 1, 'Prediction should return exactly one item.'
    assert isinstance(float(prediction[0]), float), 'Prediction output must be a float.'
