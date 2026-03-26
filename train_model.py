import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
from datetime import datetime
import os
import glob
import time


def train_and_save_model():
    """Trains the model, saves a timestamped copy, and updates the latest alias."""
    print("Generating data and training new model...")
    np.random.seed(42)
    n_samples = 1000

    df = pd.DataFrame({
        'social_ad_spend': np.random.uniform(1000, 10000, n_samples),
        'search_ad_spend': np.random.uniform(1000, 10000, n_samples),
        'discount_percent': np.random.uniform(0, 30, n_samples),
        'is_holiday': np.random.choice([0, 1], p=[0.9, 0.1], size=n_samples),
        'industry': np.random.choice(['Retail', 'Tech', 'Food', 'Healthcare'], size=n_samples),
        'location': np.random.choice(['Bengaluru', 'Mumbai', 'Delhi', 'Chennai'], size=n_samples)
    })

    base = 5000
    df['sales'] = (
        base
        + df['social_ad_spend'] * 2.1
        + df['search_ad_spend'] * 3.4
        + df['discount_percent'] * 450
        + df['is_holiday'] * 16000
        + np.random.normal(0, 1500, n_samples)
    )

    # Include disaster cases
    disaster_data = pd.DataFrame({
        'social_ad_spend': [0, 0, 0, 0, 0],
        'search_ad_spend': [0, 0, 0, 0, 0],
        'discount_percent': [200, 150, 100, 250, 180],
        'is_holiday': [0, 0, 0, 0, 0],
        'industry': ['Retail', 'Tech', 'Food', 'Healthcare', 'Retail'],
        'location': ['Bengaluru', 'Mumbai', 'Delhi', 'Chennai', 'Bengaluru'],
        'sales': [-5000, -3000, -2000, -4000, -3500]
    })

    df = pd.concat([df, disaster_data], ignore_index=True)

    # Pipeline
    categorical_features = ['industry', 'location']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])

    X = df.drop(columns=['sales'])
    y = df['sales']

    param_distributions = {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': [2, 5, 10]
    }

    print('Tuning hyperparameters with RandomizedSearchCV...')
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=5,
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=-1
    )
    search.fit(X, y)

    print(f'Best parameters found: {search.best_params_}')

    best_pipeline = search.best_estimator_

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    versioned_filename = f'sales_pipeline_{timestamp}.joblib'
    latest_filename = 'sales_pipeline_latest.joblib'

    joblib.dump(best_pipeline, versioned_filename)
    joblib.dump(best_pipeline, latest_filename)
    print(f"Retraining complete. Saved as {versioned_filename} and {latest_filename}")


def cleanup_old_checkpoints(days_old=7):
    """Deletes model versions older than N days."""
    now = time.time()
    cutoff = now - (days_old * 86400)

    files = glob.glob('sales_pipeline_20*.joblib')
    for f in files:
        if os.path.getmtime(f) < cutoff:
            os.remove(f)
            print(f"Cleaned up old checkpoint (> {days_old} days): {f}")


if __name__ == '__main__':
    train_and_save_model()
    cleanup_old_checkpoints()


