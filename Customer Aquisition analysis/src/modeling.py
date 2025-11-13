"""Model training and inference utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import joblib


def train_model(data: pd.DataFrame, target: str, id_col: str = 'customer_id', save_path: str = None) -> dict:
    data = data.copy()
    y = data[target]
    X = data.drop(columns=[target, id_col], errors='ignore')
    
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical = X.select_dtypes(include=['number']).columns.tolist()
    
    preproc = ColumnTransformer([
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical)
    ])
    
    model = Pipeline([
        ('pre', preproc),
        ('clf', LGBMClassifier(n_estimators=200, random_state=42))
    ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, preds)
    
    if save_path:
        joblib.dump(model, save_path)
    
    return {'model': model, 'auc': auc}


def predict(model, X: pd.DataFrame):
    return model.predict_proba(X)[:,1]
