import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_logistic_regression(dataset, X_train, y_train):
    print(f"[{dataset}] Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    print(f"[{dataset}] Logistic Regression Trained Sucessfully!!!")
    return lr_model

def train_random_forest(dataset, X_train, y_train):    
    print(f"[{dataset}] Training Random Forest...")
    dt_model = RandomForestClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    print(f"[{dataset}] Random Forest Trained Sucessfully!!!")
    return dt_model

def train_xgboost(dataset, X_train, y_train):
    print(f"[{dataset}] Training XGBoost...")
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    print(f"[{dataset}] XGBoost Trained Sucessfully!!!")
    return xgb_model