import shap
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np

def generate_shap(model, X_train, X_test):

    if isinstance(model, LogisticRegression):
        X_test_sample = X_test[:100]
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer(X_test_sample)
        
    else:

        X_test_sample = X_test[:20].toarray()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample, check_additivity=False)

    print(f"{type(model).__name__} SHAP Generated")
    print("*" * 50)
    print(X_train.shape)
    print(type(model).__name__)
    print(type(shap_values))

    if isinstance(shap_values, np.ndarray):
        print(shap_values.shape)
        print(shap_values.min())
        print(shap_values.max())
        print(shap_values.mean())

    elif hasattr(shap_values, "values"):
        print(shap_values.values.shape)
        print(shap_values.values.min())
        print(shap_values.values.max())
        print(shap_values.values.mean())
    print("*" * 50)
    
    results = {
        #"Explainer": explainer, 
        "SHAP Values": shap_values
    }

    return results

