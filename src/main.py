import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocessing import preprocessing_text
from feature_extraction import tfidf_vectorize 
from train_model import train_logistic_regression, train_random_forest, train_xgboost
from evaluation import model_evaluate
from xai_shap import generate_shap
import joblib
import tempfile
import os
import numpy as np

def process_dataset(file, text_column, label_column):
    
    # Read dataset
    file_extension = Path(file).suffix
    if file_extension == ".csv":
        df = pd.read_csv(file)
    elif file_extension == ".xlsx":
        df = pd.read_excel(file)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")
    print(f"\nDataset Read: {file}")

    # Preprocessing
    df = preprocessing_text(df, text_column)

    # Define feature and label
    X = df[text_column]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_column])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorization
    X_train, X_test, feature_names = tfidf_vectorize(X_train, X_test)
    print(X_train.shape)

    return X_train, X_test, y_train, y_test, feature_names, label_encoder

def run_pipeline():

    # **********STARTS HERE**********
    # Datasets
    datasets = {
        "Medical": {
            "file": "data/medical_tc_train.csv",
            "text_column": "medical_abstract",
            "label_column": "condition_label"
        },
        "Fake News": {
            "file": "data/fake_new_dataset.xlsx",
            "text_column": "text",
            "label_column": "label"
        }
    }

    # Data processing
    processed_datasets = {}
    for name, info in datasets.items():
        X_train, X_test, y_train, y_test, feature_names, label_encoder = process_dataset(info["file"], 
                                                                                    info["text_column"], 
                                                                                    info["label_column"])
        
        processed_datasets[name] = {
            "X_train": X_train, 
            "X_test": X_test, 
            "y_train": y_train,
            "y_test": y_test, 
            "feature_names": feature_names, 
            "label_encoder": label_encoder
        }

    # Model training
    trained_models = {}

    for name, data in processed_datasets.items():
        print(f"\n=========={name}==========")
        trained_models[name] = {
            "Logistic Regression": train_logistic_regression(name, data["X_train"], data["y_train"]),
            "Random Forest": train_random_forest(name, data["X_train"], data["y_train"]),
            "XGBoost": train_xgboost(name, data["X_train"], data["y_train"])
        }


    # Model evaluation 
    evaluation_results = {}
    for name, models in trained_models.items():
        evaluation_results[name] = {}

        # Access Trained Model Dictionary and evaluate based on dataset and model 
        for model_name, model in models.items():
            evaluation_results[name][model_name] = model_evaluate(model, 
                                                                processed_datasets[name]["X_test"], 
                                                                processed_datasets[name]["y_test"])

    # SHAP Analysis
    shap_results = {}
    for name, models in trained_models.items():
        print(f"\n=========={name}==========")

        shap_results[name] = {}

        # Access Trained Model Dictionary and generate SHAP based on dataset and model 
        for model_name, model in models.items():
            shap_results[name][model_name] = generate_shap(model, 
                                                        processed_datasets[name]["X_train"], 
                                                        processed_datasets[name]["X_test"])

    results = {
         
        "feature_names": {
            name: processed_datasets[name]["feature_names"]
            for name in processed_datasets
        },
        "evaluation_results": evaluation_results, 
        "shap_results": shap_results
        #"processed_datasets": processed_datasets,
        #"trained_models": trained_models, 
    }

    # Save the complete results
    result_file = "results/results.pkl"
    joblib.dump(results, result_file)
    print(f"All Results have successfully saved in {result_file}!!!")

    return results

if __name__ == "__main__":
    run_pipeline()