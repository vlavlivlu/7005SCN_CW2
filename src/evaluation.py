from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def model_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)

    results = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-score": f1_score(y_test, y_pred, average="weighted"),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Predictions": y_pred
    }
    
    return results
