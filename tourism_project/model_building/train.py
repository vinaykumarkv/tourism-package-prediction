# train.py
# Model building with experiment tracking: tune a Random Forest, log everything to MLflow,
# evaluate it and save the best model so the pipeline can commit it to the repository.
import os
import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "tourism_project/deployment/tourism_package_model_v1.joblib"
CLASSIFICATION_THRESHOLD = 0.45   # a little below 0.5 so fewer likely buyers are missed

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-prediction")

# Load the train/test splits produced by the data-preparation job (workflow artifact)
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "MonthlyIncome",
]
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus"]

# Preprocessing: impute + scale numeric columns, impute + one-hot encode categorical columns
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

# Model + hyperparameter grid (kept compact so the pipeline finishes quickly on GitHub Actions)
model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
])
param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [10, 20],
    "model__max_features": ["sqrt", 0.5],
    "model__class_weight": ["balanced"],   # purchasers are only ~19% of customers
}

def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }, pred

with mlflow.start_run(run_name="random_forest_tuning"):
    # Hyperparameter tuning (5-fold CV, optimising F1 of the 'purchased' class)
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every tuned parameter combination as its own nested MLflow run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            mlflow.log_params({k.replace("model__", ""): v for k, v in results["params"][i].items()})
            mlflow.log_metric("cv_mean_f1", results["mean_test_score"][i])
            mlflow.log_metric("cv_std_f1", results["std_test_score"][i])

    # Log the best parameters on the parent run
    best_model = grid_search.best_estimator_
    mlflow.log_params({f"best_{k.replace('model__', '')}": v for k, v in grid_search.best_params_.items()})
    mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("Best params:", grid_search.best_params_)
    print(f"Best CV F1 : {grid_search.best_score_:.4f}")

    # Evaluate the best model on train and test data
    train_metrics, _ = evaluate(best_model, Xtrain, ytrain)
    test_metrics, y_pred_test = evaluate(best_model, Xtest, ytest)
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    print("\nTrain metrics:", {k: round(v, 4) for k, v in train_metrics.items()})
    print("Test metrics :", {k: round(v, 4) for k, v in test_metrics.items()})
    print("\nTest classification report:")
    print(classification_report(ytest, y_pred_test, target_names=["Not purchased", "Purchased"]))

    # Save the best model (with its decision threshold) where the Streamlit app will load it
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": best_model,
        "threshold": CLASSIFICATION_THRESHOLD,
        "features": list(Xtrain.columns),
    }, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH, artifact_path="model")
    print(f"Model saved to {MODEL_PATH}")
