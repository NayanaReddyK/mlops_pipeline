import os
import json
import yaml
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PROCESSED   = os.path.join("data", "processed")
MODELS_DIR  = "models"
METRICS_DIR = "metrics"
PARAMS_FILE = "params.yaml"
MLFLOW_EXPERIMENT = "Auto-MPG-Prediction"

def load_params():
    with open(PARAMS_FILE) as f:
        return yaml.safe_load(f)

def build_model(params):
    mtype = params["model"]["type"]
    if mtype == "linear_regression":
        return LinearRegression()
    elif mtype == "decision_tree":
        return DecisionTreeRegressor(max_depth=params["decision_tree"]["max_depth"])
    else:
        raise ValueError(f"Unknown model type: {mtype}")

def main():
    params = load_params()

    X_train = np.load(os.path.join(PROCESSED, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED, "y_train.npy"))
    X_test  = np.load(os.path.join(PROCESSED, "X_test.npy"))
    y_test  = np.load(os.path.join(PROCESSED, "y_test.npy"))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    model_type = params["model"]["type"]
    run_name   = f"{model_type}_run"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_type",    model_type)
        mlflow.log_param("test_size",     params["data"]["test_size"])
        mlflow.log_param("random_state",  params["data"]["random_state"])
        mlflow.log_param("train_samples", len(X_train))

        model = build_model(params)
        print(f"Training: {model_type} ...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2  = round(float(r2_score(y_test, y_pred)),  4)
        mse = round(float(mean_squared_error(y_test, y_pred)), 4)
        mae = round(float(mean_absolute_error(y_test, y_pred)), 4)

        mlflow.log_metric("r2",  r2)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(model, "model")

        print(f"  R2={r2}  MSE={mse}  MAE={mae}")

        os.makedirs(MODELS_DIR,  exist_ok=True)
        os.makedirs(METRICS_DIR, exist_ok=True)

        joblib.dump(model, os.path.join(MODELS_DIR, "model.pkl"))

        metrics = {"test": {"r2": r2, "mse": mse, "mae": mae}}
        with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"MLflow run '{run_name}' complete. R2={r2}")

if __name__ == "__main__":
    main()