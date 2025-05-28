import mlflow
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def main():
    mlflow.log_param("step", "model_training")

    train = pd.read_csv("data/train.csv")
    X = train.drop("median_house_value", axis=1, errors='ignore').select_dtypes(include='number')
    y = train["median_house_value"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
    mlflow.log_artifact("model.pkl")

    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mlflow.log_metric("train_rmse", rmse)

if __name__ == "__main__":
    main()
