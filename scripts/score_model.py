import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def main():
    mlflow.log_param("step", "model_scoring")

    # Load model
    model = joblib.load("model.pkl")

    # Load test data
    test = pd.read_csv("data/test.csv")
    X_test = test.drop("median_house_value", axis=1, errors='ignore').select_dtypes(include='number')
    y_test = test["median_house_value"]

    # Predict and calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mlflow.log_metric("test_rmse", rmse)

if __name__ == "__main__":
    main()
