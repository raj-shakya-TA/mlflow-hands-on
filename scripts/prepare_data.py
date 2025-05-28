import mlflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    mlflow.log_param("step", "data_preparation")

    # Load data
    data = pd.read_csv("data/housing.csv")
    mlflow.log_param("original_shape", data.shape)

    # Basic cleaning
    data.dropna(inplace=True)

    # Save split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    mlflow.log_metric("rows_train", len(train))
    mlflow.log_metric("rows_test", len(test))

if __name__ == "__main__":
    main()
