import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


ROOT_DIR = "/home/ibra/python-projects/mlops-zoomcamp"

mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/__experiments/mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


df_train = read_dataframe(ROOT_DIR + "/data/green_tripdata_2021-01.parquet")
df_val = read_dataframe(ROOT_DIR + "/data/green_tripdata_2021-02.parquet")

df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

categorical = ["PU_DO"]
numerical = ["trip_distance"]

dv = DictVectorizer()
train_dicts = df_train[categorical + numerical].to_dict(orient="records")
X_train = dv.fit_transform(train_dicts)
val_dicts = df_val[categorical + numerical].to_dict(orient="records")
X_val = dv.transform(val_dicts)

target = "duration"
y_train = df_train[target].values
y_val = df_val[target].values

with mlflow.start_run():
    mlflow.set_tag("developer", "ibra")
    mlflow.log_param("train-data-path", "/data/green_tripdata_2021-01.parquet")
    mlflow.log_param("val-data-path", "/data/green_tripdata_2021-02.parquet")

    alpha = 0.00001
    mlflow.log_param("alpha", alpha)

    lr = Lasso(alpha)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
