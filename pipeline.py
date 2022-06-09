from argparse import ArgumentParser

import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error


DIR_DATA = "/home/ibra/python-projects/mlops-zoomcamp/data/"
DIR_MODELS = "/home/ibra/python-projects/mlops-zoomcamp/models/"

parser = ArgumentParser(description="Some description.")
parser.add_argument("--train_data")
parser.add_argument("--val_data")
parser.add_argument("--model")
# parser.add_argument("--alpha")
args = parser.parse_args()


def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def vectorize():
    pass  # TODO


def train_evaluate(model: str, X_train: list, y_train: list,
                   X_val: list, y_val: list, alpha=1.0) -> tuple:
    if model == "lr":
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_val)
        return lr, mean_squared_error(y_pred, y_val, squared=False)
    elif model == "lasso":
        lasso = Lasso(alpha=alpha)
        lasso.fit(X_train, y_train)
        y_pred = lasso.predict(X_val)
        return lasso, mean_squared_error(y_pred, y_val, squared=False)


def save_model(dv, model, path: str) -> None:
    with open(DIR_MODELS + path, "wb") as f:
        pickle.dump((dv, model), f)


df_train = read_dataframe(
    args.train_data
    if args.train_data else DIR_DATA + "green_tripdata_2021-01.parquet"
)
df_val = read_dataframe(
    args.val_data
    if args.val_data else DIR_DATA + "green_tripdata_2021-02.parquet"
)

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

if args.model:
    model, rmse = train_evaluate(args.model, X_train, y_train, X_val, y_val)
else:
    model, rmse = train_evaluate("lr", X_train, y_train, X_val, y_val)

print(rmse)

save_model(dv, model, "lin_reg.bin")
