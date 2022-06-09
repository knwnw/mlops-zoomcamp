import mlflow
import pandas as pd
import xgboost as xgb

from hyperopt import STATUS_OK
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


ROOT_DIR = "/home/ibra/python-projects/mlops-zoomcamp"

mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/__experiments/mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

mlflow.xgboost.autolog()


def read_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    return df


def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(val, "validation")],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {"loss": rmse, "status": STATUS_OK}


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

train = xgb.DMatrix(X_train, label=y_train)
val = xgb.DMatrix(X_val, label=y_val)

params = {
    "learning_rate": 0.2047217,
    "max_depth": 17,
    "min_child_weight": 1.2402612,
    "objective": "reg:squarederror",
    "reg_alpha": 0.2856789,
    "reg_lambda": 0.0042644,
    "seed": 42,
}

booster = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=1000,
    evals=[(val, "validation")],
    early_stopping_rounds=50
)


# NOTE
# The code below was used to obtain the params for (almost) best RMSE.
# Be advised that it takes approximately 2 hours for the code to complete.

# from hyperopt import fmin, tpe, hp, Trials
# from hyperopt.pyll import scope

# search_space = {
#     "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
#     "learning_rate": hp.loguniform("learning_rate", -3, 0),
#     "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
#     "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
#     "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
#     "objective": "reg:squarederror",
#     "seed": 42,
# }

# best_result = fmin(
#     fn=objective,
#     space=search_space,
#     algo=tpe.suggest,
#     max_evals=50,
#     trials=Trials()
# )
