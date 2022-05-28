from argparse import ArgumentParser
import os
import pickle

import mlflow
import numpy as np

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("random-forest-hyperopt")


def load_pickle(filename: str):
    with open(filename, "rb") as f:
        return pickle.load(f)


def run(data_path: str, n_trials: int):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    def objective(params):
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_param("rmse", rmse)
        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
        "min_samples_split": scope.int(
            hp.quniform("min_samples_split", 2, 10, 1)
        ),
        "min_samples_leaf": scope.int(
            hp.quniform("min_samples_leaf", 1, 4, 1)
        ),
        "random_state": 42,
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed NYC taxi trip data was saved."
    )
    parser.add_argument(
        "--max_evals",
        default=50,
        help="the number of parameter evals for the optimizer to explore."
    )
    args = parser.parse_args()

    run(args.data_path, args.max_evals)
