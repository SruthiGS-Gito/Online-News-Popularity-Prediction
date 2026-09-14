import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "OnlineNewsPopularity.csv"
MODEL_PATH = BASE_DIR / "online_news_popularity_model.pkl"

RANDOM_STATE = 42
TARGET = "shares"
DROP_COLS = ["url", "timedelta"]


def train():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Put the original OnlineNewsPopularity.csv "
            "from the UCI Online News Popularity dataset beside this script."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Same columns removed in the supplied Colab notebook.
    df.drop(columns=DROP_COLS, inplace=True)

    X_temp = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Same mutual-information step used by the notebook.
    mi_scores = mutual_info_regression(X_temp, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=X_temp.columns).sort_values(
        ascending=False
    )
    key_cols = mi_series.head(10).index.tolist()

    # Same IQR capping used by the notebook: only the top-MI features.
    for col in [c for c in key_cols if c != TARGET]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=lower, upper=upper)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # The supplied notebook's final/best reported model was RandomizedSearchCV
    # for linear SVR, with these parameters:
    # C = 2.1584494295802448
    # epsilon = 0.9799098521619943
    # kernel = "linear"
    #
    # Recreate the final estimator directly rather than rerunning the expensive
    # search during deployment.
    model = SVR(
        kernel="linear",
        C=2.1584494295802448,
        epsilon=0.9799098521619943,
    )
    model.fit(X_train_scaled, y_train)

    # Save everything needed by Streamlit in one pickle.
    feature_names = X.columns.tolist()
    metadata = {
        "feature_names": feature_names,
        "target": TARGET,
        "drop_columns": DROP_COLS,
        "key_mi_features": key_cols,
        "mi_scores": mi_series.to_dict(),
        "feature_medians": X_train.median().to_dict(),
        "feature_mins": X_train.min().to_dict(),
        "feature_maxs": X_train.max().to_dict(),
        "feature_means": X_train.mean().to_dict(),
        "feature_stds": X_train.std().replace(0, 1).to_dict(),
    }

    bundle = {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    pred = model.predict(X_test_scaled)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }

    print("Model saved to:", MODEL_PATH)
    print("Features:", len(feature_names))
    print("Test MAE:", metrics["MAE"])
    print("Test RMSE:", metrics["RMSE"])
    print("Test R2:", metrics["R2"])
    return metrics


if __name__ == "__main__":
    train()
