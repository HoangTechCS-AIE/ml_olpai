import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


MODEL_PARAMS = {
    "n_estimators": 2000,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "predictor": "cpu_predictor",
    "n_jobs": 4,
    "random_state": 2406,
}


TEXT_COLUMNS = ["dish_name", "dish_desc", "restaurant_name"]
CATEGORICAL_COLUMNS = ["restaurant_district", "restaurant_city", "restaurant_type"]
HIGH_CARDINALITY_COLUMNS = ["restaurant_id"]


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in TEXT_COLUMNS:
        text_series = df[column].fillna("")
        df[f"{column}_char_len"] = text_series.str.len()
        df[f"{column}_word_count"] = text_series.str.split().str.len()
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement_total"] = df[["num_likes", "num_dislikes"]].fillna(0).sum(axis=1)
    df["like_ratio"] = df["num_likes"].fillna(0) / (df["engagement_total"] + 1)
    df["net_likes"] = df["num_likes"].fillna(0) - df["num_dislikes"].fillna(0)
    df["price_diff_avg"] = df["price"].fillna(0) - df["avg_price"].fillna(0)
    df["price_to_avg_ratio"] = df["price"].fillna(0) / (df["avg_price"].fillna(0) + 1)
    df["is_discounted"] = (df["price_diff_avg"] < 0).astype(int)
    df["rating_weighted"] = df["restaurant_rating"].fillna(0) * df["num_ratings"].fillna(0)
    df["lat_long_sum"] = df["Latitude"].fillna(0) + df["Longitude"].fillna(0)
    df["lat_long_diff"] = df["Latitude"].fillna(0) - df["Longitude"].fillna(0)
    return df


def frequency_encode(train_df: pd.DataFrame, test_df: pd.DataFrame, column: str) -> None:
    train_df[column] = train_df[column].fillna("missing").astype(str)
    test_df[column] = test_df[column].fillna("missing").astype(str)
    frequency = train_df[column].value_counts(dropna=False) / len(train_df)
    train_df[f"{column}_freq"] = train_df[column].map(frequency)
    test_df[f"{column}_freq"] = test_df[column].map(frequency).fillna(0)
    train_df.drop(columns=column, inplace=True)
    test_df.drop(columns=column, inplace=True)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_df = train_df.copy()
    test_df = test_df.copy()

    target = train_df.pop("num_purchases")

    train_df = add_text_features(train_df)
    test_df = add_text_features(test_df)

    train_df = add_ratio_features(train_df)
    test_df = add_ratio_features(test_df)

    for column in HIGH_CARDINALITY_COLUMNS:
        frequency_encode(train_df, test_df, column)

    train_df.drop(columns=TEXT_COLUMNS, inplace=True)
    test_df.drop(columns=TEXT_COLUMNS, inplace=True)

    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=CATEGORICAL_COLUMNS, dummy_na=True)

    combined = combined.drop(columns=["uuid"])
    combined = combined.fillna(0)

    combined.columns = (
        combined.columns.str.replace("[", "_", regex=False)
        .str.replace("]", "_", regex=False)
        .str.replace("<", "_", regex=False)
    )

    x_train_processed = combined.iloc[: len(train_df), :].reset_index(drop=True)
    x_test_processed = combined.iloc[len(train_df) :, :].reset_index(drop=True)

    return x_train_processed, x_test_processed, target.reset_index(drop=True)


def train_and_evaluate(features, target):
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=2406,
    )

    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=100,
        verbose=False,
    )

    val_predictions = model.predict(x_val)
    rmse = mean_squared_error(y_val, val_predictions, squared=False)
    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is not None:
        best_n_estimators = int(best_iteration) + 1
    else:
        best_n_estimators = MODEL_PARAMS["n_estimators"]

    tuned_params = MODEL_PARAMS.copy()
    tuned_params["n_estimators"] = best_n_estimators

    final_model = xgb.XGBRegressor(**tuned_params)
    final_model.fit(features, target, eval_metric="rmse", verbose=False)

    metrics = {
        "rmse": rmse,
        "best_n_estimators": best_n_estimators,
    }

    return final_model, metrics


def main():
    train_df = pd.read_csv("publics_train.csv")
    test_df = pd.read_csv("publics_test.csv")

    uuid_test = test_df["uuid"].copy()

    x_train, x_test, y_train = preprocess_data(train_df, test_df)
    model, metrics = train_and_evaluate(x_train, y_train)

    predictions = model.predict(x_test)
    submission = pd.DataFrame(
        {
            "uuid": uuid_test,
            "num_purchases_pred": predictions,
        }
    )
    submission.to_csv("submission.csv", index=False)

    print(f"Validation RMSE: {metrics['rmse']:.4f}")
    print(f"Best n_estimators: {metrics['best_n_estimators']}")


if __name__ == "__main__":
    main()
