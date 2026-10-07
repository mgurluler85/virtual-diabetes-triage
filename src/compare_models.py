from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def main():
    diabetes = load_diabetes(as_frame=True)

    X = diabetes.frame.drop(columns=["target"])
    y = diabetes.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    models = {
        "LinearRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        ),
        "Ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("regressor", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
    }

    print("Model comparison")
    print("----------------")

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, predictions)

        print(f"{name}: RMSE = {rmse:.4f}")


if __name__ == "__main__":
    main()
