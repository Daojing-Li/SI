import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import LeaveOneOut

def train_and_evaluate(model_class, X, y, random_state=42,n_trials=100):
    """
    Train and evaluate a regression model using Optuna for hyperparameter tuning.

    Parameters:
    model_class (class): The regression model class to be used (e.g., LinearRegression, Ridge).
    X (DataFrame): The feature matrix.
    y (Series): The target vector.
    scaler (object): The scaler object used for data normalization (default is StandardScaler).
    random_state (int): The random seed for reproducibility (default is 42).

    Returns:
    tuple: A tuple containing the best mean absolute error (MAE) from Optuna, the average MAE from test sets, and the best hyperparameters.
    """
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    def objective(trial):
        """
        Define the objective function for Optuna to optimize.

        Parameters:
        trial (Trial): A trial object for hyperparameter sampling.

        Returns:
        float: The mean absolute error (MAE) for the current trial.
        """
        params = {}
        if model_class == LinearRegression:
            params = {}
        elif model_class == Ridge:
            params = {'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True)}
        elif model_class == Lasso:
            params = {'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True)}
        elif model_class == ElasticNet:
            params = {
                'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0, log=True)
            }
        elif model_class == SVR:
            params = {
                'C': trial.suggest_float('C', 1e-2, 200),
                'epsilon': trial.suggest_float('epsilon', 1e-2, 1e2, log=True),
                'kernel': trial.suggest_categorical('kernel', ['poly', 'rbf'])
            }
        elif model_class == DecisionTreeRegressor:
            params = {
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 5),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
        elif model_class == RandomForestRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 300),
                'max_depth': trial.suggest_int('max_depth', 1, 3),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            }
        elif model_class == GradientBoostingRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 3)
            }
        elif model_class == XGBRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 3)
            }
        elif model_class == LGBMRegressor:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 3)
            }
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        model = model_class(**params)
        mae_list = []
        scaler = StandardScaler()
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(X_train):
            X_train_loo, X_test_loo = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_loo, y_test_loo = y_train.iloc[train_index], y_train.iloc[test_index]

            # Normalize the data using the provided scaler
            X_train_loo_scaled = scaler.fit_transform(X_train_loo)
            X_test_loo_scaled = scaler.transform(X_test_loo)

            # Train the model
            model.fit(X_train_loo_scaled, y_train_loo)

            # Predict and calculate MAE
            y_pred_loo = model.predict(X_test_loo_scaled)
            mae_loo = mean_absolute_error(y_test_loo, y_pred_loo)
            mae_list.append(mae_loo)

        # Calculate the mean MAE
        mae_mean = np.mean(mae_list)
        return mae_mean
    sampler = optuna.samplers.TPESampler(n_startup_trials=5, n_ei_candidates=24,seed=42)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(objective, n_jobs=-1, n_trials=n_trials)

    # Train the model with the best hyperparameters on the entire training set
    best_params = study.best_params
    best_model = model_class(**best_params)
    mae_test_list = []
    scaler = StandardScaler()
    # 获得随机数为42情况下的性能表现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(scaler.fit_transform(X_train), y_train)
    y_pred_train = best_model.predict(scaler.transform(X_train))
    y_pred_test = best_model.predict(scaler.transform(X_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"random_state = 42")
    print(f"Best hyperparameters: {best_params}")
    print(f"Train R^2: {r2_train:.4f}")
    print(f"Test R^2: {r2_test:.4f}")
    print(f"Test RMSE: {rmse_test:.4f}")
    print(f"Test MAE: {mae_test:.4f}")

    for i in range(100):
        random_state = i
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
        best_model.fit(scaler.fit_transform(X_train), y_train)
        y_pred_test = best_model.predict(scaler.transform(X_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_test_list.append(mae_test)
        if mae_test == min(mae_test_list):
            best_random_state = random_state
    mae_test_mean = np.mean(mae_test_list)
    print(f"Average MAE on 100 times random test sets: {mae_test_mean:.4f}")
    # print(f"Best random state: {best_random_state} with best MAE: {min(mae_test_list)}")


    # Print current feature combination
    print("Current feature combination:")
    print(X.columns)
    return study.best_value, mae_test_mean, best_params
