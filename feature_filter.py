import itertools
from sklearn.metrics import r2_score, mean_absolute_error
from hyperparameter_optimization_and_training import hyperparameter_optimization_and_training
from leave_one_out_validation import leave_one_out_validation
from evaluate_and_plot import evaluate_and_plot


def feature_filter(models, X, y, n_trials=50, max_features=8,mae_threshold = 16):
    """
    Perform iterative optimization by iterating through all feature combinations.

    Parameters:
    models (dict): A dictionary of model classes.
    X (DataFrame): The feature data.
    y (Series): The target data.
    n_trials (int): The number of trials for hyperparameter optimization.

    Returns:
    results (dict): The results of the optimization.
    char_change (dict): The character change information.
    """
    results = {}
    char_change = {}
    for model_name, model_class in models.items():
        X_model = X.copy()
        removed_features = []
        final_features = []
        result = []
        num_features = min(len(X_model.columns), max_features)
        for r in range(1, num_features + 1):
            feature_combinations = itertools.combinations(X_model.columns, r)
            for feature_combination in feature_combinations:
                try:
                    X_train = X_model[list(feature_combination)]
                    X_test = X_model[list(feature_combination)]
                    best_model, scaler, X_train, X_test, y_train, y_test, mae_mean, best_params = hyperparameter_optimization_and_training(model_class, X_train, y, n_trials=n_trials)
                    y_pred_train = best_model.predict(scaler.transform(X_train))
                    y_pred_test = best_model.predict(scaler.transform(X_test))
                    evaluate_and_plot(model_name, y_train, y_pred_train, y_test, y_pred_test, X_train, X_test,mae_threshold=mae_threshold,mae_mean=mae_mean)

                    # Calculate additional metrics
                    r2_test = r2_score(y_test, y_pred_test)
                    mae_test = mean_absolute_error(y_test, y_pred_test)

                    result.append({
                        'mae_mean': mae_mean,
                        'r2_test_avg': r2_test,
                        'mae_test_avg': mae_test,
                        'best_params_avg': best_params,
                        'final_features': list(feature_combination)
                    })

                except Exception as e:
                    print(f"Error in iterative_optimization for model {model_name}: {e}")
                    break

            # Sort the results based on MAE (Test) in ascending order and keep only top 5
            result = sorted(result, key=lambda x: x['mae_mean'])[:5]
            results[model_name] = result

    # Print the top 5 models for each model type with their hyperparameters and performance
    for model_name, model_results in results.items():
        print(f'Model: {model_name}')
        for i, result in enumerate(model_results):
            print(f'Rank {i+1}:')
            print(f'Best Params: {result["best_params_avg"]}')
            print(f'Average MAE: {result["mae_mean"]}')
            print(f'R^2 (Test): {result["r2_test_avg"]}')
            print(f'MAE (Test): {result["mae_test_avg"]}')
            print(f'Final Features: {result["final_features"]}')
            print('-' * 40)

    return results, char_change
