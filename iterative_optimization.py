import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from hyperparameter_optimization_and_training import hyperparameter_optimization_and_training
from feature_importance_analysis import feature_importance_analysis
from feature_correlation_analysis import feature_correlation_analysis
from feature_selection import feature_selection
from evaluate_and_plot import evaluate_and_plot
from leave_one_out_validation import leave_one_out_validation
from sklearn.preprocessing import StandardScaler



# scalar = StandardScaler()
def iterative_optimization(models, X, y,n_trials=100,mae_threshold=20):
    """
    Perform iterative optimization by removing features based on importance and correlation.

    Parameters:
    models (dict): A dictionary of model classes.
    X (DataFrame): The feature data.
    y (Series): The target data.

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
        best_model = None
        while True:
            best_model, scaler, X_train, X_test, y_train, y_test, mae_mean, best_params = hyperparameter_optimization_and_training(model_class, X_model, y,n_trials=n_trials)
            y_pred_train = best_model.predict(scaler.transform(X_train))
            y_pred_test = best_model.predict(scaler.transform(X_test))
            evaluate_and_plot(model_name, y_train, y_pred_train, y_test, y_pred_test, X_train, X_test,mae_threshold=mae_threshold,mae_mean=mae_mean)
            
            # Calculate additional metrics
            r2_test = r2_score(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            
            result = {
                'mae_mean': mae_mean,
                'r2_test_avg': r2_test,
                'mae_test_avg': mae_test,
                'best_params_avg': best_params
            }
            feature_importance_percent = feature_importance_analysis(model_name, best_model, scaler, X_train,mae_mean,mae_threshold)
            corr_matrix = feature_correlation_analysis(X_model,mae_mean,mae_threshold)
            remove_feature = feature_selection(X_model, feature_importance_percent, corr_matrix, removed_features)
            if not remove_feature:
                final_features = X_model.columns.tolist()
                break
            # 输出删除的特征及其相关性和重要性
            for feature in remove_feature:
                print(f"Removing feature: {feature}")
                feature_index = X_model.columns.get_loc(feature)
                print(f"Feature importance: {feature_importance_percent[feature_index]:.2f}%")
                print(f"Correlation with other features:")
                for j, other_feature in enumerate(X_model.columns):
                    if other_feature != feature and corr_matrix.iloc[feature_index, j] > 0.8:
                        print(f"  {feature} - {other_feature}: {corr_matrix.iloc[feature_index, j]:.2f}")
            for feature in remove_feature:
                removed_features.append(feature)
                X_model = X_model.drop(columns=[feature])
                X_train = X_train.drop(columns=[feature])
                X_test = X_test.drop(columns=[feature])
        r2_loo = leave_one_out_validation(best_model, scaler, X_model, y)
        result['r2_loo_avg'] = r2_loo
        result['final_features'] = final_features
        results[model_name] = result

    for model_name, result in results.items():
        print(f'Model: {model_name}')
        print(f'Average MAE: {result["mae_mean"]}')
        print(f'R^2 (Test): {result["r2_test_avg"]}')
        print(f'MAE (Test): {result["mae_test_avg"]}')
        print(f'Best Params: {result["best_params_avg"]}')
        print(f'R^2 (LOO): {result["r2_loo_avg"]}')
        print(f'Final Features: {result["final_features"]}')
        print('-' * 40)

    return results, char_change
