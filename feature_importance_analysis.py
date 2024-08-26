import shap
import numpy as np


def feature_importance_analysis(model_name, best_model, scaler, X_train,mae_mean,mae_threshold):
    """
    Analyze the feature importance using SHAP values.

    Parameters:
    model_name (str): The model name.
    best_model (object): The trained best model.
    scaler (object): The standardization scaler.
    X_train (DataFrame): The training set feature data.

    Returns:
    feature_importance_percent (array): The feature importance in percentage.
    """
    if model_name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
        explainer = shap.LinearExplainer(best_model, scaler.transform(X_train))
    elif model_name in ['SVR']:
        explainer = shap.KernelExplainer(best_model.predict, scaler.transform(X_train))
    elif model_name in ['DecisionTree', 'RandomForest', 'GradientBoosting', 'XGBoost', 'LightGBM']:
        explainer = shap.TreeExplainer(best_model)
    else:
        raise ValueError(f"Unsupported model type for SHAP analysis: {model_name}")
    shap_values = explainer(scaler.transform(X_train))
    feature_importance = np.mean(np.abs(shap_values.values), axis=0)
    total_importance = np.sum(feature_importance)
    feature_importance_percent = (feature_importance / total_importance) * 100
    # Plot feature importance
    if mae_mean < mae_threshold:
        shap.summary_plot(shap_values, X_train, feature_names=X_train.columns, plot_type="bar")
    return feature_importance_percent
