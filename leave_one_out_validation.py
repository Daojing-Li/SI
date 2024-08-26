from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


def leave_one_out_validation(best_model, scaler, X_model, y):
    """
    Perform leave-one-out validation.

    Parameters:
    best_model (object): The trained best model.
    scaler (object): The standardization scaler.
    X_model (DataFrame): The feature data.
    y (Series): The target data.

    Returns:
    r2_loo (float): The R^2 score from leave-one-out validation.
    """
    loo = LeaveOneOut()
    y_pred_loo = []
    y_true_loo = []
    for train_index, test_index in loo.split(X_model):
        X_train_loo, X_test_loo = X_model.iloc[train_index], X_model.iloc[test_index]
        y_train_loo, y_test_loo = y.iloc[train_index], y.iloc[test_index]
        best_model.fit(scaler.fit_transform(X_train_loo), y_train_loo)
        y_pred_loo.append(best_model.predict(scaler.transform(X_test_loo))[0])
        y_true_loo.append(y_test_loo.values[0])
    r2_loo = r2_score(y_true_loo, y_pred_loo)
    return r2_loo
