import pandas as pd
import numpy as np


def feature_selection(X_model, feature_importance_percent, corr_matrix, removed_features):
    """
    Select features based on importance and correlation.

    Parameters:
    X_model (DataFrame): The feature data.
    feature_importance_percent (array): The feature importance in percentage.
    corr_matrix (DataFrame): The correlation matrix.
    removed_features (list): The list of removed features.

    Returns:
    remove_feature (list): The list of features to be removed.
    """
    remove_feature = []
    max_remove_count = 3  # Maximum number of features to remove at once
    remove_count = 0  # Counter for the number of features removed

    for i in range(len(feature_importance_percent)):
        if i < len(feature_importance_percent):  # Ensure the index is within the valid range
            if X_model.columns[i] not in removed_features:
                for j in range(i+1, len(feature_importance_percent)):
                    if j < len(feature_importance_percent):  # Ensure the index is within the valid range
                        if X_model.columns[j] not in removed_features:
                            if corr_matrix.iloc[i, j] > 0.8:
                                # Find the two features with the smallest importance
                                if feature_importance_percent[i] < feature_importance_percent[j]:
                                    if remove_count < max_remove_count:
                                        remove_feature.append(X_model.columns[i])
                                        remove_count += 1
                                else:
                                    if remove_count < max_remove_count:
                                        remove_feature.append(X_model.columns[j])
                                        remove_count += 1
                                if remove_count >= max_remove_count:
                                    break
                if remove_count >= max_remove_count:
                    break

    # Check if only two features are left and their correlation is > 0.8
    remaining_features = [col for col in X_model.columns if col not in removed_features and col not in remove_feature]
    if len(remaining_features) == 2:
        feat1, feat2 = remaining_features
        if corr_matrix.loc[feat1, feat2] > 0.8:
            idx1 = X_model.columns.get_loc(feat1)
            idx2 = X_model.columns.get_loc(feat2)
            if feature_importance_percent[idx1] < feature_importance_percent[idx2]:
                remove_feature.append(feat1)
            else:
                remove_feature.append(feat2)

    remove_feature = list(set(remove_feature))
    if not remove_feature:
        remove_feature = [X_model.columns[i] for i in range(len(feature_importance_percent)) if feature_importance_percent[i] < 5 and X_model.columns[i] not in removed_features]
    return remove_feature
