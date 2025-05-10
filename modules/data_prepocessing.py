import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

def preprocess_data(data, method='z_score'):
    """
    Preprocess the input data using the specified method.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data to preprocess
    method : str
        Preprocessing method to apply (z_score, minmax, robust, log)
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    logger.info(f"Preprocessing data with method: {method}")
    
    # Make a copy to avoid modifying the original data
    processed_data = data.copy()
    
    # Get non-numeric columns to exclude from scaling
    non_numeric_cols = processed_data.select_dtypes(exclude=['number']).columns.tolist()
    
    # Separate target and non-numeric columns
    target_cols = non_numeric_cols.copy()
    if 'target' in processed_data.columns:
        target_cols = ['target']
    
    # Get features for scaling (exclude target and non-numeric)
    feature_cols = [col for col in processed_data.columns if col not in target_cols]
    
    # Apply preprocessing method
    if method == 'z_score':
        scaler = StandardScaler()
        processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])
    
    elif method == 'minmax':
        scaler = MinMaxScaler()
        processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])
    
    elif method == 'robust':
        scaler = RobustScaler()
        processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])
    
    elif method == 'log':
        # Add a small constant to avoid log(0)
        processed_data[feature_cols] = np.log1p(processed_data[feature_cols])
    
    logger.info(f"Preprocessing completed. Data shape: {processed_data.shape}")
    return processed_data


def feature_selection(data, method='variance', n_features=50):
    """
    Select features from the data using the specified method.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for feature selection
    method : str
        Feature selection method (variance, anova, mutual_info, pca)
    n_features : int
        Number of features to select
        
    Returns:
    --------
    pandas.DataFrame
        Data with selected features
    """
    logger.info(f"Running feature selection: {method}, n_features={n_features}")
    
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Get non-numeric columns to exclude from feature selection
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Separate target column
    target_col = None
    if 'target' in df.columns:
        target_col = 'target'
    
    # Get feature columns for selection
    feature_cols = [col for col in df.columns if col not in non_numeric_cols and col != target_col]
    
    # Apply feature selection method
    if method == 'variance':
        # Remove features with low variance
        selector = VarianceThreshold(threshold=0.1)
        selected_features = selector.fit_transform(df[feature_cols])
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_cols[i] for i in selected_indices]
        
    elif method == 'anova' and target_col is not None:
        # ANOVA F-value between feature and target
        selector = SelectKBest(f_classif, k=min(n_features, len(feature_cols)))
        selected_features = selector.fit_transform(df[feature_cols], df[target_col])
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_cols[i] for i in selected_indices]
        
    elif method == 'mutual_info' and target_col is not None:
        # Mutual information between feature and target
        selector = SelectKBest(mutual_info_classif, k=min(n_features, len(feature_cols)))
        selected_features = selector.fit_transform(df[feature_cols], df[target_col])
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_cols[i] for i in selected_indices]
        
    elif method == 'pca':
        # Principal Component Analysis
        pca = PCA(n_components=min(n_features, len(feature_cols)))
        selected_features = pca.fit_transform(df[feature_cols])
        # Create new feature names for PCA components
        selected_names = [f'PC{i+1}' for i in range(selected_features.shape[1])]
        
    else:
        logger.warning(f"Unknown feature selection method: {method}. Using all features.")
        return df
    
    # Create new dataframe with selected features
    result_df = pd.DataFrame(selected_features, columns=selected_names, index=df.index)
    
    # Add back non-numeric columns
    for col in non_numeric_cols:
        result_df[col] = df[col].values
    
    # Add back target column if it exists
    if target_col:
        result_df[target_col] = df[target_col].values
    
    logger.info(f"Feature selection completed. Selected {len(selected_names)} features.")
    return result_df