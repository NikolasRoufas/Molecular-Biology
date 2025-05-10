import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def get_model(algorithm, params=None):
    """
    Get a machine learning model instance based on the specified algorithm.
    
    Parameters:
    -----------
    algorithm : str
        Name of the machine learning algorithm
    params : dict, optional
        Parameters to initialize the model
        
    Returns:
    --------
    object
        Machine learning model instance
    """
    if params is None:
        params = {}
    
    if algorithm == "SVM":
        return SVC(probability=True, **params)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(**params)
    elif algorithm == "KNN":
        return KNeighborsClassifier(**params)
    elif algorithm == "Neural Network":
        return MLPClassifier(**params)
    elif algorithm == "Logistic Regression":
        return LogisticRegression(**params)
    else:
        logger.warning(f"Unknown algorithm: {algorithm}. Using Random Forest as default.")
        return RandomForestClassifier()

def train_model(X_train, y_train, algorithm="SVM", params=None):
    """
    Train a machine learning model on the provided data.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature data
    y_train : pandas.Series or numpy.ndarray
        Training target data
    algorithm : str
        Name of the machine learning algorithm to use
    params : dict, optional
        Parameters for the machine learning algorithm
        
    Returns:
    --------
    object
        Trained machine learning model
    """
    logger.info(f"Training {algorithm} model")
    
    model = get_model(algorithm, params)
    model.fit(X_train, y_train)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature data
    y_test : pandas.Series or numpy.ndarray
        Test target data
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    # Calculate probabilities if model supports it
    try:
        y_pred_proba = model.predict_proba(X_test)
    except:
        logger.warning("Model does not support predict_proba")
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
    }
    
    # Add AUC if probabilities are available and it's a binary classification problem
    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
        metrics["auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report
    
    logger.info(f"Evaluation complete: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
    
    return metrics

def hyperparameter_tuning(X_train, y_train, algorithm="SVM", param_grid=None):
    """
    Perform hyperparameter tuning for a machine learning model.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training feature data
    y_train : pandas.Series or numpy.ndarray
        Training target data
    algorithm : str
        Name of the machine learning algorithm
    param_grid : dict, optional
        Parameter grid for hyperparameter tuning
        
    Returns:
    --------
    dict
        Dictionary of best parameters
    """
    logger.info(f"Performing hyperparameter tuning for {algorithm}")
    
    if param_grid is None:
        # Define default parameter grids for each algorithm
        if algorithm == "SVM":
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
        elif algorithm == "Random Forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        elif algorithm == "KNN":
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]  # Manhattan or Euclidean distance
            }
        elif algorithm == "Neural Network":
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        elif algorithm == "Logistic Regression":
            param_grid = {
                'C': [0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
    
    # Get the model
    model = get_model(algorithm)
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return best_params