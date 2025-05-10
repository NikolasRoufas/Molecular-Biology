import pandas as pd
import numpy as np
import os
import json
import logging
import yaml
import tempfile
import datetime
from pathlib import Path
import pickle

def setup_logging():
    """
    Set up logging configuration.
    
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger('molecular_analysis')
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(f'logs/app_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Set levels
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger

def load_data(file):
    """
    Load data from a file and infer metadata.
    
    Parameters:
    -----------
    file : streamlit.UploadedFile
        Uploaded file object
    
    Returns:
    --------
    tuple (pandas.DataFrame, dict)
        Loaded data and metadata dictionary
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from file: {file.name}")
    
    # Check file type
    file_extension = file.name.split('.')[-1].lower()
    
    # Infer delimiter based on file extension
    delimiter = ',' if file_extension == 'csv' else '\t'
    
    try:
        # Read data
        df = pd.read_csv(file, delimiter=delimiter, index_col=0)
        
        # Basic metadata
        metadata = {
            'file_name': file.name,
            'file_size': file.size,
            'n_samples': df.shape[0],
            'n_features': df.shape[1],
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'missing_values': df.isna().sum().to_dict()
        }
        
        # Try to identify target column if exists
        possible_target_cols = [col for col in df.columns if col.lower() in 
                             ['target', 'class', 'label', 'group', 'condition']]
        
        if possible_target_cols:
            metadata['target_column'] = possible_target_cols[0]
        else:
            # Check if there's a column with fewer than 10 unique values (potential target)
            categorical_cols = [col for col in df.columns if df[col].nunique() < 10]
            if categorical_cols:
                metadata['potential_target_columns'] = categorical_cols
        
        logger.info(f"Data loaded successfully: {df.shape[0]} samples, {df.shape[1]} features")
        return df, metadata
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def save_results(model, metrics, data, metadata, output_dir="results"):
    """
    Save model, metrics, and processed data to disk.
    
    Parameters:
    -----------
    model : object
        Trained machine learning model
    metrics : dict
        Model evaluation metrics
    data : pandas.DataFrame
        Processed data
    metadata : dict
        Data metadata
    output_dir : str, default="results"
        Directory to save results
        
    Returns:
    --------
    str
        Path to saved results
    """
    logger = logging.getLogger(__name__)
    
    # Create timestamp for unique folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"analysis_{timestamp}")
    
    # Create directory if it doesn't exist
    os.makedirs(result_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(result_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(result_dir, "metrics.json")
    # Convert numpy arrays to lists for JSON serialization
    metrics_json = {}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            metrics_json[k] = v.tolist()
        elif isinstance(v, dict):
            metrics_json[k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv 
                              for kk, vv in v.items()}
        else:
            metrics_json[k] = v
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save processed data
    data_path = os.path.join(result_dir, "processed_data.csv")
    data.to_csv(data_path)
    logger.info(f"Processed data saved to {data_path}")
    
    # Save metadata
    metadata_path = os.path.join(result_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")
    
    return result_dir

def load_config(config_path="configs/app_config.yaml"):
    """
    Load application configuration from YAML file.
    
    Parameters:
    -----------
    config_path : str, default="configs/app_config.yaml"
        Path to configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        # Return default configuration
        return {
            "preprocessing_methods": ["z_score", "minmax", "robust", "log"],
            "feature_selection_methods": ["variance", "anova", "mutual_info", "pca"],
            "ml_algorithms": ["SVM", "Random Forest", "KNN", "Neural Network", "Logistic Regression"]
        }

def export_results_to_report(result_dir, output_format="latex"):
    """
    Export analysis results to a formatted report.
    
    Parameters:
    -----------
    result_dir : str
        Directory containing analysis results
    output_format : str, default="latex"
        Output format ("latex", "pdf", "html")
        
    Returns:
    --------
    str
        Path to generated report
    """
    logger = logging.getLogger(__name__)
    
    # Load results
    with open(os.path.join(result_dir, "metrics.json"), 'r') as f:
        metrics = json.load(f)
    
    with open(os.path.join(result_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Generate report content based on format
    if output_format == "latex":
        # Create LaTeX report
        report_content = generate_latex_report(metrics, metadata, result_dir)
        report_path = os.path.join(result_dir, "report.tex")
        
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"LaTeX report generated at {report_path}")
        return report_path
    
    elif output_format == "html":
        # Create HTML report (simplified example)
        report_content = f"""
        <html>
            <head>
                <title>Analysis Report</title>
            </head>
            <body>
                <h1>Molecular Biology Data Analysis Report</h1>
                <h2>Analysis Results</h2>
                <p>Accuracy: {metrics.get('accuracy', 'N/A')}</p>
                <p>F1 Score: {metrics.get('f1', 'N/A')}</p>
                <!-- Add more content here -->
            </body>
        </html>
        """
        
        report_path = os.path.join(result_dir, "report.html")
        with open(report_path, 'w') as f:
            f.write(report_content)
            
        logger.info(f"HTML report generated at {report_path}")
        return report_path
    
    else:
        logger.error(f"Unsupported output format: {output_format}")
        return None

def generate_latex_report(metrics, metadata, result_dir):
    """
    Generate a LaTeX report from analysis results.
    
    Parameters:
    -----------
    metrics : dict
        Model evaluation metrics
    metadata : dict
        Data metadata
    result_dir : str
        Directory containing analysis results
        
    Returns:
    --------
    str
        LaTeX report content
    """
    # Basic LaTeX report template
    report = r"""
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{listings}
\usepackage{xcolor}

\title{Molecular Biology Data Analysis Report}
\author{Molecular Biology Analysis Platform}
\date{\today}

\begin{document}

\maketitle

\begin{abstract}
This report presents the results of a machine learning analysis on molecular biology data. 
The analysis includes data preprocessing, feature selection, model training, and evaluation.
\end{abstract}

\section{Introduction}
This analysis was performed using the Molecular Biology Data Analysis Platform, a Streamlit-based
application designed for the analysis of molecular biology datasets, particularly gene expression data.

\section{Data Overview}
\begin{itemize}
    \item \textbf{File name:} %s
    \item \textbf{Sample size:} %d
    \item \textbf{Features:} %d
\end{itemize}

\section{Methodology}
\subsection{Preprocessing}
The data was preprocessed using standard methods to prepare it for machine learning analysis.

\subsection{Feature Selection}
Feature selection was performed to identify the most informative features in the dataset.

\subsection{Machine Learning Model}
A machine learning model was trained on the preprocessed data to predict the target variable.

\section{Results}
\subsection{Model Performance}
\begin{table}[h]
\centering
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
Accuracy & %.4f \\
Precision & %.4f \\
Recall & %.4f \\
F1 Score & %.4f \\
\bottomrule
\end{tabular}
\caption{Model performance metrics}
\end{table}

\section{Discussion}
The analysis results demonstrate the effectiveness of machine learning approaches in molecular biology data analysis.
The model achieved good performance metrics, indicating its ability to capture patterns in the data.

\section{Conclusion}
In conclusion, this analysis provides valuable insights into the molecular biology dataset and demonstrates
the potential of machine learning methods in this domain.

\appendix
\section{Technical Details}
This analysis was performed using Python with various libraries including scikit-learn, pandas, and numpy.
The full implementation is available in the Molecular Biology Data Analysis Platform.

\end{document}
""" % (
        metadata.get('file_name', 'N/A'),
        metadata.get('n_samples', 0),
        metadata.get('n_features', 0),
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1', 0)
    )
    
    return report