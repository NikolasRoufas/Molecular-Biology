import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import logging

logger = logging.getLogger(__name__)

def plot_pca(data, target_col=None, n_components=2):
    """
    Perform PCA and plot the results.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for PCA
    target_col : str, optional
        Target column name for coloring points
    n_components : int, default=2
        Number of PCA components to compute
        
    Returns:
    --------
    matplotlib.figure.Figure
        PCA plot figure
    """
    logger.info(f"Creating PCA plot with {n_components} components")
    
    # Make a copy of the data
    df = data.copy()
    
    # Separate target if provided
    targets = None
    if target_col and target_col in df.columns:
        targets = df[target_col]
        df = df.drop(columns=[target_col])
    
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # Perform PCA
    pca = PCA(n_components=min(n_components, numeric_df.shape[1]))
    principal_components = pca.fit_transform(numeric_df)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot points
    if targets is not None:
        # Get unique targets
        unique_targets = targets.unique()
        
        # Plot each target class
        for target in unique_targets:
            indices = targets == target
            ax.scatter(
                principal_components[indices, 0],
                principal_components[indices, 1],
                label=str(target),
                alpha=0.8
            )
        ax.legend()
    else:
        ax.scatter(
            principal_components[:, 0],
            principal_components[:, 1],
            alpha=0.8
        )
    
    # Add labels and title
    ax.set_xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_title('PCA Visualization')
    
    # Add grid
    ax.grid(alpha=0.3)
    
    # Annotate total explained variance
    total_var = pca.explained_variance_ratio_.sum()
    ax.annotate(
        f'Total explained variance: {total_var:.2%}',
        xy=(0.5, 0.02),
        xycoords='figure fraction',
        ha='center'
    )
    
    plt.tight_layout()
    return fig

def plot_heatmap(data, target_col=None, max_features=50):
    """
    Plot correlation heatmap for the input data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for the heatmap
    target_col : str, optional
        Target column name to include in correlations
    max_features : int, default=50
        Maximum number of features to include in the heatmap
        
    Returns:
    --------
    matplotlib.figure.Figure
        Heatmap figure
    """
    logger.info("Creating correlation heatmap")
    
    # Make a copy of the data
    df = data.copy()
    
    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    # If too many features, select only the top ones based on variance
    if numeric_df.shape[1] > max_features:
        logger.info(f"Limiting heatmap to {max_features} features with highest variance")
        variances = numeric_df.var().sort_values(ascending=False)
        top_features = variances.index[:max_features].tolist()
        
        # Add target column if provided
        if target_col and target_col in df.columns:
            top_features.append(target_col)
            
        numeric_df = numeric_df[top_features]
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    return fig

def plot_performance_metrics(model, X_test, y_test):
    """
    Plot various performance metrics for a trained model.
    
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
    matplotlib.figure.Figure
        Performance metrics figure
    """
    logger.info("Creating performance metrics plots")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
        ax=axes[0]
    )
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Confusion Matrix')
    
    # ROC curve for binary classification
    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            
            axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (area = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('Receiver Operating Characteristic')
            axes[1].legend(loc="lower right")
        except:
            precision, recall, _ = precision_recall_curve(y_test, model.decision_function(X_test))
            pr_auc = auc(recall, precision)
            
            axes[1].plot(recall, precision, color='darkorange', lw=2,
                   label=f'PR curve (area = {pr_auc:.2f})')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('Recall')
            axes[1].set_ylabel('Precision')
            axes[1].set_title('Precision-Recall Curve')
            axes[1].legend(loc="lower left")
    else:
        # For multiclass, show feature importance if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature {i}' for i in range(X_test.shape[1])]
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            top_indices = indices[:min(10, len(indices))]
            
            axes[1].barh(range(len(top_indices)), importances[top_indices])
            axes[1].set_yticks(range(len(top_indices)))
            axes[1].set_yticklabels([feature_names[i] for i in top_indices])
            axes[1].set_xlabel('Feature Importance')
            axes[1].set_title('Top 10 Feature Importance')
        else:
            axes[1].text(0.5, 0.5, 'Feature importance not available', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axes[1].transAxes)
    
    plt.tight_layout()
    return fig

def plot_gene_expression_heatmap(expression_data, gene_list=None, sample_metadata=None, n_genes=50):
    """
    Create a heatmap visualization of gene expression data.
    
    Parameters:
    -----------
    expression_data : pandas.DataFrame
        Gene expression data with genes as rows and samples as columns
    gene_list : list, optional
        List of specific genes to include in the heatmap
    sample_metadata : pandas.DataFrame, optional
        Metadata for samples to use for annotation
    n_genes : int, default=50
        Number of top variable genes to include if gene_list is not provided
        
    Returns:
    --------
    matplotlib.figure.Figure
        Gene expression heatmap figure
    """
    logger.info("Creating gene expression heatmap")
    
    # Make a copy of the data
    data = expression_data.copy()
    
    # Select specific genes if provided, otherwise select top variable genes
    if gene_list:
        # Filter to only include genes in the list that exist in the data
        available_genes = [gene for gene in gene_list if gene in data.index]
        if not available_genes:
            logger.warning("None of the specified genes found in data")
            return None
        plot_data = data.loc[available_genes]
    else:
        # Calculate gene variance and select top N variable genes
        gene_var = data.var(axis=1).sort_values(ascending=False)
        top_genes = gene_var.index[:n_genes]
        plot_data = data.loc[top_genes]
    
    # Z-score normalization across samples for each gene
    plot_data = plot_data.apply(lambda x: (x - x.mean()) / x.std(), axis=1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, len(plot_data) * 0.3 + 3))
    
    # Create color palette for metadata annotation
    if sample_metadata is not None:
        col_colors = None
        if isinstance(sample_metadata, pd.DataFrame):
            # Create a color mapping for categorical columns
            col_colors = []
            for col in sample_metadata.columns:
                if sample_metadata[col].dtype == 'object' or sample_metadata[col].dtype.name == 'category':
                    # Create a color map for this column
                    categories = sample_metadata[col].unique()
                    cmap = plt.cm.get_cmap('tab10', len(categories))
                    lut = dict(zip(categories, [cmap(i) for i in range(len(categories))]))
                    col_colors.append(sample_metadata[col].map(lut))
            
            if col_colors:
                col_colors = pd.concat(col_colors, axis=1)
    else:
        col_colors = None
    
    # Create clustered heatmap
    sns.clustermap(
        plot_data,
        cmap="RdBu_r",
        center=0,
        figsize=(12, len(plot_data) * 0.3 + 3),
        col_colors=col_colors,
        row_cluster=True,
        col_cluster=True,
        dendrogram_ratio=(0.1, 0.2),
        cbar_pos=(0.02, 0.8, 0.05, 0.18)
    )
    
    plt.suptitle('Gene Expression Heatmap', y=1.02)
    plt.tight_layout()
    
    return fig