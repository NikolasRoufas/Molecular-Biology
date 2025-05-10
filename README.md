# BioDataAnalyzer: Interactive Molecular Biology Data Analysis Platform

[![Docker](https://img.shields.io/badge/Docker-Available-blue)](https://www.docker.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.15.0-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Molecular Biology is an interactive web application built with Streamlit that provides a user-friendly interface for analyzing molecular biology datasets, particularly gene expression data. The application enables researchers and bioinformaticians to perform data preprocessing, apply machine learning algorithms, and visualize results without writing code.


## Features

- **Data Upload & Preprocessing**
  - Support for CSV, TSV, and Excel file formats
  - Multiple normalization methods (Z-score, MinMax, Quantile, etc.)
  - Missing value imputation
  - Feature selection options
  
- **Machine Learning Analysis**
  - Classification algorithms (SVM, Random Forest, XGBoost, etc.)
  - Clustering algorithms (K-means, Hierarchical, etc.)
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Hyperparameter tuning with cross-validation
  
- **Visualization**
  - Interactive PCA plots
  - Customizable heatmaps
  - Feature importance charts
  - Clustering visualizations
  - ROC curves and confusion matrices

- **Results Export**
  - Download preprocessed datasets
  - Export trained models
  - Save visualization figures

## Installation

### Using Docker (Recommended)

The easiest way to run BioDataAnalyzer is using Docker:

```bash
# Pull the image
docker pull username/biodataanalyzer:latest

# Run the container
docker run -p 8501:8501 username/biodataanalyzer:latest
```

Then open your browser and navigate to `http://localhost:8501`

### Manual Installation

1. Clone the repository:


2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Project Structure

```
Molecular-Biology/
├── app.py                  # Main Streamlit application entry point
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── config/                 # Configuration files
├── modules/                # Core functionality modules
│   ├── preprocessing/      # Data preprocessing modules
│   ├── ml/                 # Machine learning algorithms
│   ├── visualization/      # Data visualization functions
│   └── utils/              # Utility functions
├── data/                   # Sample datasets and data storage
├── models/                 # Saved model files
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation and images
```

## Usage Guide

### 1. Data Upload
- Start by uploading your dataset in CSV, TSV, or Excel format
- The first row should contain feature names
- One column should be designated as the target/class variable

### 2. Data Preprocessing
- Select columns to include in analysis
- Choose normalization/scaling method
- Set parameters for missing value imputation
- Apply optional feature selection techniques

### 3. Machine Learning Analysis
- Choose an algorithm based on your analysis goals
- Set hyperparameters or use auto-tuning option
- Configure cross-validation settings
- Train and evaluate the model

### 4. Results Visualization
- Explore dimensionality reduction plots
- Analyze feature importance
- Review model performance metrics
- Generate publication-ready figures

## Development

### Running Tests
```bash
pytest
```

### Building Docker Image
```bash
docker build -t username/Molecular-Biology:latest .
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
