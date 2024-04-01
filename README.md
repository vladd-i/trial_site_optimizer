# Clinical Trial Site Selection Optimizer

This project aims to optimize the process of selecting high quality sites to conduct clinical trials at. Utilizing data analysis and machine learning techniques, I strive to develop an AI-powered tool to shorten the total timeline from trial design to the Last Patient Last Visit (LPLV).

## Project Structure

Below is the overview of the directory structure of the project:

```
trial_site_optimizer/
│
├── notebooks/                       # Jupyter notebooks for data analysis and exploration
│   ├── exploratory_data_analysis.ipynb
│   └── feature_analysis.ipynb       # TODO
│
├── src/                             # Source code for the project
│   ├── __init__.py                  # Makes src a Python module
│   │
│   ├── data/                        # Scripts/modules for data loading and manipulation
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │   └── feature_engineer.py
│   │   │
│   ├── models/                      # Model definitions and training scripts
│   │
│   └── utils/                       # Utility functions and classes
│
├── tests/                           # Test suite for the project
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
│
├── configs/                         # Configuration files (e.g., for models, preprocessing)
│   └── config.yaml
│
├── requirements.txt                 # Project dependencies
├── setup.py                         # Setup script for installing the project module
├── .gitignore                       # Specifies intentionally untracked files to ignore
├── README.md                        # Project overview, setup, and usage instructions
└── .github/workflows/               # CI/CD pipeline definitions for GitHub Actions
    ├── data_validation.yml          # Data validation workflow
    └── model_training.yml           # Model training and evaluation workflow
```

## Setup and Installation

1. Clone this repository:
    ```
    git clone git@github.com:vladd-i/trial_site_optimizer.git
    ```
2. Install required dependencies:
    ```
    pip install -r requirements.txt
    ```

3. Set up the environment variables and configurations as needed in the `configs/` directory.

## Usage

- To perform exploratory data analysis, navigate to the `notebooks/` directory and open the Jupyter notebooks.

- For running the model training scripts, ensure you are in the project root directory and execute #TODO
