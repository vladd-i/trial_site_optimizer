# Clinical Trial Site Selection Optimizer

This project aims to optimize the process of selecting high quality sites to conduct clinical trials at. Utilizing data analysis and machine learning techniques, I strive to develop an AI-powered tool to shorten the total timeline from trial design to the Last Patient Last Visit (LPLV).

## Project Structure

Below is the overview of the directory structure of the project:

```
trial_site_optimizer/
│
├── notebooks/                       # Jupyter notebooks for data analysis and exploration
│   ├── exploratory_data_analysis.ipynb
│   ├── test_dataloader.ipynb
│   └── modeling.ipynb               # TODO
│
├── src/                             # Source code for the project
│   ├── __init__.py                  # TODO: Makes src a Python module
│   │
│   ├── data/                        # Scripts/modules for data loading and manipulation
│   │   ├── dataloader.py
│   │   ├── preprocess.py
│   │   └── feature_engineer.py
│   │   
│   └── models/                      # Model definitions and training scripts
│       └── models.py
│
├── tests/                           # TODO: Test suite for the project
│   ├── __init__.py
│   ├── test_data.py
│   └── test_models.py
│
├── configs/                         # Configuration files (e.g., for models, preprocessing)
│   └── config.yaml
│
├── requirements.txt                 # TODO: Project dependencies
├── setup.py                         # TODO: Setup script for installing the project module
├── .gitignore                       # Specifies intentionally untracked files to ignore
└── README.md                        # Project overview, setup, and usage instructions
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

- To inspect exploratory data analysis results, navigate to the `notebooks/` directory and open the Jupyter notebooks.

- For running the model training scripts and seeing the resulting metrics, ensure you are in the project root directory 
and execute:
    ```
    python3 -m src.models.models
    ```
