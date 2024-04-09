import sys
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning
from src.data.dataloader import DataLoader
from typing import Dict, Any, List

# Suppress specific sklearn and pandas warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
pd.options.mode.chained_assignment = None  # default='warn'

class BaselineModel:
    """A simple linear regression model to serve as a baseline."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}
    

class LassoModel:
    """A linear regression model with Lasso regularization."""  

    def __init__(self, alpha=0.01):
        self.model = Lasso(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}

    def feature_importance(self, feature_names):
        importance = dict(zip(feature_names, self.model.coef_))
        # Filter out features with zero importance and sort by absolute magnitude.
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True) if v != 0}
        return importance

class LassoModelWithCV:
    """A Lasso model with built-in cross-validation to find the optimal alpha value."""

    def __init__(self, alphas):
        self.alphas = alphas
        self.model_cv = LassoCV(alphas=self.alphas, cv=5, random_state=42)

    def fit(self, X, y):
        self.model_cv.fit(X, y)
        self.best_alpha = self.model_cv.alpha_

    def predict(self, X):
        return self.model_cv.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}

    def feature_importance(self, feature_names):
        # Check if feature_names is a DataFrame, extract columns if so.
        if isinstance(feature_names, pd.DataFrame):
            feature_names = feature_names.columns

        # Now feature_names should be an Index or list-like, representing column names.
        importance = {feature: coef for feature, coef in zip(feature_names, self.model_cv.coef_) if coef != 0}
        # Sort by absolute magnitude
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        return importance

class InteractionLassoModel:
    """A Lasso model that first transforms features to include polynomial and interaction terms."""

    def __init__(self, alphas):
        self.alphas = alphas
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.model = LassoCV(alphas=self.alphas, cv=5, max_iter=10000, random_state=42)

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        self.best_alpha = self.model.alpha_

    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.model.predict(X_poly)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}

    def feature_importance(self, X):
        # Check if X is a DataFrame, extract columns if so.
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns
        else:
            # Assuming X is already an Index or list-like of column names.
            feature_names = X

        # Get feature names for interaction terms.
        interaction_feature_names = self.poly.get_feature_names_out(feature_names)

        # Extract non-zero coefficients and their corresponding feature names.
        importance = {feature: coef for feature, coef in zip(interaction_feature_names, self.model.coef_) if coef != 0}

        # Sort by absolute magnitude.
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        return importance


class RandomForestModel:
    """A random forest regressor model."""
    
    def __init__(self, n_estimators=10, random_state=42):
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, np.ravel(y_train))  # Ensuring y_train is the correct shape.

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}

    def feature_importance(self, feature_names):
        importance = dict(zip(feature_names, self.model.feature_importances_))
        # Sorting by absolute magnitude
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        return importance


class TunedRandomForestModel:
    """A Random Forest model that uses randomized search to tune hyperparameters."""

    def __init__(self, param_distributions, n_iter=100, cv=3, random_state=42):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state
        self.model = None

    def fit(self, X_train, y_train):
        # Base model to tune
        rf = RandomForestRegressor(random_state=self.random_state)
        # Random search of parameters
        self.model = RandomizedSearchCV(estimator=rf, 
                                        param_distributions=self.param_distributions, 
                                        n_iter=self.n_iter, 
                                        cv=self.cv, 
                                        verbose=0, 
                                        random_state=self.random_state, 
                                        n_jobs=-1)
        # Fit the random search model
        self.model.fit(X_train, np.ravel(y_train))

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {'MSE': mse, 'R2': r2}

    def best_params(self):
        return self.model.best_params_

    def feature_importance(self, feature_names):
        best_estimator = self.model.best_estimator_
        importance = dict(zip(feature_names, best_estimator.feature_importances_))
        importance = {k: v for k, v in sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True)}
        return importance

if __name__ == "__main__":
    dataloader = DataLoader()
    X_train, y_train, X_val, y_val, X_test, y_test = dataloader.get_train_val_test_split()

    print("Fit BaselineModel")
    baseline_model = BaselineModel()
    baseline_model.fit(X_train, y_train)

    print("BaselineModel Train Performance:", baseline_model.evaluate(X_train, y_train))
    print("BaselineModel Validation Performance:", baseline_model.evaluate(X_val, y_val))
    print("BaselineModel Test Performance:", baseline_model.evaluate(X_test, y_test))

    print("\n\n\nFit LassoModel")
    lasso_model = LassoModel(alpha=0.01) 
    lasso_model.fit(X_train, y_train)

    print("\tLassoModel Train Performance:", lasso_model.evaluate(X_train, y_train))
    print("\tLassoModel Validation Performance:", lasso_model.evaluate(X_val, y_val))
    print("\tLassoModel Test Performance:", lasso_model.evaluate(X_test, y_test))

    feature_importance = lasso_model.feature_importance(X_train.columns)
    print("\n\n\nLassoModel Feature Importance:", feature_importance)
 
    # Combine X_train and X_val for training LassoCV.
    X_train_val_combined = pd.concat([X_train, X_val])
    y_train_val_combined = np.concatenate([y_train, y_val])

    alphas = np.logspace(-6, 1, 50)

    print("\n\n\nFit LassoModelWithCV")
    lasso_model_cv = LassoModelWithCV(alphas=alphas)
    lasso_model_cv.fit(X_train_val_combined, y_train_val_combined)

    print(f"\tBest alpha: {lasso_model_cv.best_alpha}")
    print("\tLassoModelWithCV Combined Train and Validation Performance:", 
          lasso_model_cv.evaluate(X_train_val_combined, y_train_val_combined))
    print("\tLassoModelWithCV Test Performance:", lasso_model_cv.evaluate(X_test, y_test))
    print("\tLassoModelWithCV Feature Importance:", lasso_model_cv.feature_importance(X_train))

    print("\n\n\nFit InteractionLassoModel")
    interaction_lasso_model = InteractionLassoModel(alphas=alphas)
    interaction_lasso_model.fit(X_train_val_combined, y_train_val_combined)

    print("\tInteractionLassoModel Combined Train and Validation Performance:", 
          interaction_lasso_model.evaluate(X_train_val_combined, y_train_val_combined))
    print("\tInteractionLassoModel Test Performance:", interaction_lasso_model.evaluate(X_test, y_test))

    print("\tInteractionLassoModel Feature Importance:", interaction_lasso_model.feature_importance(X_train))

    print("\n\n\nFit RandomForestModel")
    random_forest_model = RandomForestModel(n_estimators=100)
    random_forest_model.fit(X_train, y_train)

    print("\tRandomForestModel Train Performance:", random_forest_model.evaluate(X_train, y_train))
    print("\tRandomForestModel Validation Performance:", random_forest_model.evaluate(X_val, y_val))
    print("\tRandomForestModel Test Performance:", random_forest_model.evaluate(X_test, y_test))

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=50, stop=1000, num=10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)

    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap
    }

    print("\n\n\nFit TunedRandomForestModel")
    tuned_random_forest_model = TunedRandomForestModel(random_grid)
    tuned_random_forest_model.fit(X_train_val_combined, y_train_val_combined)

    print("\tTunedRandomForestModel Best Parameters:", tuned_random_forest_model.best_params())
    print("\tTunedRandomForestModel Combined Train and Validation Performance:", 
        tuned_random_forest_model.evaluate(X_train_val_combined, y_train_val_combined))
    print("\tTunedRandomForestModel Test Performance:", tuned_random_forest_model.evaluate(X_test, y_test))

    # Comparing models in a table.
    models = {
        "BaselineModel": baseline_model,
        "LassoModel": lasso_model,
        "LassoModelWithCV": lasso_model_cv,
        "InteractionLassoModel": interaction_lasso_model,
        "RandomForestModel": random_forest_model,
        "TunedRandomForestModel": tuned_random_forest_model
    }

    performances = []
    for model_name, model in models.items():
        performances.append({
            "Model": model_name,
            "Train MSE": model.evaluate(X_train, y_train)["MSE"],
            "Validation MSE": model.evaluate(X_val, y_val)["MSE"],
            "Test MSE": model.evaluate(X_test, y_test)["MSE"],
            "Train R2": model.evaluate(X_train, y_train)["R2"],
            "Validation R2": model.evaluate(X_val, y_val)["R2"],
            "Test R2": model.evaluate(X_test, y_test)["R2"],
        })

    # Converting the performances list to a DataFrame for display.
    performances_df = pd.DataFrame(performances)
    print("\nModel Performances:")
    print(performances_df.to_string(index=False))

    # Displaying feature importance for each model with clear labeling.
    for model_name, model in models.items():
        if hasattr(model, 'feature_importance'):
            print(f"\n{model_name} Feature Importance:")
            importance = model.feature_importance(X_train.columns)
            print(importance)