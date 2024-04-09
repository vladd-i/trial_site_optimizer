from typing import Dict, Tuple
import pandas as pd
import numpy as np
from configs.config import DATA_PATH
from .preprocess import DataPreprocessor
from .feature_engineer import FeatureEngineer

class DataLoader:
    """
    DataLoader is responsible for loading, cleaning, and preparing data for model training and evaluation.
    It handles data from various sources, merges them, and processes features for model input.
    """
    def __init__(self):
        """
        Initializes DataLoader with predefined file names.
        """
        self.file_names = ['country', 'trial_site', 'target', 'trial']

    def load_and_clean_data(self) -> Dict[str, pd.DataFrame]:
        """
        Loads data from parquet files, applies cleaning and preprocessing steps using DataPreprocessor.

        Returns:
            dict: A dictionary where keys are file names and values are preprocessed pandas DataFrames.
        """
        data = {}
        for file_name in self.file_names:
            df = pd.read_parquet(f"{DATA_PATH}/{file_name}.parquet")
            preprocessor = DataPreprocessor(df)

            if file_name == 'trial':
                df = preprocessor.clean_trial_ids().clean_ages().get_preprocessed_data()
                assert 'trial_id' in df.columns and 'maximum_age' in df.columns and 'minimum_age' in df.columns, \
                    f"Cleaning failed for {file_name}"

            if file_name == 'target':
                df = preprocessor.clean_trial_ids().clean_site_ids().get_preprocessed_data()
                assert 'trial_id' in df.columns and 'site_id' in df.columns, \
                    f"Cleaning failed for {file_name}"

            if file_name == 'trial_site':
                df = preprocessor.clean_trial_ids().clean_site_ids().clean_dates().get_preprocessed_data()
                assert 'trial_id' in df.columns and 'site_id' in df.columns and 'site_start_date' in df.columns and 'site_end_date' in df.columns, \
                    f"Cleaning failed for {file_name}"

            elif file_name == 'country':
                df = preprocessor.fill_missing_nn_region().get_preprocessed_data()
                assert 'nn_region' in df.columns and df['nn_region'].isnull().sum() == 0, \
                    f"nn_region imputation failed for {file_name}"

            data[file_name] = df

        return data

    def merge_drop_data(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merges data from multiple sources into a single DataFrame and applies additional cleaning.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary with preprocessed data frames.

        Returns:
            pd.DataFrame: The merged and cleaned DataFrame.
        """    
        # Set 'df_trial_site' as the base dataframe to preserve its indices.
        df_merged = data['trial_site'].copy()

        # Merge other dataframes onto 'df_trial_site' using appropriate keys.
        df_merged = df_merged.merge(data['target'], on=['trial_id', 'site_id'], how='left')
        df_merged = df_merged.merge(data['trial'], on='trial_id', how='left')
        df_merged = df_merged.merge(data['country'], on='country_id', how='left')

        preprocessor = DataPreprocessor(df_merged)
        df_dropped = (preprocessor
                    .drop_negative_enrolment_months()
                    .drop_trials_with_complete_missing_data()
                    .get_preprocessed_data())

        return df_dropped

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies feature engineering to the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to which feature engineering is applied.

        Returns:
            pd.DataFrame: The DataFrame with added features.
        """
        feature_engineer = FeatureEngineer(df)
        df_with_features = (feature_engineer
                            .calculate_site_trial_duration_months()
                            .calculate_no_of_trials()
                            .one_hot_encode_variables()
                            .get_data())

        return df_with_features

    def get_predictors_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and preprocesses predictor variables from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which predictors are extracted.

        Returns:
            pd.DataFrame: The DataFrame with predictor variables, scaled and imputed.
        """
        # Define the base variables that are not one-hot-encoded.
        country_vars = ["population", "oecd_pharma_expenditure_per_capita",
                        "oecd_medical_grads_per_1k", "oecd_MDs_per_1k",
                        "oecd_hospital_beds_per_1k", "oecd_MR_units_per_1m",
                        "oecd_pte_units_total", "oecd_perc_pop_insured_by_gov_health",
                        "oecd_perc_pop_insured_by_priv_health", 
                        "oecd_perc_pop_insured_by_priv_or_gov_health",
                        "wb_diabetes_prevalence_perc_pop_age_20_to_79",
                        "who_gho_ncd_paa_prev_insuff_physical_activity",
                        "who_gho_NCD_BMI_30A_obesity_prevalence_adults"]
        
        trial_vars = ["maximum_age", "minimum_age", "is_novo_nordisk_trial", "is_top_20_sponsor"]

        site_vars = ["no_of_trials"]

        # Extract one-hot-encoded columns for nn_region, trial_phase, and site_type.
        one_hot_country_vars = [col for col in df.columns if col.startswith('nn_region_')]
        one_hot_trial_vars = [col for col in df.columns if col.startswith('trial_phase_')]
        one_hot_site_vars = [col for col in df.columns if col.startswith('site_type_')]

        # Combine all predictor variables.
        predictor_columns = \
            (country_vars + trial_vars + site_vars + one_hot_country_vars + one_hot_trial_vars + one_hot_site_vars)
        df_predictors = df[predictor_columns]

        # Ensure all intended columns are present.
        assert all(column in df_predictors.columns for column in predictor_columns), "Incorrect predictor variables"
        
        # Scale the variables to mean 0, std 1. Impute certain missing vals with means of the columns.
        preprocessor = DataPreprocessor(df_predictors)
        df_predictors_scaled_imputed = (preprocessor
                    .standardize(indicator_vars=one_hot_country_vars + one_hot_trial_vars + one_hot_site_vars)
                    .impute_mean()
                    .get_preprocessed_data())
        return df_predictors_scaled_imputed
    

    def get_response_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts and processes the response variable from the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which the response variable is extracted.

        Returns:
            np.ndarray: The processed response variable as an array.
        """
        outcome_variables = ["no_of_patients", "enrolment_months", "site_trial_duration_months"]

        # Select only the outcome variables.
        df_outcomes = df[outcome_variables]

        # Ensure all intended columns are present.
        assert all(column in df_outcomes.columns for column in outcome_variables), "Missing outcome variables"

        # Standardize 'no_of_patients' and 'enrolment_months' before kNN imputation.
        preprocessor = DataPreprocessor(df_outcomes)
        df_outcomes_imputed = (preprocessor
                                    .standardize(columns=['no_of_patients', 'enrolment_months', 'site_trial_duration_months'])
                                    .impute_knn()
                                    .get_preprocessed_data())

        # Apply feature engineering to calculate ratios and site quality score.
        feature_engineer = FeatureEngineer(df_outcomes_imputed)
        df_response = (feature_engineer
                       .calculate_patients_enroled_per_month()
                       .calculate_patients_treated_per_month()
                       .get_data())
        
        # Standardize variables that factor into calculation of site_quality_score to weigh them equally.
        preprocessor = DataPreprocessor(df_response)
        df_response_scaled = (preprocessor
                              .standardize(columns=['no_of_patients', 
                                                    'patients_enroled_per_month', 
                                                    'patients_treated_per_month'])
                                    .get_preprocessed_data())
        
        feature_engineer = FeatureEngineer(df_response_scaled)
        df_response = (feature_engineer
                       .calculate_site_quality_score() 
                       .get_data())

        # Drop all variables except 'site_quality_score'.
        df_response = df_response[['site_quality_score']]

        assert 'site_quality_score' in df_response.columns and df_response.shape[1] == 1, \
                "DataFrame should contain only 'site_quality_score' as a column"

        return df_response.values
    
    def get_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepares the predictors and response data for model training.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: The predictors as a DataFrame and the response variable as an array.
        """
        data = self.load_and_clean_data()
        df = self.merge_drop_data(data)
        df_with_features = self.add_features(df)

        X = self.get_predictors_data(df_with_features)
        y = self.get_response_data(df_with_features)

        # Ensure X and y are pandas DataFrames
        X = pd.DataFrame(X, index=df_with_features.index)
        y = pd.DataFrame(y, index=df_with_features.index)

        return X, y

    def get_train_val_test_split(self) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Splits the data into training, validation, and test sets based on trial IDs.

        Returns:
            Tuple: Six elements consisting of training predictors, training response, validation predictors, validation response, test predictors, and test response.
        """
        # Load and clean data to get a DataFrame with trial_id and site_end_date.
        data = self.load_and_clean_data()
        df_merged = self.merge_drop_data(data)

        # Get the latest site_end_date for each trial_id.
        latest_dates = df_merged.groupby('trial_id')['site_end_date'].max().reset_index()

        # Sort the trials based on site_end_date.
        sorted_trials = latest_dates.sort_values('site_end_date')

        # Select the first 8 trial_ids for training
        trial_ids_train = sorted_trials['trial_id'][:8].tolist()
        
        # Remaining trial_ids for validation and test
        remaining_trial_ids = sorted_trials['trial_id'][8:].tolist()
        
        # Randomly split the remaining trial_ids into validation and test
        np.random.shuffle(remaining_trial_ids)
        midpoint = len(remaining_trial_ids) // 2
        trial_ids_val = remaining_trial_ids[:midpoint]
        trial_ids_test = remaining_trial_ids[midpoint:]

        X, y = self.get_data()  # Full dataset

        # Filtering function using preserved indices.
        def filter_by_indices(df_merged, X, y, trial_ids):
            # Filter 'df_merged' by 'trial_ids' and use the resulting indices to filter 'X' and 'y'.
            filtered_indices = df_merged[df_merged['trial_id'].isin(trial_ids)].index
            X_filtered, y_filtered = X.loc[filtered_indices], y.loc[filtered_indices]
            
            assert len(X_filtered) == len(y_filtered), "Mismatch in filtered X and y lengths."
            return X_filtered, y_filtered

        # Apply filtering for train, validation, and test sets.
        X_train, y_train = filter_by_indices(df_merged, X, y, trial_ids_train)
        X_val, y_val = filter_by_indices(df_merged, X, y, trial_ids_val)
        X_test, y_test = filter_by_indices(df_merged, X, y, trial_ids_test)

        assert len(X_train) == len(df_merged[df_merged['trial_id'].isin(trial_ids_train)]['site_id']), "X_train size mismatch"
        assert len(X_val) == len(df_merged[df_merged['trial_id'].isin(trial_ids_val)]['site_id']), "X_val size mismatch"
        assert len(X_test) == len(df_merged[df_merged['trial_id'].isin(trial_ids_test)]['site_id']), "X_test size mismatch"
        
        assert len(y_train) == len(df_merged[df_merged['trial_id'].isin(trial_ids_train)]['site_id']), "y_train size mismatch"
        assert len(y_val) == len(df_merged[df_merged['trial_id'].isin(trial_ids_val)]['site_id']), "y_val size mismatch"
        assert len(y_test) == len(df_merged[df_merged['trial_id'].isin(trial_ids_test)]['site_id']), "y_test size mismatch"

        return X_train, y_train, X_val, y_val, X_test, y_test
