# data/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from typing import List

class DataPreprocessor:
    def __init__(self, df):
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        self.df = df

    def clean_trial_ids(self):
        assert 'trial_id' in self.df.columns, "'trial_id' column is missing"
        
        self.df['trial_id'] = self.df['trial_id'].str.replace('trial_', '').astype(int)
        
        # Ensure the conversion was successful
        assert self.df['trial_id'].dtype == 'int', "'trial_id' is not an integer after cleaning"
        return self
    
    def clean_site_ids(self):
        assert 'site_id' in self.df.columns, "'site_id' column is missing"
        
        self.df['site_id'] = self.df['site_id'].str.replace('site_', '').astype(int)
        
        # Ensure the conversion was successful
        assert self.df['site_id'].dtype == 'int', "'site_id' is not an integer after cleaning"
        return self

    def clean_dates(self):
        assert 'site_start_date' in self.df.columns, "'site_start_date' column is missing"
        assert 'site_end_date' in self.df.columns, "'site_end_date' column is missing"

        self.df['site_start_date'] = pd.to_datetime(self.df['site_start_date'])
        self.df['site_end_date'] = pd.to_datetime(self.df['site_end_date'])
        
        # Verify the conversion
        assert pd.api.types.is_datetime64_any_dtype(self.df['site_start_date']), \
            "'site_start_date' is not datetime type after cleaning"
        assert pd.api.types.is_datetime64_any_dtype(self.df['site_end_date']), \
            "'site_end_date' is not datetime type after cleaning"
        return self
    
    def clean_ages(self):
        # Whenever maximum_age is 0, set maximum_age to the max of the column self.df["maximum_age"].
        self.df['maximum_age'] = self.df['maximum_age'].replace(0, self.df['maximum_age'].max())
        assert self.df['maximum_age'].min() > 0, "Maximum age contains zero or negative values"
        assert (self.df['maximum_age'] > self.df['minimum_age']).all(), \
            "Maximum age is not consistently greater than minimum age"

        return self
    
    def impute_mean(self):
        # Impute missing variables in the following columns with means of the corresponding columns in the dataset.
        columns_to_impute = \
            ["population", "oecd_pharma_expenditure_per_capita", "oecd_medical_grads_per_1k", "oecd_MDs_per_1k", 
             "oecd_hospital_beds_per_1k", "oecd_MR_units_per_1m", "oecd_pte_units_total",
             "oecd_perc_pop_insured_by_gov_health", "oecd_perc_pop_insured_by_priv_health", 
             "oecd_perc_pop_insured_by_priv_or_gov_health", "wb_diabetes_prevalence_perc_pop_age_20_to_79",
             "who_gho_ncd_paa_prev_insuff_physical_activity", "who_gho_NCD_BMI_30A_obesity_prevalence_adults"]
        for column in columns_to_impute:
            self.df[column].fillna(self.df[column].mean(), inplace=True)
        assert not self.df[columns_to_impute].isnull().any().any(), "Mean imputation failed"
        return self
    
    def impute_knn(self, columns: List[str] = None, k=10):
        if columns is None:
            columns = self.df.columns
        # Assert that all columns to be imputed exist in the DataFrame.

        assert all(column in columns for column in columns), \
            f"One or more columns from {columns} not found in DataFrame for kNN imputing."
        
        imputer = KNNImputer(n_neighbors=k)
        self.df[columns] = imputer.fit_transform(self.df[columns])
        
        # Assert that there are no missing values in the specified columns after imputation.
        assert not self.df[columns].isnull().any().any(), \
            f"Missing values present in {columns} after kNN imputation."
        
        return self

    def fill_missing_nn_region(self):
        self.df.loc[self.df["country_id"] == "GTM", "nn_region"] = "LATAM"
        self.df.loc[self.df["country_id"] == "PER", "nn_region"] = "LATAM"

        # Assert no more missing entries in 'nn_region'.
        assert not self.df['nn_region'].isnull().any(), "There are still missing entries in 'nn_region'."

        return self

    def drop_negative_enrolment_months(self):
        assert 'enrolment_months' in self.df.columns, "'enrolment_months' column is missing"
        initial_count = len(self.df)
        self.df = self.df.loc[~(self.df['enrolment_months'] < 0)]
        final_count = len(self.df)
        
        # Check if any rows were dropped
        assert final_count <= initial_count, "Number of rows increased after dropping negative 'enrolment_months'"
        return self

    def drop_trials_with_complete_missing_data(self):
        assert 'trial_id' in self.df.columns, "'trial_id' column is missing"
        initial_count = len(self.df)
        
        missing_percentage = self.df.groupby('trial_id').apply(lambda x: pd.Series({
            'perc_missing_no_of_patients': x['no_of_patients'].isnull().mean() * 100,
            'perc_missing_enrolment_months': x['enrolment_months'].isnull().mean() * 100
        })).reset_index()
        
        trials_to_drop = missing_percentage[
            (missing_percentage['perc_missing_no_of_patients'] == 100) |
            (missing_percentage['perc_missing_enrolment_months'] == 100)
        ]['trial_id']
        
        self.df = self.df[~self.df['trial_id'].isin(trials_to_drop)]
        final_count = len(self.df)
        
        # Ensure rows were dropped
        assert final_count <= initial_count, "Number of rows increased after dropping trials with complete missing data"
        return self
    
    def standardize(self, columns=None):
        scaler = StandardScaler()
        if columns is None:
            columns = self.df.columns
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return self

    def standardize(self, columns: List[str] = None, indicator_vars: List[str] = None):
        scaler = StandardScaler()
        
        if columns is None:
            columns = self.df.columns
        if indicator_vars:
            columns = [col for col in columns if col not in indicator_vars]

        # Scale only non-indicator variables
        self.df[columns] = scaler.fit_transform(self.df[columns])

        # Ensure indicator variables remain untouched.
        if indicator_vars:
            assert all(self.df[indicator_vars].isin([0, 1]).all()), "Indicator variables altered during scaling"
        
        # Verify the mean and std of scaled columns.
        assert self.df[columns].mean().abs().max() < 1e-2, "Mean of scaled columns is not close to 0"
        assert (self.df[columns].std() - 1).abs().max() < 1e-2, "Std of scaled columns is not close to 1"
        
        return self

    def get_preprocessed_data(self):
        assert not self.df.empty, "The cleaned DataFrame is empty"
        return self.df 