# data/feature_engineer.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        self.df = df

    def calculate_site_trial_duration_months(self):
        assert 'site_end_date' in self.df.columns, "'site_end_date' column is missing"
        assert 'site_start_date' in self.df.columns, "'site_start_date' column is missing"
        
        self.df['site_trial_duration_months'] = (
            self.df['site_end_date'] - self.df['site_start_date']
        ).dt.days / 30.44
        
        assert 'site_trial_duration_months' in self.df.columns, "Failed to calculate 'site_trial_duration_months'"
        return self

    def calculate_no_of_trials(self):
        assert 'site_id' in self.df.columns, "'site_id' column is missing"
        
        site_trial_counts = self.df['site_id'].value_counts().rename('no_of_trials')
        self.df = self.df.merge(site_trial_counts, how='left', left_on='site_id', right_index=True)
        
        assert 'no_of_trials' in self.df.columns, "Failed to calculate 'no_of_trials'"
        return self

    def calculate_patients_enroled_per_month(self):
        assert 'enrolment_months' in self.df.columns, "'enrolment_months' column is missing"
        assert 'no_of_patients' in self.df.columns, "'no_of_patients' column is missing"
        
        self.df['patients_enroled_per_month'] = np.where(
            self.df['enrolment_months'] < 1,
            self.df['no_of_patients'],
            self.df['no_of_patients'] / self.df['enrolment_months']
        )
        
        assert 'patients_enroled_per_month' in self.df.columns, "Failed to calculate 'patients_enroled_per_month'"
        return self

    def calculate_patients_treated_per_month(self):
        assert 'no_of_patients' in self.df.columns, "'no_of_patients' column is missing"
        assert 'site_trial_duration_months' in self.df.columns, "'site_trial_duration_months' column is missing after calculation"
        
        self.df['patients_treated_per_month'] = (
            self.df['no_of_patients'] / self.df['site_trial_duration_months']
        )
        
        assert 'patients_treated_per_month' in self.df.columns, "Failed to calculate 'patients_treated_per_month'"
        return self

    def one_hot_encode_variables(self):
        columns_to_one_hot_encode = ['site_type', 'trial_phase', 'nn_region']

        for column in columns_to_one_hot_encode:
            # Ensure the column exists.
            assert column in self.df.columns, f"'{column}' column is missing"
            
            # Remember the original number of unique values in the column.
            unique_values = self.df[column].nunique()
            
            # One-hot encode the column
            self.df = pd.get_dummies(self.df, columns=[column], prefix=column, drop_first=True)

            # Drop the original variable after one-hot-encoding if it exists.
            if column in self.df.columns:
                self.df = self.df.drop(columns=[column])

            assert column not in self.df.columns, f"'{column}' was not dropped after one-hot encoding"

            # Verifying that one-hot encoding created the correct number of new columns.
            one_hot_columns = [col for col in self.df.columns if col.startswith(f'{column}_')]

            # The -1 is because drop_first=True will drop one of the categories.
            assert len(one_hot_columns) == (unique_values - 1), \
                f"One-hot encoding for '{column}' did not create the expected number of columns"

        return self

    def calculate_site_quality_score(self):
        assert 'no_of_patients' in self.df.columns \
            and 'patients_enroled_per_month' in self.df.columns \
                and 'patients_treated_per_month' in self.df.columns, \
                    "Required columns for site quality score are missing"
                
        self.df["site_quality_score"] = (self.df["no_of_patients"] + 
                                         self.df["patients_enroled_per_month"] + 
                                         self.df["patients_treated_per_month"]) / 3
        
        return self
    
    def get_data(self):
        # Add an assertion to check that the DataFrame is not empty
        assert not self.df.empty, "The final DataFrame is empty"
        return self.df