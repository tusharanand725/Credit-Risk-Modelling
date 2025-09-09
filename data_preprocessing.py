"""
Credit Risk Modeling - Data Preprocessing and Cleaning
This module handles data loading, cleaning, and initial preprocessing steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def load_data(self, filepath=None):
        """
        Load credit risk dataset
        If no filepath provided, generate synthetic data for demonstration
        """
        if filepath:
            try:
                data = pd.read_csv(filepath)
                return data
            except FileNotFoundError:
                print("File not found. Generating synthetic data...")
                
        # Generate synthetic credit data for demonstration
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.randint(18, 80, n_samples),
            'income': np.random.exponential(50000, n_samples),
            'employment_years': np.random.randint(0, 40, n_samples),
            'debt_to_income_ratio': np.random.uniform(0, 1, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'num_credit_accounts': np.random.randint(1, 20, n_samples),
            'credit_utilization': np.random.uniform(0, 1, n_samples),
            'payment_history': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples, 
                                              p=[0.3, 0.4, 0.2, 0.1]),
            'loan_amount': np.random.exponential(25000, n_samples),
            'loan_purpose': np.random.choice(['home', 'auto', 'personal', 'business'], n_samples),
            'collateral': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (loan default) based on credit score and other factors
        default_prob = (
            0.5 * (1 - (df['credit_score'] - 300) / 550) +
            0.2 * df['debt_to_income_ratio'] +
            0.1 * (df['credit_utilization'] > 0.8).astype(int) +
            0.1 * (df['payment_history'] == 'Poor').astype(int) +
            0.1 * np.random.random(n_samples)
        )
        
        df['default'] = np.random.binomial(1, np.clip(default_prob, 0, 1), n_samples)
        
        # Introduce some missing values
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'income'] = np.nan
        
        missing_indices = np.random.choice(df.index, size=int(0.03 * len(df)), replace=False)
        df.loc[missing_indices, 'employment_years'] = np.nan
        
        return df
    
    def explore_data(self, df):
        """
        Perform initial data exploration
        """
        print("Dataset Shape:", df.shape)
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing Values:")
        print(df.isnull().sum())
        print("\nTarget Distribution:")
        print(df['default'].value_counts(normalize=True))
        
        # Statistical summary
        print("\nNumerical Features Summary:")
        print(df.describe())
        
        return df
    
    def handle_missing_values(self, df):
        """
        Handle missing values using appropriate imputation strategies
        """
        # Numerical features - use median imputation
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['customer_id', 'default']]
        
        for col in numerical_features:
            if df[col].isnull().any():
                imputer = SimpleImputer(strategy='median')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                self.imputers[col] = imputer
                print(f"Imputed missing values in {col} using median")
        
        # Categorical features - use mode imputation
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_features:
            if df[col].isnull().any():
                imputer = SimpleImputer(strategy='most_frequent')
                df[col] = imputer.fit_transform(df[[col]]).ravel()
                self.imputers[col] = imputer
                print(f"Imputed missing values in {col} using mode")
        
        return df
    
    def detect_outliers(self, df, method='iqr'):
        """
        Detect outliers in numerical features
        """
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_features = [col for col in numerical_features if col not in ['customer_id', 'default']]
        
        outliers = {}
        
        for col in numerical_features:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > 3].index.tolist()
        
        return outliers
    
    def handle_outliers(self, df, outliers, method='cap'):
        """
        Handle outliers using capping or removal
        """
        if method == 'cap':
            for col, outlier_indices in outliers.items():
                if outlier_indices:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    df.loc[df[col] < lower_bound, col] = lower_bound
                    df.loc[df[col] > upper_bound, col] = upper_bound
                    print(f"Capped outliers in {col}")
        
        elif method == 'remove':
            all_outlier_indices = set()
            for outlier_indices in outliers.values():
                all_outlier_indices.update(outlier_indices)
            df = df.drop(list(all_outlier_indices))
            print(f"Removed {len(all_outlier_indices)} outlier records")
        
        return df
    
    def encode_categorical_features(self, df):
        """
        Encode categorical features
        """
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_features:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            print(f"Encoded categorical feature: {col}")
        
        return df
    
    def create_derived_features(self, df):
        """
        Create derived features for better model performance
        """
        # Credit utilization categories
        df['credit_util_category'] = pd.cut(df['credit_utilization'], 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['Low', 'Medium', 'High'])
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], 
                               bins=[0, 30, 45, 60, 100], 
                               labels=['Young', 'Middle', 'Senior', 'Elder'])
        
        # Income to loan ratio
        df['income_to_loan_ratio'] = df['income'] / (df['loan_amount'] + 1)
        
        # Credit score categories
        df['credit_score_category'] = pd.cut(df['credit_score'], 
                                           bins=[0, 580, 670, 740, 850], 
                                           labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        print("Created derived features: credit_util_category, age_group, income_to_loan_ratio, credit_score_category")
        
        return df
    
    def save_cleaned_data(self, df, filepath='cleaned_credit_data.csv'):
        """
        Save the cleaned dataset
        """
        df.to_csv(filepath, index=False)
        print(f"Cleaned data saved to {filepath}")
        return df

def main():
    """
    Main function to demonstrate data preprocessing pipeline
    """
    print("=== Credit Risk Modeling - Data Preprocessing ===")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load data
    print("\n1. Loading data...")
    df = preprocessor.load_data()
    
    # Explore data
    print("\n2. Exploring data...")
    df = preprocessor.explore_data(df)
    
    # Handle missing values
    print("\n3. Handling missing values...")
    df = preprocessor.handle_missing_values(df)
    
    # Detect and handle outliers
    print("\n4. Detecting outliers...")
    outliers = preprocessor.detect_outliers(df)
    print(f"Outliers detected in {len(outliers)} features")
    
    print("\n5. Handling outliers...")
    df = preprocessor.handle_outliers(df, outliers, method='cap')
    
    # Encode categorical features
    print("\n6. Encoding categorical features...")
    df = preprocessor.encode_categorical_features(df)
    
    # Create derived features
    print("\n7. Creating derived features...")
    df = preprocessor.create_derived_features(df)
    
    # Save cleaned data
    print("\n8. Saving cleaned data...")
    df = preprocessor.save_cleaned_data(df)
    
    print("\nData preprocessing completed successfully!")
    return df

if __name__ == "__main__":
    cleaned_data = main()
