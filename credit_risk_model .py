"""
Credit Risk Modeling - Logistic Regression Model Training and Evaluation
This module implements the final logistic regression model with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           f1_score, accuracy_score, precision_score, recall_score)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import joblib
import warnings
warnings.filterwarnings('ignore')

class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_performance = {}
        
    def load_woe_data(self, filepath='woe_transformed_data.csv'):
        """
        Load WOE-transformed dataset
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded WOE-transformed data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("WOE-transformed data file not found. Please run woe_iv_analysis.py first.")
            return None
    
    def prepare_data(self, df, target='default', test_size=0.2, random_state=42):
        """
        Prepare data for model training
        """
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id', target]]
        
        X = df[feature_cols]
        y = df[target]
        
        # Store feature names
        self.feature_names = feature_cols
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target distribution in training set:")
        print(y_train.value_counts(normalize=True))
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train, C=1.0, max_iter=1000):
        """
        Train logistic regression model
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = LogisticRegression(
            C=C, 
            max_iter=max_iter, 
            random_state=42,
            class_weight='balanced',
            solver='liblinear'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        print(f"Logistic Regression model trained successfully")
        print(f"Number of features: {len(self.feature_names)}")
        print(f"Regularization parameter C: {C}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        # Store performance metrics
        self.model_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        print("=== Model Performance ===")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC AUC:   {roc_auc:.4f}")
        print(f"PR AUC:    {pr_auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Details:")
        print(f"True Negatives:  {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
        print(f"True Positives:  {tp}")
    
    def plot_roc_curve(self, y_test, y_pred_proba):
        """
        Plot ROC curve
        """
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, y_test, y_pred_proba):
        """
        Plot Precision-Recall curve
        """
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Area Under Precision-Recall Curve (AUCPR): {pr_auc:.4f}")
    
    def plot_feature_importance(self):
        """
        Plot feature importance based on logistic regression coefficients
        """
        # Get feature coefficients
        coefficients = self.model.coef_[0]
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=True)
        
        plt.figure(figsize=(10, 8))
        colors = ['red' if x < 0 else 'green' for x in feature_importance['coefficient']]
        
        plt.barh(range(len(feature_importance)), 
                feature_importance['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Coefficient Value')
        plt.title('Feature Importance (Logistic Regression Coefficients)')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print top features
        print("\nTop 10 Most Important Features:")
        top_features = feature_importance.tail(10)
        for i, (_, row) in enumerate(top_features.iterrows()):
            print(f"{i+1:2d}. {row['feature']:<25} Coefficient: {row['coefficient']:>8.4f}")
    
    def cross_validation(self, X, y, cv_folds=5):
        """
        Perform cross-validation
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified K-Fold cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='roc_auc')
        cv_scores_pr = cross_val_score(self.model, X_scaled, y, cv=skf, scoring='average_precision')
        
        print(f"\n=== Cross-Validation Results ({cv_folds}-Fold) ===")
        print(f"ROC AUC Scores: {cv_scores}")
        print(f"ROC AUC Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"PR AUC Scores: {cv_scores_pr}")
        print(f"PR AUC Mean: {cv_scores_pr.mean():.4f} (+/- {cv_scores_pr.std() * 2:.4f})")
        
        return cv_scores, cv_scores_pr
    
    def calibration_plot(self, X_test, y_test):
        """
        Plot calibration curve to assess probability calibration
        """
        X_test_scaled = self.scaler.transform(X_test)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="Logistic Regression")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Plot')
        plt.legend()
        plt.grid(alpha=0.3