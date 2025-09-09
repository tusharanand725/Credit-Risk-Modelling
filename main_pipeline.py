"""
Credit Risk Modeling - Complete Pipeline
This script runs the entire credit risk modeling pipeline from data preprocessing to model evaluation
"""

import sys
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from data_preprocessing import DataPreprocessor, main as preprocessing_main
from feature_engineering import FeatureEngineer, main as feature_engineering_main
from woe_iv_analysis import WOEIVAnalyzer, main as woe_iv_main
from credit_risk_model import CreditRiskModel, main as model_main

class CreditRiskPipeline:
    def __init__(self):
        self.start_time = None
        self.pipeline_results = {}
        
    def log_step(self, step_name, status="STARTED"):
        """
        Log pipeline step with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] {step_name}: {status}")
        if status == "STARTED":
            print("-" * 80)
    
    def run_complete_pipeline(self, run_preprocessing=True, run_feature_engineering=True,
                            run_woe_iv=True, run_modeling=True):
        """
        Run the complete credit risk modeling pipeline
        """
        self.start_time = time.time()
        
        print("=" * 80)
        print("CREDIT RISK MODELING - COMPLETE PIPELINE")
        print("=" * 80)
        print("Pipeline Components:")
        print("1. Data Preprocessing and Cleaning")
        print("2. Feature Engineering (PCA, Correlation Analysis, K-Means)")
        print("3. WOE/IV Analysis and Data Binning")
        print("4. Logistic Regression Model Training and Evaluation")
        print("=" * 80)
        
        try:
            # Step 1: Data Preprocessing
            if run_preprocessing:
                self.log_step("STEP 1: DATA PREPROCESSING")
                preprocessor = DataPreprocessor()
                
                # Load and clean data
                df = preprocessor.load_data()
                df = preprocessor.explore_data(df)
                df = preprocessor.handle_missing_values(df)
                
                # Handle outliers
                outliers = preprocessor.detect_outliers(df)
                df = preprocessor.handle_outliers(df, outliers, method='cap')
                
                # Encode categorical features and create derived features
                df = preprocessor.encode_categorical_features(df)
                df = preprocessor.create_derived_features(df)
                
                # Save cleaned data
                cleaned_data = preprocessor.save_cleaned_data(df)
                
                self.pipeline_results['preprocessing'] = {
                    'status': 'completed',
                    'data_shape': df.shape,
                    'features_created': len([col for col in df.columns if col.endswith('_encoded')]),
                    'outliers_detected': len(outliers)
                }
                
                self.log_step("STEP 1: DATA PREPROCESSING", "COMPLETED")
            else:
                print("\nSkipping data preprocessing...")
            
            # Step 2: Feature Engineering
            if run_feature_engineering:
                self.log_step("STEP 2: FEATURE ENGINEERING")
                feature_engineer = FeatureEngineer()
                
                # Load cleaned data
                df = feature_engineer.load_cleaned_data()
                if df is None:
                    raise Exception("Cleaned data not found. Please run preprocessing first.")
                
                # Correlation analysis
                correlation_matrix, high_corr_pairs = feature_engineer.correlation_analysis(df)
                df, removed_features = feature_engineer.remove_highly_correlated_features(
                    df, high_corr_pairs, correlation_matrix)
                
                # PCA analysis
                df_pca, original_features = feature_engineer.perform_pca(df, variance_threshold=0.95)
                
                # Combine with categorical features
                categorical_cols = [col for col in df.columns if col.endswith('_encoded')]
                other_cols = ['customer_id', 'default'] + categorical_cols
                df_combined = pd.concat([df[other_cols], df_pca], axis=1)
                
                # K-Means clustering
                df_combined, optimal_k = feature_engineer.kmeans_clustering(df_combined)
                
                # Create interaction features
                df_combined = feature_engineer.create_feature_interaction(df_combined)
                
                # Feature selection
                df_final, selected_features = feature_engineer.feature_selection(df_combined, k=15)
                
                # Save engineered features
                engineered_data = feature_engineer.save_engineered_features(df_final)
                
                self.pipeline_results['feature_engineering'] = {
                    'status': 'completed',
                    'pca_components': df_pca.shape[1],
                    'clusters_created': optimal_k,
                    'features_selected': len(selected_features),
                    'final_shape': df_final.shape
                }
                
                self.log_step("STEP 2: FEATURE ENGINEERING", "COMPLETED")
            else:
                print("\nSkipping feature engineering...")
            
            # Step 3: WOE/IV Analysis
            if run_woe_iv:
                self.log_step("STEP 3: WOE/IV ANALYSIS")
                woe_iv_analyzer = WOEIVAnalyzer()
                
                # Load engineered features
                df = woe_iv_analyzer.load_data()
                if df is None:
                    raise Exception("Engineered features not found. Please run feature engineering first.")
                
                # Perform WOE/IV analysis
                results_df = woe_iv_analyzer.perform_woe_iv_analysis(df)
                
                # Apply WOE transformation
                df_woe = woe_iv_analyzer.apply_woe_transformation(df)
                
                # Create visualizations
                woe_iv_analyzer.plot_woe_iv_analysis(results_df)
                
                # Select features based on IV
                selected_features = woe_iv_analyzer.select_features_by_iv(results_df, min_iv=0.1, max_features=15)
                
                # Create final WOE dataset
                df_woe_final = woe_iv_analyzer.create_woe_dataset(df_woe, selected_features)
                
                # Save results
                woe_data, iv_results = woe_iv_analyzer.save_woe_results(df_woe_final, results_df)
                
                self.pipeline_results['woe_iv_analysis'] = {
                    'status': 'completed',
                    'features_analyzed': len(results_df),
                    'strong_features': len(results_df[results_df['iv_score'] >= 0.3]),
                    'selected_features': len(selected_features),
                    'final_woe_shape': df_woe_final.shape
                }
                
                self.log_step("STEP 3: WOE/IV ANALYSIS", "COMPLETED")
            else:
                print("\nSkipping WOE/IV analysis...")
            
            # Step 4: Model Training and Evaluation
            if run_modeling:
                self.log_step("STEP 4: MODEL TRAINING AND EVALUATION")
                credit_model = CreditRiskModel()
                
                # Load WOE data
                df = credit_model.load_woe_data()
                if df is None:
                    raise Exception("WOE-transformed data not found. Please run WOE/IV analysis first.")
                
                # Prepare data
                X_train, X_test, y_train, y_test = credit_model.prepare_data(df)
                
                # Train model
                model = credit_model.train_logistic_regression(X_train, y_train)
                
                # Evaluate model
                y_pred, y_pred_proba = credit_model.evaluate_model(X_test, y_test)
                
                # Cross-validation
                cv_scores, cv_scores_pr = credit_model.cross_validation(X_train, y_train)
                
                # Create visualizations
                credit_model.plot_confusion_matrix(y_test, y_pred)
                credit_model.plot_roc_curve(y_test, y_pred_proba)
                credit_model.plot_precision_recall_curve(y_test, y_pred_proba)
                credit_model.plot_feature_importance()
                credit_model.calibration_plot(X_test, y_test)
                
                # Generate scorecard and summary
                scorecard = credit_model.generate_scorecard()
                summary_df = credit_model.model_summary_report()
                
                # Save model
                credit_model.save_model()
                
                self.pipeline_results['modeling'] = {
                    'status': 'completed',
                    'roc_auc': credit_model.model_performance['roc_auc'],
                    'pr_auc': credit_model.model_performance['pr_auc'],
                    'accuracy': credit_model.model_performance['accuracy'],
                    'f1_score': credit_model.model_performance['f1_score'],
                    'cv_roc_auc_mean': cv_scores.mean(),
                    'cv_pr_auc_mean': cv_scores_pr.mean()
                }
                
                self.log_step("STEP 4: MODEL TRAINING AND EVALUATION", "COMPLETED")
            else:
                print("\nSkipping model training...")
            
            # Pipeline completion summary
            self.generate_pipeline_summary()
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed with error: {str(e)}")
            print("Please check the error and run the individual components.")
            return False
        
        return True
    
    def generate_pipeline_summary(self):
        """
        Generate comprehensive pipeline summary
        """
        end_time = time.time()
        total_time = end_time - self.start_time
        
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 80)
        
        print(f"Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n--- Component Status ---")
        for component, results in self.pipeline_results.items():
            print(f"{component.upper():<25}: {results['status'].upper()}")
        
        # Detailed results
        if 'preprocessing' in self.pipeline_results:
            prep_results = self.pipeline_results['preprocessing']
            print(f"\n--- Data Preprocessing Results ---")
            print(f"Final Dataset Shape: {prep_results['data_shape']}")
            print(f"Encoded Features Created: {prep_results['features_created']}")
            print(f"Outlier Groups Detected: {prep_results['outliers_detected']}")
        
        if 'feature_engineering' in self.pipeline_results:
            feat_results = self.pipeline_results['feature_engineering']
            print(f"\n--- Feature Engineering Results ---")
            print(f"PCA Components Created: {feat_results['pca_components']}")
            print(f"K-Means Clusters: {feat_results['clusters_created']}")
            print(f"Features Selected: {feat_results['features_selected']}")
            print(f"Final Shape: {feat_results['final_shape']}")
        
        if 'woe_iv_analysis' in self.pipeline_results:
            woe_results = self.pipeline_results['woe_iv_analysis']
            print(f"\n--- WOE/IV Analysis Results ---")
            print(f"Features Analyzed: {woe_results['features_analyzed']}")
            print(f"Strong Features (IV >= 0.3): {woe_results['strong_features']}")
            print(f"Selected Features: {woe_results['selected_features']}")
            print(f"Final WOE Dataset: {woe_results['final_woe_shape']}")
        
        if 'modeling' in self.pipeline_results:
            model_results = self.pipeline_results['modeling']
            print(f"\n--- Model Performance Results ---")
            print(f"ROC AUC: {model_results['roc_auc']:.4f}")
            print(f"PR AUC (AUCPR): {model_results['pr_auc']:.4f}")
            print(f"Accuracy: {model_results['accuracy']:.4f}")
            print(f"F1-Score: {model_results['f1_score']:.4f}")
            print(f"CV ROC AUC (Mean): {model_results['cv_roc_auc_mean']:.4f}")
            print(f"CV PR AUC (Mean): {model_results['cv_pr_auc_mean']:.4f}")
        
        print(f"\n--- Files Generated ---")
        generated_files = [
            "cleaned_credit_data.csv",
            "engineered_features.csv", 
            "woe_transformed_data.csv",
            "iv_analysis_results.csv",
            "credit_risk_model.pkl",
            "correlation_matrix.png",
            "pca_variance_analysis.png",
            "elbow_method.png",
            "iv_analysis.png",
            "confusion_matrix.png",
            "roc_curve.png",
            "precision_recall_curve.png",
            "feature_importance.png",
            "calibration_plot.png"
        ]
        
        for file in generated_files:
            if os.path.exists(file):
                print(f"✓ {file}")
        
        print("\n" + "=" * 80)
        print("🎉 CREDIT RISK MODELING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        if 'modeling' in self.pipeline_results:
            final_aucpr = self.pipeline_results['modeling']['pr_auc']
            print(f"🎯 FINAL MODEL AUCPR: {final_aucpr:.4f}")
            
            if final_aucpr >= 0.97:
                print("🏆 EXCELLENT! Model achieved target AUCPR of 0.97+")
            elif final_aucpr >= 0.90:
                print("✅ GOOD! Model achieved strong performance")
            else:
                print("⚠️  Model performance could be improved")
        
        print("=" * 80)
    
    def run_individual_step(self, step_name):
        """
        Run individual pipeline step
        """
        if step_name.lower() == 'preprocessing':
            return preprocessing_main()
        elif step_name.lower() == 'feature_engineering':
            return feature_engineering_main()
        elif step_name.lower() == 'woe_iv':
            return woe_iv_main()
        elif step_name.lower() == 'modeling':
            return model_main()
        else:
            print(f"Unknown step: {step_name}")
            return None

def main():
    """
    Main function to run the complete pipeline
    """
    pipeline = CreditRiskPipeline()
    
    print("Credit Risk Modeling Pipeline")
    print("Choose execution mode:")
    print("1. Run complete pipeline")
    print("2. Run individual steps")
    print("3. Run specific components")
    
    try:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            # Run complete pipeline
            success = pipeline.run_complete_pipeline()
            if success:
                print("\n✅ Pipeline completed successfully!")
            else:
                print("\n❌ Pipeline failed!")
        
        elif choice == '2':
            # Run individual steps
            print("\nAvailable steps:")
            print("1. preprocessing")
            print("2. feature_engineering") 
            print("3. woe_iv")
            print("4. modeling")
            
            step = input("Enter step name: ").strip().lower()
            result = pipeline.run_individual_step(step)
            
        elif choice == '3':
            # Run specific components
            print("\nSelect components to run:")
            run_prep = input("Run preprocessing? (y/n): ").lower().startswith('y')
            run_feat = input("Run feature engineering? (y/n): ").lower().startswith('y')
            run_woe = input("Run WOE/IV analysis? (y/n): ").lower().startswith('y')
            run_model = input("Run modeling? (y/n): ").lower().startswith('y')
            
            success = pipeline.run_complete_pipeline(
                run_preprocessing=run_prep,
                run_feature_engineering=run_feat, 
                run_woe_iv=run_woe,
                run_modeling=run_model
            )
        
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
