"""
Credit Risk Modeling - Weight of Evidence (WOE) and Information Value (IV) Analysis
This module implements WOE and IV techniques for data binning and feature transformation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

class WOEIVAnalyzer:
    def __init__(self):
        self.woe_mappings = {}
        self.iv_scores = {}
        self.binning_info = {}
        
    def load_data(self, filepath='engineered_features.csv'):
        """
        Load the engineered features dataset
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Engineered features file not found. Please run feature_engineering.py first.")
            return None
    
    def calculate_woe_iv(self, df, feature, target, bins=10):
        """
        Calculate Weight of Evidence (WOE) and Information Value (IV) for a feature
        """
        # Create bins for continuous features
        if df[feature].dtype in ['int64', 'float64'] and df[feature].nunique() > bins:
            df_temp = df.copy()
            try:
                # Use quantile-based binning
                df_temp[f'{feature}_binned'] = pd.qcut(df_temp[feature], 
                                                     q=bins, 
                                                     duplicates='drop',
                                                     precision=3)
                feature_to_use = f'{feature}_binned'
            except ValueError:
                # If quantile binning fails, use equal-width binning
                df_temp[f'{feature}_binned'] = pd.cut(df_temp[feature], 
                                                    bins=bins, 
                                                    duplicates='drop',
                                                    precision=3)
                feature_to_use = f'{feature}_binned'
        else:
            df_temp = df.copy()
            feature_to_use = feature
        
        # Calculate WOE and IV
        total_good = df_temp[df_temp[target] == 0].shape[0]
        total_bad = df_temp[df_temp[target] == 1].shape[0]
        
        woe_iv_table = df_temp.groupby(feature_to_use).agg({
            target: ['count', 'sum']
        }).reset_index()
        
        woe_iv_table.columns = [feature_to_use, 'total', 'bad']
        woe_iv_table['good'] = woe_iv_table['total'] - woe_iv_table['bad']
        
        # Add small constant to avoid division by zero
        epsilon = 0.0001
        woe_iv_table['good'] = woe_iv_table['good'] + epsilon
        woe_iv_table['bad'] = woe_iv_table['bad'] + epsilon
        
        # Calculate distribution percentages
        woe_iv_table['good_rate'] = woe_iv_table['good'] / total_good
        woe_iv_table['bad_rate'] = woe_iv_table['bad'] / total_bad
        
        # Calculate WOE
        woe_iv_table['woe'] = np.log(woe_iv_table['good_rate'] / woe_iv_table['bad_rate'])
        
        # Calculate IV for each bin
        woe_iv_table['iv_component'] = (woe_iv_table['good_rate'] - woe_iv_table['bad_rate']) * woe_iv_table['woe']
        
        # Total IV for the feature
        iv_score = woe_iv_table['iv_component'].sum()
        
        # Create WOE mapping dictionary
        woe_mapping = dict(zip(woe_iv_table[feature_to_use], woe_iv_table['woe']))
        
        return woe_iv_table, woe_mapping, iv_score, feature_to_use
    
    def interpret_iv_score(self, iv_score):
        """
        Interpret Information Value score
        """
        if iv_score < 0.02:
            return "Not useful for prediction"
        elif iv_score < 0.1:
            return "Weak predictive power"
        elif iv_score < 0.3:
            return "Medium predictive power"
        elif iv_score < 0.5:
            return "Strong predictive power"
        else:
            return "Suspicious - too good to be true"
    
    def perform_woe_iv_analysis(self, df, target='default', bins=10):
        """
        Perform WOE and IV analysis for all features
        """
        # Get numerical features for analysis
        features_to_analyze = [col for col in df.columns 
                             if col not in ['customer_id', target] 
                             and df[col].dtype in ['int64', 'float64', 'object']]
        
        woe_iv_results = []
        
        print(f"Performing WOE/IV analysis for {len(features_to_analyze)} features...")
        
        for feature in features_to_analyze:
            try:
                woe_iv_table, woe_mapping, iv_score, binned_feature = self.calculate_woe_iv(
                    df, feature, target, bins)
                
                # Store results
                self.woe_mappings[feature] = woe_mapping
                self.iv_scores[feature] = iv_score
                self.binning_info[feature] = binned_feature
                
                # Add to results list
                woe_iv_results.append({
                    'feature': feature,
                    'iv_score': iv_score,
                    'interpretation': self.interpret_iv_score(iv_score),
                    'num_bins': len(woe_mapping)
                })
                
                print(f"Feature: {feature:<25} IV: {iv_score:.4f} ({self.interpret_iv_score(iv_score)})")
                
            except Exception as e:
                print(f"Error processing feature {feature}: {str(e)}")
        
        # Create results DataFrame and sort by IV score
        results_df = pd.DataFrame(woe_iv_results)
        results_df = results_df.sort_values('iv_score', ascending=False)
        
        return results_df
    
    def apply_woe_transformation(self, df, target='default'):
        """
        Apply WOE transformation to all features
        """
        df_woe = df.copy()
        
        for feature, woe_mapping in self.woe_mappings.items():
            try:
                if feature in df.columns:
                    # For continuous features that were binned
                    if feature in self.binning_info:
                        binned_feature = self.binning_info[feature]
                        if binned_feature != feature:
                            # Recreate bins and apply WOE mapping
                            if df[feature].dtype in ['int64', 'float64']:
                                try:
                                    binned_values = pd.qcut(df[feature], 
                                                          q=len(woe_mapping), 
                                                          duplicates='drop',
                                                          precision=3)
                                except ValueError:
                                    binned_values = pd.cut(df[feature], 
                                                         bins=len(woe_mapping), 
                                                         duplicates='drop',
                                                         precision=3)
                                
                                df_woe[f'{feature}_woe'] = binned_values.map(woe_mapping)
                            else:
                                df_woe[f'{feature}_woe'] = df[feature].map(woe_mapping)
                        else:
                            df_woe[f'{feature}_woe'] = df[feature].map(woe_mapping)
                    else:
                        df_woe[f'{feature}_woe'] = df[feature].map(woe_mapping)
                    
                    # Fill NaN values with 0 (neutral WOE)
                    df_woe[f'{feature}_woe'] = df_woe[f'{feature}_woe'].fillna(0)
                    
            except Exception as e:
                print(f"Error applying WOE transformation for {feature}: {str(e)}")
        
        print(f"\nApplied WOE transformation to {len(self.woe_mappings)} features")
        return df_woe
    
    def plot_woe_iv_analysis(self, results_df, top_n=15):
        """
        Create visualizations for WOE/IV analysis
        """
        # Plot top features by IV score
        plt.figure(figsize=(12, 8))
        
        top_features = results_df.head(top_n)
        
        # Create color map based on IV interpretation
        colors = []
        for interpretation in top_features['interpretation']:
            if 'Strong' in interpretation:
                colors.append('green')
            elif 'Medium' in interpretation:
                colors.append('orange')
            elif 'Weak' in interpretation:
                colors.append('yellow')
            elif 'Suspicious' in interpretation:
                colors.append('red')
            else:
                colors.append('gray')
        
        bars = plt.barh(range(len(top_features)), top_features['iv_score'], color=colors)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Information Value (IV)')
        plt.title(f'Top {top_n} Features by Information Value')
        plt.grid(axis='x', alpha=0.3)
        
        # Add IV score labels on bars
        for i, (bar, iv_score) in enumerate(zip(bars, top_features['iv_score'])):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{iv_score:.3f}', ha='left', va='center')
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='green', label='Strong (0.3-0.5)'),
            plt.Rectangle((0,0),1,1, color='orange', label='Medium (0.1-0.3)'),
            plt.Rectangle((0,0),1,1, color='yellow', label='Weak (0.02-0.1)'),
            plt.Rectangle((0,0),1,1, color='red', label='Suspicious (>0.5)'),
            plt.Rectangle((0,0),1,1, color='gray', label='Not Useful (<0.02)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        plt.savefig('iv_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_woe(self, df, feature, target='default', bins=10):
        """
        Plot WOE values for a specific feature
        """
        woe_iv_table, _, _, binned_feature = self.calculate_woe_iv(df, feature, target, bins)
        
        plt.figure(figsize=(12, 6))
        
        # Plot WOE values
        plt.subplot(1, 2, 1)
        bars = plt.bar(range(len(woe_iv_table)), woe_iv_table['woe'])
        plt.xlabel('Bins')
        plt.ylabel('Weight of Evidence (WOE)')
        plt.title(f'WOE Analysis for {feature}')
        plt.xticks(range(len(woe_iv_table)), 
                  [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) 
                   for x in woe_iv_table[binned_feature]], rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        # Color bars based on WOE values
        for i, bar in enumerate(bars):
            if woe_iv_table.iloc[i]['woe'] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Plot distribution
        plt.subplot(1, 2, 2)
        plt.bar(range(len(woe_iv_table)), woe_iv_table['total'], alpha=0.7, label='Total')
        plt.bar(range(len(woe_iv_table)), woe_iv_table['bad'], alpha=0.7, label='Bad')
        plt.xlabel('Bins')
        plt.ylabel('Count')
        plt.title(f'Distribution Analysis for {feature}')
        plt.xticks(range(len(woe_iv_table)), 
                  [str(x)[:15] + '...' if len(str(x)) > 15 else str(x) 
                   for x in woe_iv_table[binned_feature]], rotation=45)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'woe_analysis_{feature}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print WOE table
        print(f"\nWOE/IV Table for {feature}:")
        print(woe_iv_table.round(4))
    
    def select_features_by_iv(self, results_df, min_iv=0.1, max_features=20):
        """
        Select features based on IV scores
        """
        # Filter features by minimum IV score
        strong_features = results_df[results_df['iv_score'] >= min_iv].copy()
        
        # Remove suspicious features (IV > 0.5)
        strong_features = strong_features[strong_features['iv_score'] <= 0.5]
        
        # Select top features
        selected_features = strong_features.head(max_features)['feature'].tolist()
        
        print(f"\nSelected {len(selected_features)} features based on IV analysis:")
        print(f"Minimum IV threshold: {min_iv}")
        print(f"Maximum features: {max_features}")
        print("\nSelected features:")
        for i, feature in enumerate(selected_features, 1):
            iv_score = results_df[results_df['feature'] == feature]['iv_score'].iloc[0]
            interpretation = results_df[results_df['feature'] == feature]['interpretation'].iloc[0]
            print(f"{i:2d}. {feature:<25} IV: {iv_score:.4f} ({interpretation})")
        
        return selected_features
    
    def create_woe_dataset(self, df, selected_features, target='default'):
        """
        Create a dataset with WOE-transformed features
        """
        # Start with target and ID columns
        woe_columns = ['customer_id', target] if 'customer_id' in df.columns else [target]
        
        # Add WOE-transformed selected features
        for feature in selected_features:
            woe_feature_name = f'{feature}_woe'
            if woe_feature_name in df.columns:
                woe_columns.append(woe_feature_name)
        
        df_woe_final = df[woe_columns].copy()
        
        print(f"\nCreated WOE dataset with {len(woe_columns)-1} features (excluding target)")
        print(f"Dataset shape: {df_woe_final.shape}")
        
        return df_woe_final
    
    def save_woe_results(self, df_woe, results_df, filepath_data='woe_transformed_data.csv', 
                        filepath_results='iv_analysis_results.csv'):
        """
        Save WOE-transformed data and IV analysis results
        """
        df_woe.to_csv(filepath_data, index=False)
        results_df.to_csv(filepath_results, index=False)
        
        print(f"\nWOE-transformed data saved to: {filepath_data}")
        print(f"IV analysis results saved to: {filepath_results}")
        
        return df_woe, results_df

def main():
    """
    Main function to demonstrate WOE/IV analysis pipeline
    """
    print("=== Credit Risk Modeling - WOE/IV Analysis ===")
    
    # Initialize WOE/IV analyzer
    woe_iv_analyzer = WOEIVAnalyzer()
    
    # Load engineered features
    print("\n1. Loading engineered features...")
    df = woe_iv_analyzer.load_data()
    
    if df is None:
        return None
    
    # Perform WOE/IV analysis
    print("\n2. Performing WOE/IV analysis...")
    results_df = woe_iv_analyzer.perform_woe_iv_analysis(df)
    
    # Apply WOE transformation
    print("\n3. Applying WOE transformation...")
    df_woe = woe_iv_analyzer.apply_woe_transformation(df)
    
    # Plot IV analysis results
    print("\n4. Creating visualizations...")
    woe_iv_analyzer.plot_woe_iv_analysis(results_df)
    
    # Plot WOE analysis for top 3 features
    top_features = results_df.head(3)['feature'].tolist()
    print(f"\n5. Detailed WOE analysis for top features: {top_features}")
    for feature in top_features:
        if feature in df.columns:
            woe_iv_analyzer.plot_feature_woe(df, feature)
    
    # Select features based on IV scores
    print("\n6. Selecting features based on IV scores...")
    selected_features = woe_iv_analyzer.select_features_by_iv(results_df, min_iv=0.1, max_features=15)
    
    # Create final WOE dataset
    print("\n7. Creating final WOE dataset...")
    df_woe_final = woe_iv_analyzer.create_woe_dataset(df_woe, selected_features)
    
    # Save results
    print("\n8. Saving WOE analysis results...")
    df_woe_final, results_df = woe_iv_analyzer.save_woe_results(df_woe_final, results_df)
    
    print("\nWOE/IV analysis completed successfully!")
    print(f"Final WOE dataset shape: {df_woe_final.shape}")
    
    return df_woe_final, results_df

if __name__ == "__main__":
    woe_data, iv_results = main()