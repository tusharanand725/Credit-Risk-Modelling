"""
Credit Risk Modeling - Feature Engineering
This module handles PCA, correlation analysis, and K-Means clustering for feature engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans_models = {}
        self.selected_features = []
        
    def load_cleaned_data(self, filepath='cleaned_credit_data.csv'):
        """
        Load the cleaned dataset
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Loaded cleaned data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Cleaned data file not found. Please run data_preprocessing.py first.")
            return None
    
    def correlation_analysis(self, df, threshold=0.8):
        """
        Perform correlation analysis to identify highly correlated features
        """
        # Select only numerical features for correlation analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['customer_id', 'default']]
        
        # Calculate correlation matrix
        correlation_matrix = df[numerical_cols].corr()
        
        # Visualize correlation matrix
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        print(f"\nHighly correlated feature pairs (|correlation| > {threshold}):")
        for feat1, feat2, corr in high_corr_pairs:
            print(f"{feat1} - {feat2}: {corr:.3f}")
        
        return correlation_matrix, high_corr_pairs
    
    def remove_highly_correlated_features(self, df, high_corr_pairs, correlation_matrix):
        """
        Remove one feature from each highly correlated pair
        """
        features_to_remove = set()
        
        for feat1, feat2, corr in high_corr_pairs:
            if feat1 not in features_to_remove and feat2 not in features_to_remove:
                # Keep the feature that has higher correlation with target
                if 'default' in df.columns:
                    corr_with_target1 = abs(df[feat1].corr(df['default']))
                    corr_with_target2 = abs(df[feat2].corr(df['default']))
                    
                    if corr_with_target1 > corr_with_target2:
                        features_to_remove.add(feat2)
                    else:
                        features_to_remove.add(feat1)
                else:
                    # If no target, remove the second feature by default
                    features_to_remove.add(feat2)
        
        print(f"\nFeatures to remove due to high correlation: {list(features_to_remove)}")
        df_reduced = df.drop(columns=list(features_to_remove))
        
        return df_reduced, list(features_to_remove)
    
    def perform_pca(self, df, n_components=None, variance_threshold=0.95):
        """
        Perform Principal Component Analysis for dimensionality reduction
        """
        # Select numerical features for PCA
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols 
                         if col not in ['customer_id', 'default'] and not col.endswith('_encoded')]
        
        X_numerical = df[numerical_cols]
        
        # Standardize the features
        X_scaled = self.scaler.fit_transform(X_numerical)
        
        # Determine number of components if not specified
        if n_components is None:
            # First, run PCA with all components to see explained variance
            pca_full = PCA()
            pca_full.fit(X_scaled)
            
            # Find number of components needed for specified variance threshold
            cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
            print(f"Number of components for {variance_threshold*100}% variance: {n_components}")
        
        # Perform PCA with determined number of components
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Create DataFrame with PCA components
        pca_columns = [f'PC{i+1}' for i in range(n_components)]
        df_pca = pd.DataFrame(X_pca, columns=pca_columns, index=df.index)
        
        # Visualize explained variance
        self.plot_pca_variance()
        
        # Print PCA results
        print(f"\nPCA Results:")
        print(f"Original features: {len(numerical_cols)}")
        print(f"PCA components: {n_components}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_}")
        print(f"Cumulative explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        return df_pca, numerical_cols
    
    def plot_pca_variance(self):
        """
        Plot PCA explained variance
        """
        plt.figure(figsize=(12, 5))
        
        # Explained variance by component
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Principal Component')
        
        # Cumulative explained variance
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                np.cumsum(self.pca.explained_variance_ratio_), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def kmeans_clustering(self, df, features_for_clustering=None, n_clusters=5):
        """
        Perform K-Means clustering to create cluster-based features
        """
        if features_for_clustering is None:
            # Use numerical features for clustering
            features_for_clustering = df.select_dtypes(include=[np.number]).columns.tolist()
            features_for_clustering = [col for col in features_for_clustering 
                                     if col not in ['customer_id', 'default'] and not col.endswith('_encoded')]
        
        X_cluster = df[features_for_clustering]
        
        # Standardize features for clustering
        X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
        
        # Determine optimal number of clusters using elbow method
        optimal_k = self.find_optimal_clusters(X_cluster_scaled, max_k=10)
        
        # Perform clustering with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_cluster_scaled)
        
        # Store the model
        self.kmeans_models['main'] = kmeans
        
        # Add cluster labels to dataframe
        df[f'cluster_{optimal_k}'] = cluster_labels
        
        # Create cluster-based features
        df = self.create_cluster_features(df, cluster_labels, features_for_clustering)
        
        print(f"\nK-Means Clustering Results:")
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Features used for clustering: {features_for_clustering}")
        print(f"Cluster distribution:")
        print(df[f'cluster_{optimal_k}'].value_counts().sort_index())
        
        return df, optimal_k
    
    def find_optimal_clusters(self, X, max_k=10):
        """
        Find optimal number of clusters using elbow method
        """
        inertias = []
        K_range = range(1, max_k + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow curve
        plt.figure(figsize=(10, 6))
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True)
        plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Simple elbow detection (find the point where the rate of decrease slows down)
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        
        # Find the point where second difference is maximum (elbow point)
        optimal_k = np.argmax(second_differences) + 2  # +2 because of double differencing
        
        return min(optimal_k, max_k)
    
    def create_cluster_features(self, df, cluster_labels, features_used):
        """
        Create features based on cluster analysis
        """
        # Distance to cluster center
        for i, feature in enumerate(features_used):
            cluster_means = df.groupby(cluster_labels)[feature].mean()
            df[f'{feature}_cluster_distance'] = abs(df[feature] - df[cluster_labels].map(cluster_means))
        
        # Cluster size (number of points in each cluster)
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        df['cluster_size'] = pd.Series(cluster_labels).map(cluster_sizes)
        
        print(f"Created cluster-based features for {len(features_used)} original features")
        
        return df
    
    def feature_selection(self, df, target_col='default', k=20):
        """
        Select top k features using statistical tests
        """
        # Separate features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['customer_id', target_col] 
                       and df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Use SelectKBest with f_classif for feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) 
                           if selector.get_support()[i]]
        
        # Get feature scores
        feature_scores = selector.scores_
        feature_ranking = sorted(zip(feature_cols, feature_scores), 
                               key=lambda x: x[1], reverse=True)
        
        print(f"\nFeature Selection Results:")
        print(f"Original features: {len(feature_cols)}")
        print(f"Selected features: {len(selected_features)}")
        print(f"\nTop {k} features:")
        for i, (feature, score) in enumerate(feature_ranking[:k]):
            print(f"{i+1:2d}. {feature:<25} Score: {score:.3f}")
        
        self.selected_features = selected_features
        
        return df[selected_features + [target_col]], selected_features
    
    def create_feature_interaction(self, df):
        """
        Create interaction features between important features
        """
        # Create some interaction features based on domain knowledge
        interactions = []
        
        if 'credit_score' in df.columns and 'debt_to_income_ratio' in df.columns:
            df['credit_score_debt_ratio'] = df['credit_score'] * (1 - df['debt_to_income_ratio'])
            interactions.append('credit_score_debt_ratio')
        
        if 'income' in df.columns and 'loan_amount' in df.columns:
            df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
            interactions.append('loan_to_income_ratio')
        
        if 'age' in df.columns and 'employment_years' in df.columns:
            df['employment_stability'] = df['employment_years'] / (df['age'] + 1)
            interactions.append('employment_stability')
        
        if 'credit_utilization' in df.columns and 'num_credit_accounts' in df.columns:
            df['avg_utilization_per_account'] = df['credit_utilization'] / (df['num_credit_accounts'] + 1)
            interactions.append('avg_utilization_per_account')
        
        print(f"\nCreated interaction features: {interactions}")
        
        return df
    
    def save_engineered_features(self, df, filepath='engineered_features.csv'):
        """
        Save the dataset with engineered features
        """
        df.to_csv(filepath, index=False)
        print(f"\nEngineered features saved to {filepath}")
        return df

def main():
    """
    Main function to demonstrate feature engineering pipeline
    """
    print("=== Credit Risk Modeling - Feature Engineering ===")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Load cleaned data
    print("\n1. Loading cleaned data...")
    df = feature_engineer.load_cleaned_data()
    
    if df is None:
        return None
    
    # Correlation analysis
    print("\n2. Performing correlation analysis...")
    correlation_matrix, high_corr_pairs = feature_engineer.correlation_analysis(df)
    
    # Remove highly correlated features
    print("\n3. Removing highly correlated features...")
    df, removed_features = feature_engineer.remove_highly_correlated_features(
        df, high_corr_pairs, correlation_matrix)
    
    # PCA for dimensionality reduction
    print("\n4. Performing PCA...")
    df_pca, original_numerical_features = feature_engineer.perform_pca(df, variance_threshold=0.95)
    
    # Combine PCA features with original categorical features
    categorical_cols = [col for col in df.columns if col.endswith('_encoded')]
    other_cols = ['customer_id', 'default'] + categorical_cols
    df_combined = pd.concat([df[other_cols], df_pca], axis=1)
    
    # K-Means clustering
    print("\n5. Performing K-Means clustering...")
    df_combined, optimal_k = feature_engineer.kmeans_clustering(df_combined)
    
    # Create interaction features
    print("\n6. Creating interaction features...")
    df_combined = feature_engineer.create_feature_interaction(df_combined)
    
    # Feature selection
    print("\n7. Performing feature selection...")
    df_final, selected_features = feature_engineer.feature_selection(df_combined, k=15)
    
    # Save engineered features
    print("\n8. Saving engineered features...")
    df_final = feature_engineer.save_engineered_features(df_final)
    
    print("\nFeature engineering completed successfully!")
    print(f"Final dataset shape: {df_final.shape}")
    
    return df_final

if __name__ == "__main__":
    engineered_data = main()
