import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.datalib.manipulation_data import DataManipulation  
from src.datalib.statistics import Statistics
from src.datalib.advanced_analysis import MachineLearningModels  # Import MachineLearningModels class
from src.datalib.visualization import Plotting  # Import Plotting class

def main():
    # 1. Data Manipulation
    print("1. Data Manipulation Demonstration")
    
    # Create sample dataset
    data = {
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [30000, 45000, 50000, 60000, 70000, 80000, 85000, 90000],
        'education_years': [12, 14, 16, 16, 18, 18, 20, 20],
        'city': ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Berlin', 'Toronto', 'Madrid']
    }
    df = pd.DataFrame(data)
    
    # Filter data using DataManipulation class
    filtered_df = DataManipulation.filter_data(df, {'age': lambda x: x > 35})
    print("Filtered Data (Age > 35):")
    print(filtered_df)
    
    # Normalize data using DataManipulation class
    normalized_df = DataManipulation.normalize_data(df, ['age', 'income'])
    print("\nNormalized Data:")
    print(normalized_df)
    
    # 2. Statistical Analysis
    print("\n2. Statistical Analysis Demonstration")
    
    # Descriptive statistics (using individual methods)
    print("\nAge Statistics:")
    print("Mean:", Statistics.calculate_mean(df['age']))
    print("Median:", Statistics.calculate_median(df['age']))
    print("Mode:", Statistics.calculate_mode(df['age']))
    print("Standard Deviation:", Statistics.calculate_standard_deviation(df['age']))
    
    # Correlation analysis
    print("\nCorrelation Coefficient between Age and Income:")
    correlation_matrix = Statistics.correlation_coefficient(df['age'], df['income'])
    print(correlation_matrix)

    # 3. Machine Learning Models Demonstration
    print("\n3. Machine Learning Models Demonstration")

    # Linear Regression
    X = normalized_df[['age', 'income']]  # Use normalized features
    y = normalized_df['education_years']  # Target variable (e.g., education years)
    linear_regression_model = MachineLearningModels.linear_regression(X, y)
    print("\nLinear Regression Model Coefficients:")
    print(linear_regression_model.coef_)

    # KMeans Clustering
    kmeans_model = MachineLearningModels.kmeans_clustering(X, n_clusters=2)  # 2 clusters
    print("\nKMeans Clustering Centroids:")
    print(kmeans_model.cluster_centers_)

    # PCA Analysis
    pca, transformed_data = MachineLearningModels.pca_analysis(X, n_components=2)  # Reduce to 2 components
    print("\nPCA Transformed Data:")
    print(transformed_data)

    # Decision Tree Classification (e.g., predict 'education_years' based on 'age' and 'income')
    y_class = (df['education_years'] > 16).astype(int)  # Example classification: 1 if education_years > 16, else 0
    decision_tree_model = MachineLearningModels.decision_tree_classification(X, y_class)
    print("\nDecision Tree Classifier Feature Importances:")
    print(decision_tree_model.feature_importances_)

    # 4. Plotting Demonstration
    print("\n4. Plotting Demonstration")

    # Plot Histogram of 'age' column
    Plotting.plot_histogram(df['age'], bins=5, title="Age Distribution", xlabel="Age", ylabel="Frequency")

    # Plot Scatter Plot of 'age' vs 'income'
    Plotting.plot_scatter(df['age'], df['income'], title="Age vs Income", xlabel="Age", ylabel="Income")


if __name__ == "__main__":
    main()
