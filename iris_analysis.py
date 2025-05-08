# iris_analysis.py
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """Load and prepare the Iris dataset"""
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target_names[iris.target]
    
    # Rename columns for consistency (remove units and spaces)
    iris_df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    return iris_df

def explore_data(df):
    """Perform data exploration"""
    print("\n=== Data Overview ===")
    print(df.head())
    
    print("\n=== Dataset Information ===")
    print(df.info())
    
    print("\n=== Statistical Summary ===")
    print(df.describe())
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

def analyze_data(df):
    """Perform data analysis"""
    print("\n=== Species Distribution ===")
    print(df['species'].value_counts())
    
    print("\n=== Mean Measurements by Species ===")
    print(df.groupby('species').mean())

def create_visualizations(df):
    """Create and save visualizations"""
    # Create images directory if it doesn't exist
    import os
    os.makedirs('images', exist_ok=True)
    
    # 1. Line Chart
    plt.figure(figsize=(10, 5))
    plt.plot(df.index[:50], df['sepal_length'][:50], label='Sepal Length')
    plt.title('Sepal Length Trend (First 50 Samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Sepal Length (cm)')
    plt.legend()
    plt.savefig('images/sepal_trend.png')
    plt.show()
    
    # 2. Bar Chart
    plt.figure(figsize=(8, 5))
    sns.barplot(x='species', y='petal_length', data=df)
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.savefig('images/petal_by_species.png')
    plt.show()
    
    # 3. Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['sepal_width'], bins=20, kde=True)
    plt.title('Distribution of Sepal Width')
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.savefig('images/sepal_width_dist.png')
    plt.show()
    
    # 4. Scatter Plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal_length', y='petal_length', hue='species', data=df)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.savefig('images/sepal_vs_petal.png')
    plt.show()

def main():
    """Main execution function"""
    # Load and prepare data
    iris_df = load_and_prepare_data()
    
    # Data exploration
    explore_data(iris_df)
    
    # Data analysis
    analyze_data(iris_df)
    
    # Create visualizations
    create_visualizations(iris_df)
    
    # Print findings
    print("\n=== Key Findings ===")
    print("- Virginica has the largest petals (mean length: 5.55cm)")
    print("- Strong correlation between sepal and petal length visible in scatter plot")
    print("- Setosa has distinctly smaller petals compared to other species")
    print("- Sepal width shows approximately normal distribution")

if __name__ == "__main__":
    main()