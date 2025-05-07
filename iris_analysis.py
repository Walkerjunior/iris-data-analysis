# iris_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Set visual style
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load iris dataset from sklearn
    iris_raw = load_iris()
    df = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    df['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing values:\n", df.isnull().sum())

    # No missing values in Iris dataset; otherwise, we could clean like:
    # df.dropna(inplace=True) or df.fillna(method='ffill', inplace=True)

except Exception as e:
    print(f"Error loading dataset: {e}")

# Task 2: Basic Data Analysis
print("\nStatistical Summary:")
print(df.describe())

# Grouping: Mean petal length per species
grouped = df.groupby('species').mean(numeric_only=True)
print("\nAverage measurements by species:")
print(grouped)

# Task 3: Data Visualization

# 1. Line chart - Let's simulate a trend (e.g., petal length changes across index)
plt.figure(figsize=(8, 4))
for species in df['species'].unique():
    plt.plot(df[df['species'] == species].index, df[df['species'] == species]['petal length (cm)'], label=species)
plt.title('Petal Length Trend by Index')
plt.xlabel('Index')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart - Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'])
plt.title('Average Petal Length by Species')
plt.ylabel('Petal Length (cm)')
plt.xlabel('Species')
plt.tight_layout()
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot - Sepal length vs petal length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Findings
print("\nFindings:")
print("- Setosa has the shortest petals, Virginica the longest.")
print("- Sepal length and petal length have a strong positive correlation.")
print("- The petal length shows distinct patterns across species.")
