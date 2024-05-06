import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
data_path = 'processed_cleveland.csv'
heart_data = pd.read_csv(data_path)

# Preprocessing
# Convert categorical data to numeric if necessary
heart_data['ca'] = pd.to_numeric(heart_data['ca'], errors='coerce')
heart_data['thal'] = pd.to_numeric(heart_data['thal'], errors='coerce')
heart_data.dropna(inplace=True)  # Drop rows with any NaN values

# Separate features and target
features = heart_data.drop('num', axis=1)
target = heart_data['num']

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply PCA
pca = PCA()
features_pca = pca.fit_transform(features_scaled)

# Explained variance and eigenvectors
eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# Plotting
# Scree Plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.grid(True)
plt.savefig('Scree_Plot.png')
plt.show()

# Cumulative variance Pareto Plot
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(eigenvalues) + 1), eigenvalues / np.sum(eigenvalues), alpha=0.6, color='blue')
plt.plot(range(1, len(eigenvalues) + 1), cumulative_variance, marker='o', color='black', linestyle='-')
plt.title('Pareto Plot of PCA Components')
plt.xlabel('Principal Components')
plt.ylabel('Percentage Explained Variance')
plt.axvline(x=3, color='red', linestyle='--')
plt.savefig('Pareto_Plot.png')
plt.show()

# PC Coefficient Plot
plt.figure(figsize=(8, 5))
for i, feature in enumerate(features.columns):
    plt.arrow(0, 0, eigenvectors[0, i], eigenvectors[1, i], color='black', alpha=0.5, head_width=0.05)
    plt.text(eigenvectors[0, i], eigenvectors[1, i], feature)
plt.title('PC Coefficient Plot')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.savefig('PC_Coefficient_Plot.png')
plt.show()

# Biplot
plt.figure(figsize=(12, 8))
scatter = plt.scatter(features_pca[:, 0], features_pca[:, 1], c=target, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Class')
for i, feature in enumerate(features.columns):
    plt.arrow(0, 0, pca.components_[0, i]*3, pca.components_[1, i]*3, color='black', alpha=0.5, head_width=0.05)
    plt.text(pca.components_[0, i]*3, pca.components_[1, i]*3, feature, fontsize=9)
plt.title('Biplot of PCA Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.savefig('Biplot.png')
plt.show()
