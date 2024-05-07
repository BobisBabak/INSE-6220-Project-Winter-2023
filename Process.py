# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
data_path = 'E:\\processed_cleveland.csv'
data = pd.read_csv(data_path)

# Convert 'ca' and 'thal' columns to numeric, coercing errors to NaN
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')

# Handling missing values: Replace NaNs with the mean of the column
data.fillna(data.mean(), inplace=True)

# Standardize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Data Matrix
print("Data Matrix:")
print(data_scaled[:5, :])  # Displaying first 5 rows for brevity

# Exploratory Data Analysis - Box and Whisker Plots
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(data_scaled, columns=data.columns))
plt.title('Box and Whisker Plot of Standardized Data')
plt.show()

# Correlation Matrix
correlation_matrix = np.corrcoef(data_scaled.T)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".1f", linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# PCA - Calculating Eigenvectors and Eigenvalues
pca = PCA()
principalComponents = pca.fit_transform(data_scaled)
explained_variance = pca.explained_variance_ratio_

# Scree plot
plt.figure()
plt.plot(np.cumsum(explained_variance))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Explained Variance')
plt.show()

# Biplot
def biplot(score, coeff, labels=None):
    xs = score[:,0]
    ys = score[:,1]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley)
    for i in range(len(coeff)):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color='r',alpha=0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color='g', ha='center', va='center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color='g', ha='center', va='center')
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

biplot(principalComponents[:, :2], np.transpose(pca.components_[:2, :]), list(data.columns))
plt.show()

# Scatter plot of first two principal components
plt.figure(figsize=(8, 6))
plt.scatter(principalComponents[:, 0], principalComponents[:, 1], edgecolors='k', c='orange')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.grid(True)
plt.show()
