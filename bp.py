import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for handling CSV files
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Load your dataset
file_path = 'E:/processed_cleveland.csv'  # Adjust path if necessary
data = pd.read_csv(file_path)

# Assuming the last column is the target and the first two are the features
X = data.iloc[:, :2].values  # Using only two features for visualization
y = data.iloc[:, -1].values  # Target variable

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the models
models = [
    ("Logistic Regression", LogisticRegression()),
    ("K-Nearest Neighbour (KNN)", KNeighborsClassifier(n_neighbors=5)),
    ("Quadratic Discriminant Analysis (QDA)", QuadraticDiscriminantAnalysis())
]

# Setup plot details
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
h = .02  # step size in the mesh

# Create a mesh grid
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plotting each model's decision boundaries
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, (name, model) in enumerate(models):
    axs[i].set_xlim(xx.min(), xx.max())
    axs[i].set_ylim(yy.min(), yy.max())
    axs[i].set_xlabel('Feature 1')
    axs[i].set_ylabel('Feature 2')
    axs[i].set_title(name)

    # Train each model
    model.fit(X_scaled, y)
    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Put the result into a color plot
    axs[i].contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # Plot also the training points
    axs[i].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)

fig.suptitle('Decision Boundaries of Different Algorithms', fontsize=16)
plt.tight_layout(pad=3.0)
plt.show()
