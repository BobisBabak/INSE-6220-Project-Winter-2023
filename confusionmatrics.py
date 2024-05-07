import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load your dataset
file_path = 'E:/processed_cleveland.csv'  # Adjust path if necessary
data = pd.read_csv(file_path)

# Assuming the last column is the target and the first two are the features for visualization
X = data.iloc[:, :2].values  # Using only two features for visualization
y = data.iloc[:, -1].values  # Target variable

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
models = [
    ("Logistic Regression", LogisticRegression()),
    ("K-Nearest Neighbour (KNN)", KNeighborsClassifier(n_neighbors=5)),
    ("Quadratic Discriminant Analysis (QDA)", QuadraticDiscriminantAnalysis())
]

# Create subplots for the confusion matrices
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

for i, (name, model) in enumerate(models):
    # Train each model
    model.fit(X_train_scaled, y_train)
    # Predict on the test data
    y_pred = model.predict(X_test_scaled)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axs[i])
    axs[i].set_title(f'{name} Confusion Matrix')
    axs[i].set_xlabel('Predicted Class')
    axs[i].set_ylabel('True Class')

plt.tight_layout()
plt.show()
