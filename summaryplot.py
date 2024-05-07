import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap

# Load your data
file_path = 'E:\\processed_cleveland.csv'  # Adjust the file path
data = pd.read_csv(file_path)

# Handle missing values and convert data types
data = data.replace('?', np.nan)
data.dropna(inplace=True)  # You might want to handle this more gracefully in a real scenario
data['ca'] = pd.to_numeric(data['ca'])
data['thal'] = pd.to_numeric(data['thal'])

# Define features and target
X = data.drop(columns=['num'])  # Assuming 'num' is the target
y = data['num']

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create SHAP Explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Generate SHAP summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=['Component 1', 'Component 2', 'Component 3'])
