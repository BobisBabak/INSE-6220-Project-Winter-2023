import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt

# Load your data
file_path = 'E:\\processed_cleveland.csv'  # Adjust the file path
data = pd.read_csv(file_path)

# Handle missing values and convert data types
data = data.replace('?', np.nan)
data['ca'] = pd.to_numeric(data['ca'], errors='coerce')
data['thal'] = pd.to_numeric(data['thal'], errors='coerce')
data.dropna(inplace=True)

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

# Convert X_test back to DataFrame for better naming in plots
X_test_df = pd.DataFrame(X_test, columns=['Component 1', 'Component 2', 'Component 3'])

# Selecting the class index correctly based on your target
# This assumes you have binary classification (0 or 1)
class_index = 1  # Change this if your model is multiclass and you need a different class

# Choose an observation from the test set to visualize (for example, the first observation)
observation_index = 0  # Adjust as needed
shap.initjs()  # Initialize JavaScript visualization in Jupyter if applicable

# Debugging output to check the alignment of SHAP values and features
print(f"SHAP values for observation {observation_index}: {shap_values[class_index][observation_index]}")
print(f"Feature values: {X_test_df.iloc[observation_index].values}")

# Plot the force plot
try:
    force_plot = shap.force_plot(
        explainer.expected_value[class_index], 
        shap_values[class_index][observation_index], 
        X_test_df.iloc[observation_index], 
        feature_names=['Component 1', 'Component 2', 'Component 3'],
        matplotlib=True
    )
    plt.show(force_plot)
except Exception as e:
    print(f"Error in plotting: {e}")
