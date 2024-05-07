import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# Load your dataset
file_path = 'E:/processed_cleveland.csv'  # Adjust path if necessary
data = pd.read_csv(file_path)

# Replace '?' with NaN
data = data.replace('?', np.nan)

# Attempt to convert all columns to float, this will force errors on non-convertible columns
for col in data.columns:
    try:
        data[col] = data[col].astype(float)
    except ValueError:
        print(f"Column {col} cannot be converted to float and may contain non-numeric values.")

# Now, fill NaN with the mean of each column (for numeric columns only)
data.fillna(data.mean(), inplace=True)

# Assuming the last column is the target
X = data.iloc[:, :-1].values  # All columns except the last as features
y = data.iloc[:, -1].values  # Last column as target variable

# Convert to binary classification (e.g., predicting class 1 vs all others)
y = (y == 1).astype(int)  # Change '1' to the class you are interested in

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predict probabilities for the test data
y_scores = model.predict_proba(X_test_scaled)[:, 1]  # probabilities for the positive class

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Logistic Regression')
plt.legend(loc="lower right")
plt.show()
