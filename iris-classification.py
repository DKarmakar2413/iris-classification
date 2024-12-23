#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['species'] = data.target


# In[5]:


df.describe()


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df, hue='species')
plt.show()


# In[9]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import pandas as pd

# Assuming 'df' is your DataFrame, and 'species' is the target column

# Separate features and target
X = df.iloc[:, :-1]  # Feature columns
y = df['species']    # Target column

# Step 1: Feature Scaling
min_max_scaler = MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Step 2: Create Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled) 
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))

# Step 3: Feature Selection
# 3a. Remove low-variance features
var_thresh = VarianceThreshold(threshold=0.01)  # Features with variance < 0.01
X_high_variance = var_thresh.fit_transform(X_scaled)
selected_features = X.columns[var_thresh.get_support()]
X_high_variance = pd.DataFrame(X_high_variance, columns=selected_features)

# 3b. Mutual Information for feature importance
mi_scores = mutual_info_classif(X_scaled, y, discrete_features=False)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values(by='MI Score', ascending=False)

# Print mutual information scores
print("Mutual Information Scores:")
print(mi_df)

top_features = mi_df['Feature'].iloc[:3]  # Selecting top 3 features
X_selected = X_scaled[top_features]

# Final transformed DataFrame
print("Final DataFrame with selected features:")
print(X_selected.head())


# In[10]:


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Support Vector Machine": SVC(kernel='rbf')
}

# Train and evaluate models
best_model = None
best_accuracy = 0

print("Model Evaluation Results:\n")
for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")
    
    # Update the best model
    if accuracy > best_accuracy:
        best_model = model
        best_accuracy = accuracy
    
    # Display classification report
    print(f"Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Print the best model
print(f"\nBest Model: {best_model.__class__.__name__} with Accuracy: {best_accuracy:.4f}")


# In[11]:


import joblib

# Save the best model
model_filename = "best_model.pkl"
joblib.dump(best_model, model_filename)

print(f"Best model saved as '{model_filename}'.")


# In[12]:


# Load the saved model
loaded_model = joblib.load("best_model.pkl")

# Make predictions with the loaded model
sample_data = X_test.iloc[0:5]  # Replace with new data
predictions = loaded_model.predict(sample_data)

print("Predictions on sample data:", predictions)

