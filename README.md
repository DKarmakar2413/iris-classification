
# Iris Flower Classification

## Objective
This project classifies Iris flowers into three species:
- **Setosa**
- **Versicolor**
- **Virginica**

The classification is based on their petal and sepal measurements.

---

## Project Steps

### 1. **Data Loading**
- The Iris dataset is loaded using `sklearn.datasets.load_iris`.

### 2. **Data Exploration**
- Descriptive statistics and dataset structure are analyzed.
- Visualizations:
  - Pairplots for feature relationships.
  - Correlation heatmaps for feature dependencies.

### 3. **Feature Engineering**
- **Scaling**: Applied Min-Max Scaling to normalize the dataset.
- **Polynomial Features**: Added non-linear feature interactions.
- **Feature Selection**: Used Mutual Information (MI) to select the most informative features.

### 4. **Model Selection**
The following models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

### 5. **Model Evaluation**
- Models were evaluated using:
  - Cross-validation accuracy.
  - Classification metrics: Precision, Recall, F1-Score.

### 6. **Model Deployment**
- The best-performing model was saved using `joblib`.
- The saved model can make predictions on new data.

---

## File Structure
```
├── iris_classification.py      # Main Python script for the project
├── best_model.pkl              # Saved model after training
├── README.md                   # Project documentation
```

---

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- joblib

To install all dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Results

- **Best Model**: The model with the highest accuracy during testing.
- **Performance Metrics**:
  - Accuracy: ~96-100% depending on the model.
  - Detailed classification reports for each model.

---

## Future Enhancements
- Explore advanced models like Gradient Boosting or Neural Networks.
- Deploy the model as a web application using Flask/Django.
- Integrate with an API for real-time classification.

---

## Author
- **Diptesh Karmakar**  
  Machine Learning Enthusiast  
  dipteshkarmakar007@gmail.com

---
