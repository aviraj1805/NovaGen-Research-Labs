# HealthPredict Classification

## Project Description

HealthPredict is a machine learning classification system that predicts health outcomes based on comprehensive patient medical and lifestyle data. The project implements and compares multiple ensemble learning algorithms including Random Forest, Gradient Boosting, and Stacking to identify the most effective predictive model. The dataset contains 10,000 patient records with 22 features covering physiological metrics, lifestyle factors, and medical history.

## Dataset Overview

- **Total Samples**: 10,000 patient records
- **Features**: 22 attributes
- **Feature Categories**:
  - Physiological Metrics: Age, BMI, Blood Pressure, Cholesterol, Glucose Level, Heart Rate
  - Lifestyle Factors: Sleep Hours, Exercise Hours, Water Intake, Stress Level
  - Medical Information: Mental Health, Physical Activity, Medical History, Allergies
  - Demographic Data: Diet Type, Blood Group

## Model Performance

The following table presents the classification accuracy achieved by each model on the test dataset:

| Model | Algorithm | Accuracy | Parameters |
|-------|-----------|----------|-----------|
| Random Forest (Best) | Tree-based Ensemble | 94.29% | n_estimators=1500, max_depth=16, max_features='sqrt', min_samples_leaf=1 |
| Stacking | Meta-Learner Ensemble | 93.98% | RF + GB base learners, Logistic Regression meta-learner |
| Gradient Boosting | Boosting Ensemble | 93.82% | n_estimators=150, learning_rate=0.5, max_depth=5 |

## Model Specifications

### Best Performing Model: Random Forest

```python
RandomForestClassifier(
    n_estimators=1500,
    max_depth=16,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    min_impurity_decrease=0.0,
    random_state=42,
    n_jobs=-1
)
```

**Performance**: 94.29% test accuracy

### Gradient Boosting Classifier

```python
GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.5,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)
```

**Performance**: 93.82% test accuracy

### Stacking Classifier

Base Learners:
- Random Forest (n_estimators=100)
- Gradient Boosting (n_estimators=100)

Meta-Learner: Logistic Regression

Cross-Validation Folds: 5

**Performance**: 93.98% test accuracy

## Methodology

1. **Data Preprocessing**: Feature engineering and encoding of categorical variables
2. **Train-Test Split**: Standard 80-20 split for model evaluation
3. **Hyperparameter Tuning**: Grid search and randomized search for optimal parameters
4. **Model Evaluation**: Cross-validation (5-fold) with accuracy scoring
5. **Ensemble Methods**: Implementation of stacking to combine base learner predictions

## Key Findings

- Random Forest achieved the highest accuracy at 94.29%, demonstrating superior generalization
- Stacking improved upon individual base learner performance, achieving 93.98% accuracy
- Gradient Boosting performance (93.82%) indicates the potential of sequential boosting approaches
- The square root feature selection strategy provided optimal results in the Random Forest model

## Requirements

- Python 3.9+
- scikit-learn 1.0+
- pandas 1.3+
- numpy 1.21+

## Installation

```bash
pip install scikit-learn pandas numpy
```

## Usage

```python
from sklearn.ensemble import RandomForestClassifier

# Initialize and train the best model
model = RandomForestClassifier(
    n_estimators=1500,
    max_depth=16,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

## Model Comparison Summary

The project demonstrates that:

1. Random Forest with optimized hyperparameters provides the best predictive performance at 94.29%
2. Ensemble stacking methods show promise in combining complementary model strengths
3. Gradient Boosting offers competitive performance despite increased computational requirements
4. Careful hyperparameter tuning is essential for achieving optimal model performance

## Future Improvements

- Implement cross-validation grid search for meta-learner optimization in stacking
- Explore additional base learners (SVM, XGBoost) for stacking ensemble
- Conduct feature importance analysis to identify key predictive factors
- Implement class balance techniques if target variable is imbalanced
- Deploy model with REST API for production use

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or collaboration opportunities, please refer to the project documentation and code comments.
