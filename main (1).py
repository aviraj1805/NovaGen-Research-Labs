# # Loading...


from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv("novagen_dataset.csv")

df

# # Data Preprocessing


df.isnull().sum()

df.info()

df.describe()

X = df.drop("Target",axis=1)
y = df["Target"]

X

# # Co-relation Heatmap


import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================================
# CORRELATION HEATMAP
# ============================================================================

# Create correlation matrix
correlation_matrix = X.corr()

# Create figure
plt.figure(figsize=(15, 13))

# Plot heatmap
sns.heatmap(correlation_matrix, 
            annot=True,           # Show correlation values
            cmap='coolwarm',       # Color scheme
            center=0,              # Center colormap at 0
            fmt='.2f',             # Format to 2 decimal places
            square=True,           # Make cells square
            linewidths=0.5,        # Add gridlines
            cbar_kws={'label': 'Correlation Coefficient'})

plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")

# # Train Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

X_train

X_train.shape

# # Base Model - LogisticRegression


# Base model - LR
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression(max_iter=2000)
base_model.fit(X_train, y_train)

y_pred_bm = base_model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

print("Accuracy: ", accuracy_score(y_test, y_pred_bm)*100,"%")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bm))

# # Base Model Accuracy - 82.25 %


# # Random Forest


# Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=12,
    oob_score=True,
    n_jobs=-1,
    ccp_alpha = 0.01
)

rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred_rf)*100,"%")

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_bm))

# # Random Forest Accuracy (Without prunning) - 84.13 %


# # Grid Search CV on Random Forest


# GridSearchCV
from sklearn.model_selection import GridSearchCV

params = {
    "n_estimators": [500, 1500],
    "max_depth": [8, 16],
    "min_samples_split": [2, 12, 24],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", "log2"],
    "min_impurity_decrease": [0.0, 0.05],
}

rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

grid = GridSearchCV(
    estimator=rf_clf,
    param_grid=params,
    scoring="accuracy",
    n_jobs=-1,
    cv=3, 
    verbose=2
)

grid.fit(X_train, y_train)

y_pred_grid = grid.predict(X_test)

print("Accuracy of GridSearchCV: ", accuracy_score(y_test, y_pred_grid))
print("best_params_ ", grid.best_params_)
print("best_score_", grid.best_score_)

# # Accuracy after Hyperparameter Tunning - 94.29 %


y_pred_train = grid.predict(X_train)

print("Accuracy of GridSearchCV on training: ", accuracy_score(y_train, y_pred_train))

# # Overfitting check - Training accuracy is 99.86 % and testing accuracy is 94.29 % 
# ## Which describes there is no drastical overfitting senario.


# # Gradient Boosting


# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    learning_rate=0.5,
    n_estimators=150,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42,
    verbose=1
)

gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)

print("Accuracy of Gradient Boosting: ", accuracy_score(y_test, y_pred_gb))

# # Gradient Boosting accuracy - 93.82 %


# # Stacking


# Stacking
from sklearn.ensemble import StackingClassifier

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]

meta_learner = LogisticRegression(random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5 
)

stacking_clf.fit(X_train, y_train)
y_pred = stacking_clf.predict(X_test)
score = stacking_clf.score(X_test, y_test)
print(f"Stacking Score: {score:.4f}")

# # Stacking Accuracy - 93.98 %


# # Best Fit Model: 


model = RandomForestClassifier(
    max_depth=16,
    max_features="sqrt",  
    min_impurity_decrease=0.0,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=1500,
    random_state=42,  
    n_jobs=-1 
)

model.fit(X_train, y_train)
score = model.score(X_test, y_test)

# # Final Accuracy - 94.29 %