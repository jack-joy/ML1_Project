import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import matplotlib.pyplot as plt

# Parameters for app
SELECTION = 'pca'

# ----------------------
# 1. Load and separate features/target
# ----------------------
nba = pd.read_csv("DATA/nba_data_with_salaries.csv")

# Drop categorical columns (KNN relies on distances)
drop_cols = [
    "PLAYER_ID", 
    "PLAYER_NAME",
    "NICKNAME_base",
    "TEAM_ID_base",
    "TEAM_ABBREVIATION_base",
    "NICKNAME_adv",
    "TEAM_ID_adv",
    "TEAM_ABBREVIATION_adv"
]

# Create a new variable of salary tiers to predict a player's salary
nba["salary_tier"], bins = pd.qcut(nba["SALARY"], q=4, labels=[0, 1, 2, 3], retbins=True)

y = nba["salary_tier"]
X = nba.drop(columns=drop_cols + ["SALARY", "salary_tier"])

# ----------------------
# 2. Train/validation split
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------
# 4. Build pipeline: scaling, PCA/select K, KNN
# ----------------------
if SELECTION == 'pca':
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA()),                      # PCA stage
        ("knn", KNeighborsClassifier())      # classifier
    ])
else:
    pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("select", SelectKBest(mutual_info_classif)),
    ("knn", KNeighborsClassifier())
])

# ----------------------
# 5. Grid search hyperparameters for PCA/Select K & KNN
# ----------------------
if SELECTION == 'pca':
    param_grid = {
        "pca__n_components": [5, 10, 20, 40, 60, 80],     # choose dimension
        "knn__n_neighbors": [1, 3, 5, 7, 11, 15],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }
else:
    param_grid = {
        "select__k": [20, 40, 60, 80],
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"]
    }

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# ----------------------
# 6. Evaluate Results
# ----------------------
print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
print("\nTEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# ----------------------
# 7. PCA Analysis and Plots
# ----------------------
if SELECTION == 'pca':
    pca = best_model.named_steps["pca"]

    print("Number of PCA components:", pca.n_components_)
    print("Explained variance ratio (first 10):")
    print(pca.explained_variance_ratio_[:10])
    print("Cumulative explained variance:", pca.explained_variance_ratio_.cumsum()[-1])

    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, pca.n_components_+1),
            pca.explained_variance_ratio_.cumsum(),
            marker="o")
    plt.xlabel("Number of PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True)
    plt.show()

y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Best KNN Model")
plt.show()