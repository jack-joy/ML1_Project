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

# ======================================================
# 1. USER-CONFIGURABLE PARAMETERS (for dashboard)
# ======================================================

# Feature-reduction choice: 'default', 'pca', or 'selectk'
SELECTION = 'default'

DEFAULT_FEATURES = [
    'PTS', 'AST', 'REB', 'TS_PCT', 'USG_PCT',
    'DEF_RATING', 'OFF_RATING', 'FG3M', 'STL'
]

TEST_SIZE = 0.2
TIER_COUNT = 4

# ----------------------
# 2. LOAD DATA & PREPARE FEATURES/TARGET
# ----------------------

nba = pd.read_csv("DATA/nba_data_with_salaries.csv")

# Columns not usable with KNN (categorical / strings)
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

# Create quantile-based salary tiers (target variable)
nba["salary_tier"], bins = pd.qcut(
    nba["SALARY"],
    q=TIER_COUNT,
    labels=range(TIER_COUNT),
    retbins=True
)

# Feature matrix X
if SELECTION == "default":
    X = nba[DEFAULT_FEATURES]
else:
    # Use all numeric features except dropped ones
    X = nba.drop(columns=drop_cols + ["SALARY", "salary_tier"])

y = nba["salary_tier"]

# ----------------------
# 3. TRAIN / TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

# ----------------------
# 4. BUILD PIPELINE
# ----------------------
def build_pipeline(selection):
    """Return pipeline and parameter grid based on selection type."""
    
    base_steps = [("scaler", StandardScaler())]

    # ----- Feature Selection Choice -----
    if selection == "pca":
        base_steps.append(("pca", PCA()))
        param_grid = {
            "pca__n_components": [5, 10, 20, 40, 60, 80],
            "knn__n_neighbors": [1, 3, 5, 7, 11, 15],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }

    elif selection == "selectk":
        base_steps.append(("select", SelectKBest(mutual_info_classif)))
        param_grid = {
            "select__k": [1, 5, 10, 20, 40],
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }

    elif selection == "default":
        param_grid = {
            "knn__n_neighbors": range(1, 41, 2)
        }

    # Append KNN classifier at the end
    base_steps.append(("knn", KNeighborsClassifier()))

    return Pipeline(base_steps), param_grid


pipeline, param_grid = build_pipeline(SELECTION)

# ----------------------
# 5. GRID SEARCH
# ----------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best parameters:", grid.best_params_)
print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

# ======================================================
# 6. FINAL EVALUATION
# ======================================================

y_pred = best_model.predict(X_test)

print("\nTEST ACCURACY:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Best KNN Model")
plt.show()

# ======================================================
# 7. OPTIONAL: PCA DIAGNOSTICS
# ======================================================

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
