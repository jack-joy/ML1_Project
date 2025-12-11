import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
import matplotlib.pyplot as plt

# ----------------------
# BUILD PIPELINE
# ----------------------
def build_knn_pipeline(selection):
    """Return pipeline and parameter grid based on selection type."""
    
    base_steps = [("scaler", StandardScaler())]
    param_grid = {}

    if selection == "selectk":
        base_steps.append(("select", SelectKBest(mutual_info_classif)))
        param_grid = {
            "select__k": [1, 5, 10, 20, 40, 60, 80, 100],
            "knn__n_neighbors": range(1, 41, 2),
        }

    elif selection == "default":
        param_grid = {
            "knn__n_neighbors": range(1, 41, 2)
        }

    # Append KNN classifier at the end
    base_steps.append(("knn", KNeighborsClassifier()))

    return Pipeline(base_steps), param_grid

# ======================================================
# TRAIN AND EVALUATE KNN MODEL
# ======================================================
def train_knn(
    nba: pd.DataFrame,
    selection: str,
    test_size: float,
    tier_count: int,
    use_grid_search: bool,
    manual_params: dict = None
):
    # Create quantile-based salary tiers (target variable)
    nba["salary_tier"], bins = pd.qcut(
        nba["SALARY"],
        q=tier_count,
        labels=range(tier_count),
        retbins=True
    )

    # Columns not usable with KNN (categorical / strings)
    drop_cols = [
        "PLAYER_ID", "PLAYER_NAME", "NICKNAME_base",
        "TEAM_ID_base", "TEAM_ABBREVIATION_base",
        "NICKNAME_adv", "TEAM_ID_adv", "TEAM_ABBREVIATION_adv"
    ]

    default_features = [
        'MIN_base', 'FGM_base', 'FG3M', 'FTM', 
        'FG_PCT_base', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 
        'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS', 'TD3'
    ]

    # Feature matrix X
    if selection == "default":
        X = nba[default_features]
    else:
        # Use all numeric features except dropped ones
        X = nba.drop(columns=drop_cols + ["SALARY", "salary_tier"])

    # Target Variable
    y = nba["salary_tier"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Build pipeline and parameter grid
    pipeline, param_grid = build_knn_pipeline(selection)

    # Option 1: GridSearchCV
    if use_grid_search:
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring="balanced_accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        parameters = grid.best_params_

    # Option 2: Manual user parameters
    else:
        if manual_params:
            pipeline.set_params(**manual_params)
            parameters = manual_params
        else:
            parameters = {}
        model = pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": model,
        "parameters": parameters,
        "metrics": metrics,
        "confusion_matrix": cm,
        "bins": bins,
    }
