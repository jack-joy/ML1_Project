# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# %% [markdown]
# ## Goal: Can we predict whether a player is an All-Star?
# ## Approach: Logistic Model

# %%
players = pd.read_csv('DATA/nba_data_with_salaries.csv')
players.head()

# %%
# Initial Exploration of data to look at columns and summary statistics
#players.describe()
players.columns.tolist()


# %%
# players['PIE_RANK'].value_counts()
# players["PIE_RANK"].isnull().sum()
players['PIE_RANK'].describe()

# %%
# An All-Star is any player ranked in the top 24 by PIE_RANK. Player Impact Estimate (PIE) measures a player's statistical contribution as a percent of all games played, combining positive actions and subtracting it from negative ones.
players["all_star"] = (players["PIE_RANK"] <= 24).astype(int)
print("Class balance (all_star):")
print(players["all_star"].value_counts())
print(players["all_star"].value_counts(normalize=True))

# %%
# Identify numeric columns
num_cols = players.select_dtypes("number").columns.tolist()

# Identify ID columns because they are descriptions and need to be excluded
id_cols = [id_col for id_col in num_cols if "ID" in id_col.upper()]
id_cols


# %%
players[id_cols].head(20)


# %%
# these colums need to be excluded because aren't player statistics
players[["TEAM_COUNT_base", "TEAM_COUNT_adv"]].head(10)

# %%
drop_cols = [
    "all_star",        # target
    "PLAYER_ID",       # an ID
    "TEAM_ID_base",    # an ID
    "TEAM_ID_adv",     # an ID
    "TEAM_COUNT_base", # not a player stat
    "TEAM_COUNT_adv",  # not a player stat
    "PIE",             # underlying metric used for ranking
    "PIE_RANK",        # used to define target
]

# removes the inappropriate columns if they are numeric
X_cols = [c for c in num_cols if c not in drop_cols]

X = players[X_cols]
y = players["all_star"]

# Train–test split, stratify to prevent class inbalance
# Used a typical 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42
)

logit_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()), # logistic needs standardized features
    ("model", LogisticRegression(
        class_weight="balanced",  # handle class imbalance (~4% positives), so that there is more importance to rare positive examples
        solver="liblinear",
        max_iter=1000,
    ))
])

logit_pipe.fit(X_train, y_train) # fit the model are the training data

 
proba_test = logit_pipe.predict_proba(X_test)[:, 1] # Evaluate on the test set
y_pred = (proba_test >= 0.5).astype(int)
acc = round(accuracy_score(y_test, y_pred), 3)
ll = round(log_loss(y_test, proba_test), 3)
auc = round(roc_auc_score(y_test, proba_test), 3)

print("Accuracy:", acc)
print("Log loss:", ll)
print("ROC AUC:", auc)

cm = confusion_matrix(y_test, y_pred)

# Here is the displayed data
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not All-Star", "All-Star"]
)
disp.plot()
plt.show()



# %%
# accuracy = (TP + TN) / (TP + TN + FP + FN)
# precision = TP / (TP + FP)
# recall = TP / (TP + FN)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]

accuracy = (TP + TN) / (TP + TN + FP + FN) # Here are the metrics from above
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * (precision * recall) / (precision + recall)


print("Confusion Matrix Values")
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)
print("True Positives:", TP)

print("\nModel Performance Metrics")
print("Accuracy:", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall:", round(recall, 3))
print("F1 Score:", round(f1, 3))



