import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

#Import Data
pd.set_option('display.max_columns', None)
df = pd.read_csv("DATA/nba_data_with_salaries.csv")
df.head()

#Preparing the data
def prep_data(path="DATA/nba_data_with_salaries.csv"):
    df = pd.read_csv(path)
    df = df[df["MIN_base"] > 0].copy()

    #Convert to minutes per game to normalize playing time
    #Industry standard is 36 minutes per game in order to normalize
    #We do this so that the data doesnt skew towards players who play more minuters or less
    per36 = 36.0 / df["MIN_base"]
    df["PTS_per36"] = df["PTS"] * per36
    df["AST_per36"] = df["AST"] * per36
    df["REB_per36"] = df["REB"] * per36
    df["STL_per36"] = df["STL"] * per36
    df["BLK_per36"] = df["BLK"] * per36

    #Direct Impact Score
    df["impact_r"] = (
        0.5 * df["PTS_per36"] 
        + (0.7 * df["AST_per36"]) 
        + (0.7 * df["REB_per36"]) 
        + (1.0 * df["STL_per36"]) 
        + (1.0 * df["BLK_per36"]) 
        + (5.0 * df["TS_PCT"]) 
        - df["TS_PCT"].mean()
    )

    impact_mean = df["impact_r"].mean()
    impact_std = df["impact_r"].std()
    df["impact_z"] = (df["impact_r"] - impact_mean) / impact_std

    #Salary Score
    df["salary_m"] = df["SALARY"] / 1000000
    base_salary = df["salary_m"].median()
    impact_clip = df["impact_z"].clip(-2, 3)
    df["fair_salary_m"] = base_salary + 5 * (impact_clip + 2)
    df["salary_surplus_m"] = df["fair_salary_m"] - df["salary_m"]
    df["salary_surplus_norm"] = df["salary_surplus_m"] / 5

    #Age Score
    def age_score(age):
        if age <= 24:
            return 1
        elif age < 32:
            return 1 - 0.5 * (age - 24) / 8
        return max(0.0, 0.5 - 0.05 * (age - 32))
    
    df["age_score"] = df["AGE_base"].apply(age_score)

    #Frequency Score
    df["freq_score"] = np.minimum(1.0, df["GP_base"] / 75.0)

    #Final Score
    df["final"] = (
        1.3 * df["impact_z"] 
        + 1.0 * df["salary_surplus_norm"]
        + 0.5 * df["age_score"]
        + 0.3 * df["freq_score"]
    )

    final_mean = df["final"].mean()
    final_std = df["final"].std()
    df["final_score"] = (df["final"] - final_mean) / final_std

    return df

def train_model_mlp(df):
    #Features that we want apart of our model
    features = [
        "AGE_base", "GP_base", "salary_m", 
        "PTS_per36", "AST_per36", "REB_per36", 
        "STL_per36", "BLK_per36", "TS_PCT", 
        "FG_PCT_base", "FG3_PCT"]
    
    #Create X and y
    X = df[features].values
    y = df["final_score"].values

    #Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    #Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    #Create model
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )

    mlp.fit(X_train_scaled, y_train)
    df["final_score_pred"] = mlp.predict(scaler.transform(X))

    return mlp, scaler, df