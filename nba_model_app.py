# Imports
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from knn_pca_model import train_knn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------------------------------------------------
nba = pd.read_csv("DATA/nba_data_with_salaries.csv")

# -------------------------------------------------------------------------------------------------------------------------------------
# Title
# -------------------------------------------------------------------------------------------------------------------------------------
st.title("NBA Dashboard")

# -------------------------------------------------------------------------------------------------------------------------------------
#Side Bar
# -------------------------------------------------------------------------------------------------------------------------------------
st.sidebar.title("NBA Dashboard")
logo = Image.open('Logo.png')
st.sidebar.image(logo, use_container_width=True)
tab = st.sidebar.radio("Navigation", ["README", "Data Table", "Exploratory Data Analysis", 'Models'])


# -------------------------------------------------------------------------------------------------------------------------------------
# README
# -------------------------------------------------------------------------------------------------------------------------------------
if tab == "README":
    st.title("**Machine Learning 1 Final Project**: NBA General Manager Trade Simulation")
    st.write("""
    **Members:** Eddie, Chase, Adam, Neel, Timothy, Jack, Harrison

### **OVERVIEW**:
Using all of the models we have used this semester, we will analyze NBA player data from the 2024-2025 seasons to answer various research questions. We clean and transform the data, explore it through descriptive statistics and visualizations, and build multiple predictive models depending on the prediction type. Finally, we deploy a Streamlit app to showcase our findings in an interactive way.

### **RESEARCH QUESTIONS & OBJECTIVES**:
1. Can we accurately predict player salary, all-star nominations, and other accomplishment features?
2. Assess which players are undervalued/overvalued in order to build a new team and predict transfers.
3. Can we predict the categorical variable of whether a player will be an all-star based on their season statistics?
4. KNN: Can we classify players into different archetypes based on their playing style and performance metrics?
5. Create a trade analysis model based on projected evaluated salaries + other evaluative metrics.
6. Predict win/loss for next season based on current roster and player statistics.


### **MODELS**:
1. Multiple Linear Regression (Add Polynomial?) -- Adam 
2. Logistic Regression -- Tim
3. KNN: K-Nearest Neighbors -- Chase
4. K-Means Clustering -- Eddie
    - Clustering players into different archetypes based on performance metrics
    - Clustering players based on their valuation to identify undervalued/overvalued players
5. PCA Model -- 
6. MLP Neural Network --6
7. Model Extension -- Harrison

### **PROJECT/FILE STRUCTURE**


### **INSTRUCTIONS FOR VIEWERS - HOW TO RUN**
Our x file does y
Our z file does a
...

### **DATA SOURCES**: 
1. *NBA API:* https://github.com/swar/nba_api
2. ESPN Salary Data -- Scraped from https://www.espn.com/nba/salaries
3. `2012-2023 NBA Stats.csv
             
### ** Viewing Data**
1. Run get_data.py
2. Run scrape_salaries.py
3. Run data_cleaning.py
    """)

# -------------------------------------------------------------------------------------------------------------------------------------
# Data Table
# -------------------------------------------------------------------------------------------------------------------------------------
if tab == "Data Table":
    st.subheader("NBA Data Table")

    # Rows per page
    rows_per_page = 25
    total_rows = nba.shape[0]
    total_pages = (total_rows - 1) // rows_per_page + 1

    # Initialize page number
    if "page_number" not in st.session_state:
        st.session_state.page_number = 0
    if "page_slider" not in st.session_state:
        st.session_state.page_slider = 1

    # -----------------------------------------------------------------
    # Pagination controls
    # -----------------------------------------------------------------
    st.write("")  # spacing
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("Previous"):
            st.session_state.page_number = (st.session_state.page_number - 1) % total_pages
            st.session_state.page_slider = st.session_state.page_number + 1

    with col3:
        if st.button("Next"):
            st.session_state.page_number = (st.session_state.page_number + 1) % total_pages
            st.session_state.page_slider = st.session_state.page_number + 1

    # Jump-to-page slider
    new_page = st.slider(
        "Jump to page",
        min_value=1,
        max_value=total_pages,
        step=1,
        key='page_slider'
    )
    st.session_state.page_number = new_page - 1 

    # Calculate start and end indices
    start_idx = st.session_state.page_number * rows_per_page
    end_idx = start_idx + rows_per_page
    df_page = nba.iloc[start_idx:end_idx]

    # Display the page (scrollable horizontally)
    st.dataframe(df_page, width=1500, height=400)

    # Page info
    st.write(f"Page {st.session_state.page_number + 1} of {total_pages} — Showing rows {start_idx + 1} to {min(end_idx, total_rows)} of {total_rows}")
    

# -------------------------------------------------------------------------------------------------------------------------------------
# Exploratory Data Analysis
# -------------------------------------------------------------------------------------------------------------------------------------
#if tab == "Exploratory Data Analysis":
    
# -------------------------------------------------------------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------------------------------------------------------------
if tab == "Models":
    st.title("Model Training & Prediction")
    st.subheader("Select Model")
    model_choice = st.radio("Choose a model:", 
                            ["Multiple Linear Regression", "Logistic Regression", "K-Means", "KNN", "PCA", "MLP Neural Network"])

# Multiple Linear Regression ----------------------------------------------------------------------------------------------------------
    if model_choice == "Multiple Linear Regression":
        st.write("Multiple Linear Regression")


        
# Logistic Regression ----------------------------------------------------------------------------------------------------------------
    if model_choice == "Logistic Regression":
        
        st.title("Logistic Regression: Predicting All-Star Status")

        players = nba.copy()

        # Create target variable: top 24 by PIE_RANK = All-Star
        players["all_star"] = (players["PIE_RANK"] <= 24).astype(int)

        st.write("An All-Star is any player ranked in the top 24 by PIE_RANK. Player Impact Estimate (PIE) measures a player's statistical contribution as a percent of all games played, combining positive actions and subtracting it from negative ones.")
        st.write(players["all_star"].value_counts())

        num_cols = players.select_dtypes("number").columns.tolist()

        # Columns we DO NOT want to include as features
        drop_cols = [
            "all_star",
            "PLAYER_ID",
            "TEAM_ID_base",
            "TEAM_ID_adv",
            "TEAM_COUNT_base",
            "TEAM_COUNT_adv",
            "PIE",
            "PIE_RANK",
        ]

        # Final feature set
        X_cols = [c for c in num_cols if c not in drop_cols]
        X = players[X_cols]
        y = players["all_star"]

        st.write(f"Number of numeric features used: {len(X_cols)}")

        # User can control these parameters
        test_size = st.slider(
            "Test Set Size",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )

        random_state = st.number_input(
            "Random State",
            min_value=0, max_value=999, value=42
        )

        if st.button("Train Logistic Regression Model"):

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                stratify=y,
                random_state=random_state
            )

            # Build model pipeline
            logit_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000
                ))
            ])

            logit_pipe.fit(X_train, y_train)

            # Predictions
            proba_test = logit_pipe.predict_proba(X_test)[:, 1]
            y_pred = (proba_test >= 0.5).astype(int)

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            TN, FP, FN, TP = cm.ravel()

            # Metrics
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2*precision*recall / (precision + recall) if (precision + recall) else 0
            ll = log_loss(y_test, proba_test)
            auc = roc_auc_score(y_test, proba_test)

            st.subheader("Model Performance")
            st.write("Accuracy:", round(accuracy, 3))
            st.write("Precision:", round(precision, 3))
            st.write("Recall:", round(recall, 3))
            st.write("F1 Score:", round(f1, 3))
            st.write("Log Loss:", round(ll, 3))
            st.write("ROC AUC:", round(auc, 3))

            st.subheader("Confusion Matrix")
            st.write(f"True Negatives: {TN}")
            st.write(f"False Positives: {FP}")
            st.write(f"False Negatives: {FN}")
            st.write(f"True Positives: {TP}")

            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Not All-Star", "All-Star"]
            )
            disp.plot(ax=ax)
            st.pyplot(fig)


# K-Means ----------------------------------------------------------------------------------------------------------------------------
    if model_choice == "K-Means":
        st.write("K-Means")
        
        # Import k_means_model functions
        import k_means_model as kmm
        
        # Load Data and Possible Features
        df = kmm.load_and_clean_data()
        attribute_options = sorted(df.columns.tolist())

        # User Input Section
        FEATURES = {
            'Player Archetype' : ['Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 'True Shooting Percentage', 
                                'Usage Percentage', 'Defensive Rating', 'Offensive Rating', 
                                '3 Pointer Attempts', '3 Pointer Made', 'Steals'],
            'Player Valuation' : ['Salary', 'Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 
                                'Steals', '3 Pointer Made'],
            'Custom Model' : [] # User input features (dropdown of all features)
        } # User will choose one of these options in the app
        MODEL_TYPE = st.selectbox("Select Model Type:", list(FEATURES.keys()))
        if MODEL_TYPE == 'Custom Model':
            selected_features = st.multiselect("Select Features for Custom Model:", attribute_options)
            FEATURES['Custom Model'] = selected_features
        K_VALUE = st.text_input("Enter K Value (or type 'Auto' for optimal K):", 'Auto')
        
        # Feature Selection
        X = df[FEATURES[MODEL_TYPE]]
        
        # Test Different K Values
        sil_scores, wcssm, opt_k = kmm.test_k_values(X)
        elbow_graph, sil_graph = kmm.graph_elbow_silhouette(wcssm, sil_scores)
        st.pyplot(elbow_graph)
        st.pyplot(sil_graph)
        
        # Build Final Model
        k_value = opt_k if K_VALUE == 'Auto' else int(K_VALUE)
        pipeline, labels = kmm.final_model_and_labels(X, k_value)
        df['Cluster'] = labels
        st.write(f'Final K-Means Model with K={k_value} Fitted!')
        
        # Show Styled DataFrame
        st.write("Cluster Summary:")
        summary = df.groupby(['Cluster'])[FEATURES[MODEL_TYPE]].mean().reset_index()
        summary.set_index('Cluster', inplace=True)
        styled = (
            summary.style
            .format("{:,.1f}")
            .background_gradient(cmap='Blues', axis=0)
        )
        st.dataframe(styled)
        
        # Interactive Plotting
        st.subheader("Cluster Visualization")
        options = summary.columns.tolist()
        X_AXIS = st.selectbox("Select X-Axis:", options)
        Y_AXIS = st.selectbox("Select Y-Axis:", options)
        COLOR = 'Cluster'
        ALPHA = st.slider("Scatterplot Transparency:", 0.0, 1.0, 0.8, 0.05)
        cluster_plot = kmm.graph_clusters(df, X_AXIS, Y_AXIS, COLOR, MODEL_TYPE, ALPHA)
        st.pyplot(cluster_plot)
        

# KNN --------------------------------------------------------------------------------------------------------------------------------
    if model_choice == "KNN":
        st.write("KNN")

        default_features = [
            'PTS', 'AST', 'REB', 'TS_PCT', 'USG_PCT',
            'DEF_RATING', 'OFF_RATING', 'FG3M', 'STL'
        ]

        # --- User controls ---
        selection = st.selectbox(
            "Feature Selection Method",
            ["default", "pca", "selectk"]
        )

        # Reset model metrics if user changes feature selection method
        if "prev_selection" not in st.session_state:
            st.session_state["prev_selection"] = selection

        if selection != st.session_state["prev_selection"]:
            st.session_state["knn_trained"] = False
            st.session_state["knn_results"] = None
            st.session_state["knn_model"] = None
            st.session_state["knn_selection"] = None

        st.session_state["prev_selection"] = selection

        test_size = st.slider(
            "Test Size (0.1 to 0.5)",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )

        tier_count = st.selectbox(
            "Number of Salary Tiers",
            [2, 3, 4, 5]
        )

        # Search mode
        search_mode = st.radio(
            "Training Mode",
            ["Grid Search", "Manual Parameters"]
        )

        use_grid_search = search_mode == "Grid Search"

        manual_params = None
        if not use_grid_search:
            st.markdown("### Manual Hyperparameters")

            n_neighbors = st.slider("K (neighbors)", 1, 50, 5)
            metric = st.selectbox("Distance Metric", ["euclidean", "manhattan"])
            weights = st.selectbox("Weights", ["uniform", "distance"])

            manual_params = {
                "knn__n_neighbors": n_neighbors,
                "knn__metric": metric,
                "knn__weights": weights
            }

        if st.button("Train KNN Model"):
            results = train_knn(
                nba=nba,
                selection=selection,
                test_size=test_size,
                tier_count=tier_count,
                use_grid_search=use_grid_search,
                manual_params=manual_params
            )

            st.session_state["knn_trained"] = True
            st.session_state["knn_results"] = results 
            st.session_state["knn_model"] = results["model"]
            st.session_state["knn_selection"] = selection

        if st.session_state.get("knn_trained", False):
            results = st.session_state["knn_results"]

            st.success("Training complete!")

            st.write("### Accuracy:", results["metrics"]["accuracy"])

            st.write("### Parameters used:", results["parameters"])

            st.write("### Classification Report")
            st.json(results["metrics"]["classification_report"])

            st.write("### Confusion Matrix")
            st.write(results["confusion_matrix"])
        
        if (
            "knn_model" in st.session_state
            and st.session_state["knn_selection"] == "default"
        ):
            st.markdown("---")
            st.header("Predict Salary Tier for a New Player")

            with st.form("prediction_form"):
                st.write("Enter player stats:")

                # Layout inputs in two columns for cleaner UI
                col1, col2 = st.columns(2)

                PTS = col1.number_input("Points (PTS)", 0.0, 50.0, 10.0)
                AST = col2.number_input("Assists (AST)", 0.0, 20.0, 5.0)
                REB = col1.number_input("Rebounds (REB)", 0.0, 20.0, 6.0)
                TS_PCT = col2.number_input("True Shooting % (TS_PCT)", 0.2, 0.8, 0.55)
                USG_PCT = col1.number_input("Usage % (USG_PCT)", 5.0, 40.0, 22.0)
                DEF_RATING = col2.number_input("Defensive Rating", 80.0, 130.0, 110.0)
                OFF_RATING = col1.number_input("Offensive Rating", 80.0, 140.0, 115.0)
                FG3M = col2.number_input("3-Pointers Made (FG3M)", 0.0, 10.0, 2.0)
                STL = col1.number_input("Steals (STL)", 0.0, 5.0, 1.0)

                submitted = st.form_submit_button("Predict Tier")

            if submitted:
                model = st.session_state["knn_model"]

                # Convert to DataFrame in correct order
                X_new = pd.DataFrame([{
                    "PTS": PTS,
                    "AST": AST,
                    "REB": REB,
                    "TS_PCT": TS_PCT,
                    "USG_PCT": USG_PCT,
                    "DEF_RATING": DEF_RATING,
                    "OFF_RATING": OFF_RATING,
                    "FG3M": FG3M,
                    "STL": STL
                }], columns=default_features)

                pred = model.predict(X_new)[0]

                st.success(f"Predicted Salary Tier: **{pred}**")

# PCA --------------------------------------------------------------------------------------------------------------------------------
    if model_choice == "PCA":
        st.write("PCA")

# MLP Neural Network -----------------------------------------------------------------------------------------------------------------
    if model_choice == "MLP Neural Network":
        st.write("MLP Neural Network")

# To run this dashboard, use the terminal command:
# streamlit run nba_model_app.py
# then the dashboard will automatically open in your web browser.