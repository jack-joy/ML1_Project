# Imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    mean_squared_error,
    r2_score
)
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import seaborn as sns
from MLP import prep_data, train_model_mlp
from pca_app import get_nba_stats, scrape_salaries, merge_stats_salaries, run_pca

# -------------------------------------------------------------------------------------------------------------------------------------
# Data Prep for MLR
# -------------------------------------------------------------------------------------------------------------------------------------

# Helper function to handle multi-team players
def select_player_row(group):
    if len(group) == 1:
        return group
    else:
        multi_team_rows = group[group['Team'].isin(['2TM', '3TM', '4TM'])]
        if len(multi_team_rows) > 0:
            return multi_team_rows
        else:
            return group.head(1)

# Caching the data load so it doesn't re-scrape every time you click a button
@st.cache_data
def load_mlr_data():
    # 1. Scrape Per Game Stats
    url_pg = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html#per_game_stats"
    nba_pg = pd.read_html(url_pg, header=0)[0]
    nba_pg = nba_pg[nba_pg['Player'] != 'Player']
    nba_pg = nba_pg.groupby('Player').apply(select_player_row).reset_index(drop=True)
    
    # 2. Scrape Advanced Stats
    url_adv = "https://www.basketball-reference.com/leagues/NBA_2025_advanced.html#advanced"
    nba_value = pd.read_html(url_adv, header=0)[0]
    nba_value = nba_value[nba_value['Player'] != 'Player']
    nba_value = nba_value.groupby('Player').apply(select_player_row).reset_index(drop=True)
    
    # Clean Columns
    nba_pg = nba_pg.drop(columns=['Rk', 'Age', 'GS', 'MP', 'Team'], errors='ignore')
    nba_value = nba_value.drop(columns=['Rk', 'Age', 'G', 'GS', 'MP', 'Team', 'Pos', 'Awards'], errors='ignore')

    # 3. Load LEBRON Data (Update this path to your repo structure!)
    # Ideally, put this CSV in your 'DATA' folder
    try:
        nba_lebron = pd.read_csv('DATA/LEBRON Data - Sheet1.csv') 
    except FileNotFoundError:
        # Fallback for local testing if path varies
        nba_lebron = pd.read_csv('LEBRON Data - Sheet1.csv') 

    nba_lebron = nba_lebron.rename(columns={'Rank': 'LEBRON_Rank'})

    # 4. Merge
    nba_df = nba_pg.merge(nba_value, on='Player', how='inner')\
                .merge(nba_lebron, on='Player', how='inner')
    
    # Convert numeric columns that might be strings
    cols_to_numeric = ['3P%', '3PAr', 'eFG%', 'FT%', 'AST', 'TRB', 'STL', 'BLK', 'USG%', 'WS/48', 'VORP', 'Age', 'Minutes', 'LEBRON']
    for col in cols_to_numeric:
        if col in nba_df.columns:
            nba_df[col] = pd.to_numeric(nba_df[col], errors='coerce')
            
    return nba_df

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
    st.write(
        """
    **Members:** Eddie, Chase, Adam, Neel, Timothy, Jack, Harrison

### **OVERVIEW**:
Using all of the models we have used this semester, we will analyze NBA player data from the 2024-2025 seasons to answer various research questions. We will clean and transform the data, explore it through descriptive statistics and visualizations, and build multiple predictive models depending on the prediction type. Finally, we deploy a Streamlit app to showcase our findings in an interactive way.

### **RESEARCH QUESTIONS & OBJECTIVES**:
1. Can we accurately predict player salary, all-star nominations, and other accomplishment features?
2. Can we predict the categorical variable of whether a player will be an all-star based on their season statistics?
3. K-Means: Can we cluster players based on their performance metrics and valuation to get a sense of player archetypes and undervalued players? 
4. KNN: Can we classify players into different archetypes based on their playing style and performance metrics?
5. Create a trade analysis model based on projected evaluated salaries + other evaluative metrics.
6. Predict win/loss for next season based on current roster and player statistics.

### **MODELS**:
1. Multiple Linear Regression (Add Polynomial?)
2. Logistic Regression
3. KNN: K-Nearest Neighbors
4. K-Means Clustering
    - Clustering players into different archetypes based on performance metrics
    - Clustering players based on their valuation to identify undervalued/overvalued players
5. PCA Model
6. MLP Neural Network -- Trade Analysis

### **App Structure**:
1. Page 1: This ReadMe file.
2. Page 2: Interactive data table to view the data.
3. Page 3: Exploratory Data Analysis (EDA) to get a better understanding of the data.
4. Page 4: Statistical model to evaluate research questions.

### **INSTRUCTIONS FOR VIEWERS - HOW TO RUN**
1. Create Conda Environment: `conda env create -f environment.yml`
2. Activate conda environment: conda activate nba_ml_project
3. Run `scrape_salaries.py` and `get_clean_data.py`
4. Run App: `streamlit run nba_model_app.py` 

### **DATA SOURCES**: 
1. *NBA API:* https://github.com/swar/nba_api
2. ESPN Salary Data -- Scraped from https://www.espn.com/nba/salaries
3. `2012-2023 NBA Stats.csv

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
                            ["Multiple Linear Regression", "Logistic Regression", "K-Means", "KNN", "PCA", "MLP Trade Analysis"])

# Multiple Linear Regression ----------------------------------------------------------------------------------------------------------
    if model_choice == "Multiple Linear Regression":
        st.title("LEBRON Impact Predictor")
        st.divider()

        # --- LOAD DATA ---
        with st.spinner('Accessing Real-Time NBA Data...'):
            try:
                nba = load_mlr_data()
                
                # Define Features
                final_features = ['3P%', '3PAr', 'eFG%', 'FT%', 'AST', 'TRB', 'STL', 'BLK', 
                                  'USG%', 'WS/48', 'VORP', 'Age', 'Minutes']
                
                # Prepare Model Data
                model_data = nba[final_features + ['LEBRON']].dropna()
                X = model_data[final_features]
                y = model_data['LEBRON']
                
                # Split & Train
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                X_train_const = sm.add_constant(X_train)
                X_test_const = sm.add_constant(X_test)
                
                model = sm.OLS(y_train, X_train_const).fit()
                
                # Predictions
                y_pred = model.predict(X_test_const)
                
            except Exception as e:
                st.error(f"Error loading or processing data: {e}")
                st.stop()

        # --- TAB INTERFACE ---
        # Separating the view into distinct sections for clarity
        tab1, tab2 = st.tabs(["Model Overview", "Feature Analysis"])

        # TAB 1: EXECUTIVE SUMMARY & PERFORMANCE
        with tab1:
            st.subheader("Model Performance Executive Summary")
            
            # Metrics Row
            col1, col2, col3 = st.columns(3)
            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            
            col1.metric("Predictive Power (R²)", f"{test_r2:.3f}", 
                        help="How much of the LEBRON variance is explained by stats. Closer to 1.0 is better.")
            col2.metric("Error Margin (MSE)", f"{test_mse:.3f}",
                        help="Average squared difference between predicted and actual score.")
            col3.metric("Sample Size", f"{len(model_data)} Players",
                        help="Number of NBA players included in this analysis.")
            
            st.markdown("### Synopsis")
            st.info("""
            This model uses a **Multiple Linear Regression (MLR)** approach. It takes 13 key statistical inputs (like VORP, Shooting Splits, and Usage) 
            to predict a player's **LEBRON** score.
                    
            **LEBRON**: From the creators of the metric, "LEBRON evaluates a player’s contributions using the box score (weighted using boxPIPM’s weightings stabilized using Offensive Archetypes) and advanced on/off calculations (using Luck-Adjusted RAPM methodology) for a holistic evaluation of player impact per 100 possessions on-court."
            
            **Why this matters:** By isolating the relationship between these box-score stats and the LEBRON metric, we can objectively evaluate 
            if a player is underperforming or overperforming their 'expected' impact based on their raw production.
            """)
            
            # Scatter Plot: Actual vs Predicted
            st.markdown("#### Actual vs. Predicted LEBRON Scores")
            fig = go.Figure()
            player_names = X_test.index

            # Add scatter plot for actual vs predicted
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(size=8, opacity=0.6),
                text=player_names,
                hovertemplate='<b>%{text}</b><br>Actual: %{x:.2f}<br>Predicted: %{y:.2f}<br>Error: %{customdata:.2f}<extra></extra>',
                customdata=[abs(a - p) for a, p in zip(y_test, y_pred)]
            ))

            # Add diagonal line (perfect prediction line)
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))

            # Add red vertical lines showing the difference between actual and predicted
            for actual, predicted in zip(y_test, y_pred):
                fig.add_trace(go.Scatter(
                    x=[actual, actual],
                    y=[actual, predicted],
                    mode='lines',
                    line=dict(color='red', width=1),
                    showlegend=False,
                    hoverinfo='skip'
                ))

            fig.update_layout(
                xaxis_title='Actual LEBRON',
                yaxis_title='Predicted LEBRON',
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("View Detailed Statistical Summary (Raw Output)"):
                st.text(model.summary())

        # TAB 2: INTERPRETATION (COEFFICIENTS)
        with tab2:
            st.subheader("What Drives a High LEBRON Score?")
            st.markdown("The chart below shows the **weight** the model assigns to each stat. Bigger bars mean that stat is *more important* for a high LEBRON rating.")
            
            # Extract coefficients
            coef_df = pd.DataFrame({
                'Feature': final_features,
                'Weight': model.params[1:] # Exclude constant
            }).sort_values(by='Weight', ascending=False)
            
            # Color code positive vs negative impact
            coef_df['Impact'] = coef_df['Weight'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
            
            # Display readable bar chart
            st.bar_chart(coef_df.set_index('Feature')['Weight'], color="#FF4B4B")
            
            st.markdown("""
            **Key Takeaways for Coaching Staff:**
            * **Positive Bars:** Focus development here. Increasing these stats directly correlates with a higher LEBRON impact score.
            * **Negative Bars:** These stats might have diminishing returns or are negatively correlated with impact in this specific model structure.
            """)
        
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
        st.title("K-Means: Clustering Player Archetypes and Valuations")
        st.write('''
                 With the K-Means clustering below, we group players into distinct groups, 
                 identifying unique player archetypes, valuations, or custom groupings.
                 ''')
        
        # Import k_means_model functions
        import k_means_model as kmm
        
        # Load Data and Possible Features
        df = kmm.load_and_clean_data()
        attribute_options = sorted(df.columns.tolist())

        # User Input Section
        FEATURES = {
            'Player Archetype' : ['Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 'True Shooting Percentage', 
                                'Usage Percentage', 'Defensive Rating', 'Offensive Rating', 
                                '3 Pointer Attempts', '3 Pointer Made', 'Steals', 'Blocks'],
            'Player Valuation' : ['Salary', 'Points', 'Assists', 'Rebounds', 'Defensive Rebounds', 
                                'Steals', '3 Pointer Made', 'Blocks'],
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
        st.title("K-Nearest Neighbors: Predicting Player Salary")
        st.write('''
                 Using a player's per game performance metrics, can we accurately classify
                 players into different salary tiers in order to predict a salary range for new players
                 to be paid based on their peformance.

                 Below, the default features include a player's per game statistics for 
                 fitting a KNN model, but users also have the option of selecting 
                 the k best features from all columns in the dataset based on mutual information
                 before fitting a KNN model.
                 ''')

        default_features = [
            'MIN_base', 'FGM_base', 'FG3M', 'FTM', 
            'FG_PCT_base', 'FG3_PCT', 'FT_PCT', 'REB', 'AST', 
            'TOV', 'STL', 'BLK', 'PF', 'PTS', 'PLUS_MINUS', 'TD3'
        ]

        # --- User controls ---
        selection = st.selectbox(
            "Feature Selection Method",
            ["default", "selectk"]
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

            st.write("### Salary Tiers")
            bins = results["bins"]

            for i in range(len(bins) - 1):
                lower = bins[i]
                upper = bins[i + 1]

                st.write(
                    f"**Tier {i}:** \${lower:,.0f} → \${upper:,.0f}"
                )
            
        
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

                MIN_base = col1.number_input("Minutes (MIN)", 0.0, 48.0, 28.0)
                FGM_base = col2.number_input("Field Goals Made (FGM)", 0.0, 20.0, 5.0)
                FG3M = col1.number_input("3-Pointers Made (FG3M)", 0.0, 10.0, 2.0)
                FTM = col2.number_input("Free Throws Made (FTM)", 0.0, 15.0, 3.0)

                FG_PCT_base = col1.number_input("FG% (FG_PCT)", 0.0, 1.0, 0.45)
                FG3_PCT = col2.number_input("3P% (FG3_PCT)", 0.0, 1.0, 0.36)
                FT_PCT = col1.number_input("FT% (FT_PCT)", 0.0, 1.0, 0.80)

                REB = col2.number_input("Rebounds (REB)", 0.0, 20.0, 6.0)
                AST = col1.number_input("Assists (AST)", 0.0, 15.0, 4.0)
                TOV = col2.number_input("Turnovers (TOV)", 0.0, 10.0, 2.0)
                STL = col1.number_input("Steals (STL)", 0.0, 5.0, 1.0)
                BLK = col2.number_input("Blocks (BLK)", 0.0, 5.0, 1.0)
                PF = col1.number_input("Personal Fouls (PF)", 0.0, 6.0, 2.0)

                PTS = col2.number_input("Points (PTS)", 0.0, 60.0, 15.0)
                PLUS_MINUS = col1.number_input("Plus/Minus (PLUS_MINUS)", -20, 20, 0)
                TD3 = col2.number_input("Triple Doubles (TD3)", 0, 40, 0)

                submitted = st.form_submit_button("Predict Tier")

            if submitted:
                model = st.session_state["knn_model"]

                # Convert to DataFrame in correct order
                X_new = pd.DataFrame([{
                    "MIN_base": MIN_base,
                    "FGM_base": FGM_base,
                    "FG3M": FG3M,
                    "FTM": FTM,
                    "FG_PCT_base": FG_PCT_base,
                    "FG3_PCT": FG3_PCT,
                    "FT_PCT": FT_PCT,
                    "REB": REB,
                    "AST": AST,
                    "TOV": TOV,
                    "STL": STL,
                    "BLK": BLK,
                    "PF": PF,
                    "PTS": PTS,
                    "PLUS_MINUS": PLUS_MINUS,
                    "TD3": TD3
                }], columns=default_features)

                pred = model.predict(X_new)[0]

                st.success(f"Predicted Salary Tier: **{pred}**")

# PCA --------------------------------------------------------------------------------------------------------------------------------
    if model_choice == "PCA":
        st.write("PCA")

        if "df_merged" not in st.session_state:
            st.session_state.df_merged = None

        if st.button("Fetch & Process Data"):
            with st.spinner("Loading NBA base & advanced stats..."):
                stats = get_nba_stats()

            with st.spinner("Scraping salary data..."):
                salaries = scrape_salaries()

            with st.spinner("Merging datasets..."):
                st.session_state.df_merged = merge_stats_salaries(stats, salaries)

            st.success("Data successfully merged!")

        # Only show df_merged if it actually exists
        if st.session_state.df_merged is not None:
            df_merged = st.session_state.df_merged
            st.write(df_merged.head())

            with st.spinner("Running PCA..."):
                pca, pca_df = run_pca(df_merged)

            st.success("PCA completed!")

            st.write("### PCA Components (PC1–PC5)")
            st.dataframe(pca_df)

            st.write("### Explained Variance")
            st.bar_chart(pca.explained_variance_ratio_)

            st.download_button(
                "Download PCA CSV",
                pca_df.to_csv(index=False),
                "pca_components.csv",
                "text/csv"
            )


# MLP Neural Network -----------------------------------------------------------------------------------------------------------------
    if model_choice == "MLP Trade Analysis":
        #Load data and model
        df = prep_data("DATA/nba_data_with_salaries.csv")
        model, scaler, df = train_model_mlp(df)


        #Title
        st.title("MLP NBA Trade Analyzer")
        st.write("Choose players for Side A and Side B and compare trade value.")

        # Select Players
        all_players = nba["PLAYER_NAME"].tolist()

        col1, col2 = st.columns(2)

        with col1:
            team_a = st.multiselect("Select players for Team A", all_players)

        with col2:
            team_b = st.multiselect("Select players for Team B", all_players)

        # Calculate Trade Value
        A_out = df[df["PLAYER_NAME"].isin(team_a)]["final_score_pred"].sum()
        B_out = df[df["PLAYER_NAME"].isin(team_b)]["final_score_pred"].sum()

        A_in = B_out
        B_in = A_out

        A_net = A_in - A_out
        B_net = B_in - B_out

        #Trade Data Summary
        st.subheader("Trade Summary")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Team A")
            st.write(f"Outgoing Trade Value: **{A_out:.2f}**")
            st.write(f"Incoming Trade Value: **{A_in:.2f}**")
            st.write(f"Net Value: **{A_net:+.2f}**")

        with col2:
            st.markdown("### Team B")
            st.write(f"Outgoing Trade Value: **{B_out:.2f}**")
            st.write(f"Incoming Trade Value: **{B_in:.2f}**")
            st.write(f"Net Value: **{B_net:+.2f}**")

        # Trade Graph
        fig = go.Figure()

        fig.add_trace(go.Bar(
            name="Outgoing Trade Value",
            x=["Team A", "Team B"],
            y=[A_out, B_out],
            marker_color="red"
        ))

        fig.add_trace(go.Bar(
            name="Incoming Trade Value",
            x=["Team A", "Team B"],
            y=[A_in, B_in],
            marker_color="green"
        ))

        fig.update_layout(
            title="Trade Value Comparison",
            barmode="group",
            yaxis_title="Trade Value"
        )

        st.plotly_chart(fig)

        st.subheader("Player Details")

        #List Player Info
        def show_player_info(player):
            row = df[df["PLAYER_NAME"] == player].iloc[0]
            st.markdown(
                f"""
                **{player}**  
                - Trade Value: **{row['final_score_pred']:.2f}**  
                - PTS: {row['PTS']}  
                - REB: {row['REB']}  
                - AST: {row['AST']}  
                - TS%: {row['TS_PCT']:.3f}  
                - Salary: ${row['SALARY']:,.0f}  
                """
            )

        st.write("### Team A Players")
        for p in team_a:
            show_player_info(p)

        st.write("### Team B Players")
        for p in team_b:
            show_player_info(p)


# To run this dashboard, use the terminal command:
# streamlit run nba_model_app.py
# then the dashboard will automatically open in your web browser.
