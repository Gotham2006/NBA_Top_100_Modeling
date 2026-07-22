import streamlit as st
import pandas as pd

# Set up the page layout and title
st.set_page_config(page_title="NBA Advanced Analytics Project", layout="wide")

st.title("🏀 NBA Player Rankings: What statistics are Media Rankings based on? How much of it is narrative?")

# Create the main navigation tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Introduction", "Classical Rankings", "Modern Rankings", "NLP Model", "Conclusion"])

# ==========================================
# 📖 TAB 1: INTRODUCTION
# ==========================================
with tab1:
    st.header("Project Motivation")
    st.write("""
    The goal of this project is to decode how media members judge NBA players and compare those 
    narratives against statistics. We aim to find which statistics hold the most weight 
    in historical rankings and how modern analytics differ from traditional eye-test consensus.
    """)
    
    st.divider()
    
    st.header("Data Sources & Official Rankings")
    st.write("""
    Our models are trained on curated consensus rankings and official statistical databases. 
    You can explore the source material here:
    """)
    
    # Using columns to create a clean, organized list of source links
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Statistical Source")
        st.link_button("Basketball-Reference (Stats Hub)", "https://www.basketball-reference.com/")
        st.write("The primary source for all raw career metrics, PER, and advanced stats.")
        
    with col2:
        st.subheader("Consensus Ranking Sources")
        st.link_button("Bleacher Report Top 100", "https://bleacherreport.com/articles/25223594-brs-top-100-nba-players-all-time-ranked")
        st.link_button("The Basketball 100", "https://www.reddit.com/r/NBATalk/comments/1h2t2xj/john_hollingerdavid_aldridges_top_100_players_of/")
        st.link_button("ESPN Top 100", "https://www.espn.com/nba/story/_/page/nbarankalltime/greatest-players-ever")
        st.write("Used to aggregate the 'Consensus Rank' labels in our training data.")

    st.divider()
    
    st.header("Ranking Methodologies")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Classical Rankings")
        st.write("These models are trained on traditional counting stats (PPG, Rebounds, All-Star appearances, Championships) to reflect how players were historically judged.")
    with col2:
        st.subheader("Modern Rankings")
        st.write("These models incorporate advanced efficiency and rate metrics (PER, BPM, WS/48, True Shooting) to provide a more well rounded analysis.")

# ==========================================
# 🏛️ TAB 2: CLASSICAL RANKINGS
# ==========================================
with tab2:
    st.header("Classical Rankings: Legacy & Longevity")
    
    st.subheader("Features Used")
    st.markdown("""
    To mimic historical media consensus, our Classical models are trained on raw counting stats and 
    accumulated accolades. The feature set includes:
    * **Career_PPG**: Total scoring volume.
    * **Championships**: Team success.
    * **All_NBA_Teams**: Expert recognition of peak performance.
    * **Seasons_Played**: Longevity and career durability.
    * **All_Defensive_Teams**: Recognition of defensive impact.
    """)
    
    class_tab1, class_tab2 = st.tabs(["Random Forest Model", "Gradient Boosted Model"])
    
    with class_tab1:
        st.subheader("How Random Forest Works")
        st.write("""
        The Random Forest model functions as an 'ensemble' of decision trees. Instead of relying on one 
        analytical path, it grows hundreds of individual decision trees—each looking at different subsets 
        of player data—and asks them to 'vote' on a player's rank. By averaging these results, the model 
        becomes highly robust to outliers and noise in the data.
        """)
        
        st.subheader("Model Findings & Longevity Bias")
        st.write("""
        **What the Model Found:** The most striking takeaway is the massive weight (82.21%) placed on 
        'All_NBA_Teams'. The model has effectively learned that expert consensus is fundamentally built on 
        end-of-season accolades.
        
        **The Longevity Factor:** Unlike modern advanced metrics (which normalize for games played), 
        the Random Forest model explicitly rewards 'Seasons_Played'. We see a 'Variety Effect' where players 
        with long, consistent careers are ranked higher than they might be in a purely 'peak-performance' 
        advanced metric model. This mirrors how media analysts often grade players—by looking at the 
        total body of work rather than just per-possession efficiency.
        """)
        
        # Feature Importance display
        st.markdown("**Feature Weights:**")
        data_weights = {
            "Metric": ["All_NBA_Teams", "Championships", "Career_PPG", "Seasons_Played"],
            "Weight": ["82.21%", "7.96%", "5.07%", "3.91%"]
        }
        st.table(pd.DataFrame(data_weights))
        
        # ADDED LINK BUTTON HERE
        st.link_button("View Full Random Forest Rankings", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/Classic_Rankings/Random_Forest_Model/Random_Forest_Model.csv")
        
        # Random Forest Rankings Data
        st.subheader("Top 20 Classical Random Forest Rankings")
        rf_data = {
            "RF Rank": list(range(1, 21)),
            "Player Name": ["LeBron James", "Kareem Abdul-Jabbar", "Kobe Bryant", "Tim Duncan", "Shaquille O'Neal", "Karl Malone", "Bob Cousy", "Bill Russell", "Michael Jordan", "John Havlicek", "Hakeem Olajuwon", "Dirk Nowitzki", "Jerry West", "Kevin Durant", "Oscar Robertson", "Bob Pettit", "Stephen Curry", "Chris Paul", "Charles Barkley", "Magic Johnson"],
            "Model Index": [3.1016, 2.1472, 2.1106, 2.0361, 1.8489, 1.7808, 1.4243, 1.3864, 1.3627, 1.3595, 1.3447, 1.3286, 1.3099, 1.2045, 1.1010, 1.0722, 1.0547, 1.0519, 1.0469, 0.9922],
            "Actual Consensus Rank": [2.00, 3.00, 9.33, 6.33, 7.00, 18.00, 37.00, 4.67, 1.00, 30.67, 12.33, 18.33, 17.00, 13.00, 16.33, 33.00, 11.67, 28.00, 26.00, 5.00]
        }
        display_df = pd.DataFrame(rf_data).set_index("RF Rank")
        st.dataframe(pd.DataFrame(rf_data), use_container_width=True, hide_index=True)

    with class_tab2:
        st.subheader("Gradient Boosted Regressor (GBR)")
        st.write("""
        Unlike Random Forest, which builds independent trees in parallel, Gradient Boosting builds trees 
        sequentially. Each tree learns from the mistakes of the previous one, minimizing the 'loss' 
        (the difference between the model's prediction and the actual consensus) step-by-step.
        """)
        
        # Display performance metrics
        st.subheader("Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", "0.9558", "Variance Explained")
        col2.metric("MAE", "4.78", "Avg. Deviation")
        col3.metric("RMSE", "6.47", "Outlier Penalty")
        
        st.divider()
        
        # Narrative on why it's 'almost perfect'
        st.subheader("The 'Almost Perfect' Hierarchy")
        st.write("""
        This model effectively captures the 'Consensus Truth.' The ranking is nearly indistinguishable 
        from human expert consensus at the top end (1–3), which is rare for machine learning models. 
        
        **Why it differs from the Random Forest:** While Random Forest provides a 'crowd-sourced' view (averaging many trees), Gradient Boosting 
        is 'consensus-seeking.' It identifies the precise features that define the top tier, which is why 
        the 'Rank Delta' for players like Michael Jordan and LeBron James is 0.0. 
        
        The variance outliers (like Earl Monroe) highlight where the model strictly adheres to 
        available data metrics, while human consensus often incorporates narrative or 'eye-test' 
        factors that aren't captured in the box score.
        """)
        
        # Link to GitHub CSV
        st.link_button("View Full GBR Rankings", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/Classic_Rankings/Gradient_Boosted_CR/Gradient_Boosted_Metrics_Values.csv")

        # Top 20 Table
        st.subheader("Top 20 Gradient Boosted Rankings")
        # You can either load the CSV dynamically or hardcode this for the preview
        gbr_data = {
            "GB Rank": list(range(1, 21)),
            "Player Name": ["Michael Jordan", "LeBron James", "Kareem Abdul-Jabbar", "Tim Duncan", "Bill Russell", "Kobe Bryant", "Larry Bird", "Magic Johnson", "Shaquille O'Neal", "Wilt Chamberlain", "Kevin Durant", "Hakeem Olajuwon", "Karl Malone", "Stephen Curry", "Julius Erving", "David Robinson", "Kevin Garnett", "Dirk Nowitzki", "Jerry West", "Oscar Robertson"],
            "Model Index": [100.00, 99.49, 96.91, 96.88, 95.93, 95.56, 94.19, 94.18, 93.35, 92.98, 90.75, 89.80, 89.36, 88.29, 87.95, 84.53, 84.37, 84.33, 83.57, 79.71],
            "Actual Consensus Rank": [1.00, 2.00, 3.00, 6.33, 4.67, 9.33, 9.00, 5.00, 7.00, 8.00, 13.00, 12.33, 18.00, 11.67, 15.33, 19.67, 16.67, 18.33, 17.00, 16.33]
        }
        display_df = pd.DataFrame(gbr_data).set_index("GB Rank")
        st.dataframe(pd.DataFrame(gbr_data), use_container_width=True, hide_index=True)

# ==========================================
# 🚀 TAB 3: MODERN RANKINGS
# ==========================================
with tab3:
    st.header("Modern Analytics & Deep Learning Models")
    st.markdown("""
        ### The Holistic Approach
        This model moves beyond raw career totals to provide a high-fidelity view of player impact, 
        specifically designed to neutralize the statistical inflation seen across different eras.
        
        *   **Advanced Metric Integration:** We utilize **Box Plus-Minus (BPM)** for overall impact, **Win Shares per 48 (WS/48)** for efficiency, and **True Shooting Percentage (TS%)** to capture scoring accuracy.
        *   **Era-Scaling:** Every raw statistic is normalized against the league average of the specific year the player competed. This ensures a 1980s paint beast is measured by the same standard as a 2020s heliocentric playmaker.
        *   **Holistic Modeling:** By feeding these era-adjusted metrics into a **Gradient Boosting Regressor**, the model learns to prioritize players who dominated their specific competitive environment, rather than simply those who played the most games.
        """)
    
    # Sub-tabs for Modern Models
    mod_tab1, mod_tab2, mod_tab3 = st.tabs(["PCA Model", "Gradient Boosted Model", "Custom Deep Learning Model"])
    
    with mod_tab1:
        st.subheader("Principal Component Analysis (PCA)")
        st.write("""
        PCA is a dimensionality reduction technique used to simplify complex, high-dimensional player data 
        into a smaller set of 'Principal Components' that retain the most important patterns.
        """)
        
        # Methodology Explanation
        with st.expander("📊 Why 4 Components? (Methodology)"):
            st.write("""
            We selected 4 components based on the **Cumulative Explained Variance** threshold. 
            By retaining these 4 components, we captured **83.62%** of the total variance in NBA performance data. 
            
            This allows us to strip away 16.38% of 'noise' (statistically redundant info) while 
            keeping the 'signal' (the core factors of greatness). 
            
            - **Component 1 (54.10%):** The primary driver of "greatness," heavily correlated with PER, Win Shares, and accolades.
            - **Components 2-4 (29.52%):** Capture secondary nuances like defensive impact and era-specific efficiency adjustments.
            """)
        
        # Display performance metrics
        col1, col2 = st.columns(2)
        col1.metric("Total Variance Explained", "83.62%")
        col2.metric("R² Score", "0.7381", "Model Fit")
        
        st.divider()
        
        # Drivers and Importance
        st.subheader("Top Drivers (PC1) & Importance Profile")
        col_feat1, col_feat2 = st.columns(2)
        
        with col_feat1:
            st.markdown("**Top 5 Drivers (PC1):**")
            drivers = pd.DataFrame({
                "Metric": ["Playoff_WS/48", "Career_PER", "Career_WS/48", "Playoff_PER", "Peak_5Yr_PER"],
                "Weight": [0.268, 0.268, 0.268, 0.268, 0.251]
            })
            st.table(drivers)
            
        with col_feat2:
            st.markdown("**Component Importance:**")
            importance = pd.DataFrame({
                "Component": ["PC1", "PC2", "PC3", "PC4"],
                "Importance": [0.059, 0.015, 0.034, 0.021]
            })
            st.table(importance)

        st.link_button("View Full PCA Rankings", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/Modern_Rankings/PCA_Analysis/nba_pca_model_results.csv")

        # Top 20 Rankings
        st.subheader("Pure PCA Driven Top 20")
        pca_data = {
            "PCA Rank": list(range(1, 21)),
            "Player Name": [
                "Michael Jordan", "LeBron James", "Kobe Bryant", "Shaquille O'Neal", "Kareem Abdul-Jabbar", 
                "Larry Bird", "Tim Duncan", "Karl Malone", "Kevin Durant", "Hakeem Olajuwon", 
                "Magic Johnson", "Giannis Antetokounmpo", "Nikola Jokic", "Stephen Curry", "Dirk Nowitzki",
                "Charles Barkley", "James Harden", "Allen Iverson", "David Robinson", "Elvin Hayes"
            ],
            "Era": [
                "1990-1999", "2010-2019", "2000-2009", "2000-2009", "1980-1989", 
                "1980-1989", "2000-2009", "1990-1999", "2010-2019", "1990-1999", 
                "1980-1989", "2020-2029", "2020-2029", "2010-2019", "2000-2009",
                "1990-1999", "2010-2019", "2000-2009", "1990-1999", "1970-1979"
            ],
            "Consensus Rank": [
                1.00, 2.00, 7.67, 5.67, 3.00, 
                7.33, 5.33, 14.67, 10.67, 10.33, 
                4.33, 18.00, 17.67, 9.67, 15.00,
                21.67, 26.00, 34.33, 16.33, 38.33
            ]
        }
        display_df = pd.DataFrame(pca_data).set_index("PCA Rank")
        st.dataframe(pd.DataFrame(pca_data), use_container_width=True, hide_index=True)
        
    with mod_tab2:
        st.subheader("Modern Gradient Boosted Regressor (GBR)")
        st.write("""
        This model represents the 'Analytics-First' approach. By training exclusively on advanced rate 
        metrics, it strips away the 'narrative' weight of accolades and focuses entirely on 
        on-court productivity and efficiency.
        """)
        
        # Display performance metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", "0.8155", "Variance Explained")
        col2.metric("MAE", "11.64", "Avg. Rank Deviation")
        col3.metric("RMSE", "13.26", "Outlier Penalty")
        
        st.divider()
        
        # Feature Importance
        st.subheader("Feature Importance: The Efficiency Hierarchy")
        st.write("The model prioritizes 'Per-Possession' and 'Peak' efficiency metrics above all else.")
        
        importance_data = {
            "Metric": ["Career_PER", "Peak_5Yr_PER", "Career_TS_Pct", "Playoff_WS/48", "Career_WS/48", "Career_BPM", "Playoff_PER", "Playoff_BPM"],
            "Weight": ["37.93%", "24.14%", "17.39%", "4.90%", "4.37%", "4.13%", "3.79%", "3.36%"]
        }
        st.table(pd.DataFrame(importance_data))
        
        st.subheader("What These Statistics Tell Us")
        st.write("""
        - **PER (Player Efficiency Rating):** The core of the model. It aggregates all positive contributions 
          into a single per-minute rating.
        - **TS% (True Shooting Percentage):** Adjusts for the value of 3-pointers and free throws. It is 
          the ultimate measure of scoring efficiency.
        - **WS/48 (Win Shares per 48 Minutes):** Quantifies how many wins a player contributes 
          relative to a league average team.
        - **BPM (Box Plus/Minus):** Estimates a player's contribution to team performance per 100 
          possessions compared to a league-average player.
        """)

        st.link_button("View Full Modern GBR Rankings", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/Modern_Rankings/Gradient_Boosted/Gradient_Boosted_Advanced_Metrics.csv")

        # Top 10 Table
        st.subheader("Top 10 Advanced GBR Rankings")
        modern_gbr_data = {
            "Rank": list(range(1, 11)),
            "Player Name": ["Michael Jordan", "LeBron James", "Magic Johnson", "Kareem Abdul-Jabbar", "Tim Duncan", "Shaquille O'Neal", "Larry Bird", "Kobe Bryant", "Hakeem Olajuwon", "Stephen Curry"],
            "Maximal GBR Index": [100.00, 97.62, 95.66, 95.53, 93.21, 93.20, 92.49, 89.31, 88.39, 88.37]
        }
        display_df = pd.DataFrame(modern_gbr_data).set_index("Rank")
        st.dataframe(pd.DataFrame(modern_gbr_data), use_container_width=True, hide_index=True)
        
    with mod_tab3:
        st.subheader("Custom Machine-Optimized Model")
        st.write("""
        This model utilizes a custom gradient descent optimization to discover the 'ideal' weighting 
        of metrics that best approximates human expert consensus. It features era-relative 
        normalization to ensure players are compared fairly against their own contemporaries.
        """)
        
        # Display performance metrics
        col1, col2 = st.columns(2)
        col1.metric("R² Score", "0.8155", "Model Fit")
        col2.metric("MAE (Avg. Deviation)", "11.64", "spots")

        st.divider()
        
        st.subheader("Feature Importance: The Machine's Preference")
        st.write("These weights were learned via gradient descent optimization.")
        
        # Display the importance table you generated
        importance_df = pd.DataFrame({
            "Feature": ["All_NBA_PS", "Championships", "Peak_5Yr_PER", "All_NBA_Teams", "Playoff_PPG"],
            "Weight": [6.9171, 4.6608, 4.6384, 4.4411, 2.3219]
        })
        st.table(importance_df)
        
        st.link_button("View Full Model Results", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/Modern_Rankings/Deep_Learning/nba_machine_optimized_results.csv")

        # Display Top 20 Table
        st.subheader("Machine Optimized Top 20")
        dl_data = {
            "Rank": list(range(1, 21)),
            "Player": ["LeBron James", "Michael Jordan", "Kobe Bryant", "Kareem Abdul-Jabbar", "Shaquille O'Neal", "Tim Duncan", "Karl Malone", "Larry Bird", "Kevin Durant", "Giannis", "Hakeem Olajuwon", "Stephen Curry", "Magic Johnson", "David Robinson", "Charles Barkley", "Nikola Jokic", "Dirk Nowitzki", "James Harden", "George Gervin", "Elvin Hayes"],
            "Era": ["2010-2019", "1990-1999", "2000-2009", "1980-1989", "2000-2009", "2000-2009", "1990-1999", "1980-1989", "2010-2019", "2020-2029", "1990-1999", "2010-2019", "1980-1989", "1990-1999", "1990-1999", "2020-2029", "2000-2009", "2010-2019", "1980-1989", "1970-1979"]
        }
        st.dataframe(pd.DataFrame(dl_data), use_container_width=True, hide_index=True)


# ==========================================
# 🎯 TAB 4: NLP Model
# ==========================================
with tab4:
    st.header("NLP Sentiment & Ranking Model")
    
    # 1. Methodology Write-up
    st.markdown("""
        ### How the Model Works
        This module utilizes a **Deep Learning NLP Pipeline** powered by a **RoBERTa Transformer** to quantify historical and media-driven perception of NBA players. 
        
        *   **Data Ingestion:** We scrape career biographies and media commentary related to individual players to build a dense corpus of text.
        *   **Semantic Vectorization:** Instead of basic keyword matching, the model converts text into semantic embeddings, mapping the context against specific narrative dimensions using cosine similarity:
            *   **Respect Metrics:** Detecting language patterns regarding career achievements and elite status.
            *   **Failings/Narrative:** Identifying critiques, flaws, or negative media framing.
            *   **Attitude/Sentiment:** Scoring for positive leadership tone vs. toxic/negative behavior.
        *   **Net Sentiment Calculation:** The overall narrative score is mathematically derived as `(Respect + Positive) - (Failings + Negative)`.
        *   **Regressor Integration:** These derived sentiment vectors are combined with career advanced statistics (PER, BPM, Win Shares) as features for a **Gradient Boosting Regressor** pipeline.
        *   **Final Ranking:** The output score represents a synthesis of objective statistical performance and subjective media-driven "respect," creating a holistic view of a player's all-time ranking.
        """)

    # Side-by-side buttons for clean UI
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("View Full Model Results", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/NLP_Model/nlp_model_scores_ranked.csv", use_container_width=True)
    with col2:
        st.link_button("View the Scraped Articles Dataset", "https://github.com/Gotham2006/NBA_Top_100_Modeling/blob/main/NLP_Model/articles_credited.csv", use_container_width=True)

    st.divider()

    # 2. Data Loading & Display Table
    st.subheader("Model Scores by Player")
    
    try:
        # Dynamically load the generated CSV (Adjust the file path if your CSV is in a specific folder like 'NLP_Model/')
        df_nlp = pd.read_csv("nlp_model_scores_ranked.csv")
        
        # Display the dataframe with clean column formatting for your new specific metrics
        st.dataframe(
            df_nlp, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "Overall_NLP_Score": st.column_config.NumberColumn("Net Score", format="%.3f"),
                "Respect_Score": st.column_config.NumberColumn("Respect", format="%.2f"),
                "Failings_Score": st.column_config.NumberColumn("Failings", format="%.2f"),
                "Positive_Attitude_Score": st.column_config.NumberColumn("Positive", format="%.2f"),
                "Negative_Attitude_Score": st.column_config.NumberColumn("Negative", format="%.2f"),
                "Context_Length": st.column_config.NumberColumn("Text Length")
            }
        )
    except FileNotFoundError:
        st.error("⚠️ 'nlp_model_scores_ranked.csv' not found. Please ensure the path is correct and the file is pushed to GitHub.")

# ==========================================
# 🎯 TAB 5: CONCLUSION
# ==========================================
with tab5:
    st.header("Conclusion")
    
    st.markdown("""The models showed that
    the most important factors for these ranking seem to be the All-NBA Teams and other such media accolades rather than advanced statistics,
    which is suprising given the rise in the use of statistics in every major sport. Narratives also didn't play as big a role though that may be 
    due to All-NBA's and other achievements being narrative driven. 
    This project was a lot of fun as it allowed me to combine data science and NBA Rankings, two things I spend a lot of time on. Creating my own ML Model was a cool experience allowing me to apply the skills I learned from my AWS Certification course.
    The highlight of this project was seeing my custom model put Lebron James at the top of the rankings. 
    """)

    st.markdown("---")
    st.subheader("Technical Outcomes & Learning")
    st.markdown("""
    Throughout this project, we encountered and overcame several classic machine learning challenges that made the final results even more robust:

    *   **Gradient Boosting Excellence:** The Gradient Boosting Regressor proved to be the ideal choice. It excels at capturing the non-linear, complex relationships between basketball metrics, creating a remarkably strong, predictive model for greatness.
    *   **The 'Small Data, Wide Features' Paradox:** Managing a small set of historical greats against a vast number of features (PER, BPM, sentiment scores, etc.) is difficult. By tuning the tree depth and learning rate, we successfully mitigated overfitting, allowing the model to learn the true patterns of "all-time greatness."
    *   **NLP as the Narrative Soul:** It helped us look at the narrative importance of a players media ranking.
    """)

    st.markdown("---")
    st.subheader("Acknowledgments")
    st.markdown("""
    This project relied on a diverse range of high-quality data and computational support:
    *   **Data Sources:** The accuracy of our rankings is directly tied to the integrity of the basketball-reference data and the breadth of the articles scraped for our sentiment analysis.
    *   **The Collaboration:** The development of this pipeline was made possible through the iterative testing and architectural support provided by **Gemini 3.1**.
    """)


