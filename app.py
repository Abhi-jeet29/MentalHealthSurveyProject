import os
import io
import json
import base64
import joblib
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from joblib import load
import streamlit as st

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Mental Health Survey Dashboard",
    page_icon="🧠",
    layout="wide"
)

# =========================
# Custom CSS Styling
# =========================
# Add this at the top of your app.py (after imports)
st.markdown(
    """
    <style>
    /* Make all text darker and more readable */
    body, p, div, span, h1, h2, h3, h4, h5, h6 {
        color: #222222 !important;  /* Dark gray for better readability */
    }

    /* Style the subtitle area */
    .block-container p {
        font-size: 18px;
        font-weight: 500;
        color: #333333 !important; /* Darker text for subtitles */
    }

    /* Improve sidebar text visibility */
    section[data-testid="stSidebar"] {
        color: white !important;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Set page config
st.set_page_config(page_title="Mental Health Survey", layout="wide")

# Initialize session state
if "app_started" not in st.session_state:
    st.session_state.app_started = False

# Function to start app
def start_app():
    st.session_state.app_started = True

# ================================
# Splash / Landing Page
# ================================
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    # Landing page
    st.markdown("<h1 style='text-align: center;'>MentalHealthSurveyProject</h1>", unsafe_allow_html=True)
    if st.button("🚀 Start", use_container_width=True):
        st.session_state.started = True
        # st.experimental_rerun()
# if not st.session_state.app_started:
#     st.markdown(
#         """
#         <style>
#         .big-title {
#             font-size: 60px;
#             font-weight: bold;
#             color: #6A0DAD; /* Violet */
#             text-align: center;
#             margin-top: 100px;
#         }
#         .start-btn {
#             display: flex;
#             justify-content: center;
#             margin-top: 40px;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

#     st.markdown('<div class="big-title">MentalHealthSurveyProject</div>', unsafe_allow_html=True)
#     st.markdown('<div class="start-btn">', unsafe_allow_html=True)
#     if st.button("🚀 Start", key="start_button", use_container_width=True):
#         st.session_state.started = True
#         st.experimental_rerun()
#         #start_app()
#     #st.markdown('</div>', unsafe_allow_html=True)
else:
# =========================
# Load Data & Model
# =========================
    df = pd.read_csv("data\processed_dataset.csv")
    treatment_model = joblib.load("models/clf1_pipeline.joblib")
    st.cache_resource
    def load_model():
        return joblib.load("models/ridge_model_pipeline_v2.joblib")

    model = load_model()

    # ====== Load Data and Model ======
    @st.cache_data
    def load_clustered_data():
        return pd.read_csv("data/clustered_personas.csv")

    @st.cache_resource
    def load_model():
        return load("models/kmeans_pipeline.joblib")  # Your trained pipeline (Scaler + KMeans)

    cluster_df = load_clustered_data()
    pipeline = load_model()

    
    # Keep only numeric columns for PCA
    feature_cols = cluster_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ["Cluster"]]  # drop cluster ID if present

# Run PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cluster_df[feature_cols])
    cluster_df["PCA1"] = pca_result[:, 0]
    cluster_df["PCA2"] = pca_result[:, 1]



# ================================
# Main App After Start
# ================================
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to", ["🏠 Home", "📊 EDA Visualizations", "🧠 Treatment Prediction", "🎯 Age Prediction", "📌 Cluster Visualization"])

    if menu == "🏠 Home":
            st.title("Welcome to the Home Page")
            st.write("This is your dashboard after clicking Start.")
            st.markdown("<h1 style='text-align: center;'>🧠 Welcome to the Mental Health Survey Dashboard</h1>", unsafe_allow_html=True)
            st.image("https://images.unsplash.com/photo-1553729459-efe14ef6055d", use_container_width=True)
            st.write("## 🌟 Why Mental Health Matters")
            st.info("This dashboard helps explore mental health patterns, predict treatment seeking, and visualize workplace factors.")
            demo_df = pd.DataFrame({
                "Category": ["Stress", "Anxiety", "Depression", "Well-being"],
                "Percentage": [45, 30, 15, 10]
            })
            fig = px.pie(demo_df, names="Category", values="Percentage", title="Common Mental Health Issues", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
            <hr>
            <p style='text-align: center; color: #888888; font-size: 14px;'>
            🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
            </p>
            """, unsafe_allow_html=True)
    
    elif menu == "📊 EDA Visualizations":
            st.title("EDA Section")
            st.write("Show your data analysis here.")
            st.title("📊 EDA Visualizations")
            st.write("Explore survey data with univariate, bivariate, and multivariate analysis.")

            with st.expander("🔎 Filters (optional)"):
                numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                cat_cols = [c for c in df.columns if c not in numeric_cols]
                sub_df = df.copy()

                if len(cat_cols):
                    cat_to_filter = st.multiselect("Categorical filters (choose column=value)", cat_cols)
                    for col in cat_to_filter:
                        val = st.selectbox(f"{col} =", sorted(sub_df[col].dropna().unique()), key=f"fil_{col}")
                        sub_df = sub_df[sub_df[col] == val]

                if len(numeric_cols):
                    num_to_range = st.multiselect("Numeric range filters", numeric_cols)
                    for col in num_to_range:
                        mn, mx = float(sub_df[col].min()), float(sub_df[col].max())
                        a, b = st.slider(f"{col} range", mn, mx, (mn, mx))
                        sub_df = sub_df[(sub_df[col] >= a) & (sub_df[col] <= b)]

            tab1, tab2, tab3, tab4 = st.tabs(["📄 Overview", "📈 Distributions", "🔗 Relationships", "🧩 Merged Plots"])

            with tab1:
                st.subheader("Quick Look")
                c1, c2 = st.columns(2) 
                with c1:
                    st.subheader("Head")
                    st.dataframe(sub_df.head(10), use_container_width=True)
                with c2:
                    st.subheader("Describe")
                    st.dataframe(sub_df.describe(include="all").T, use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.subheader("DTypes")
                    st.dataframe(sub_df.dtypes.to_frame("dtype"))
                    with c4:
                        st.subheader("Missingness")
                        miss = sub_df.isna().mean().sort_values(ascending=False).rename("missing_ratio")
                        st.bar_chart(miss)

    # ========= UNIVARIATE ANALYSIS =========
            st.header("⿡ Univariate Analysis")
            st.subheader("Age Distribution")
            fig1 = px.histogram(df, x="Age", nbins=20, title="Age Distribution")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Gender Distribution")
            fig2 = px.pie(df, names="Gender", title="Gender Distribution", hole=0.4)
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Workplace Support Distribution")
            fig3 = px.histogram(df, x="workplace_support", title="Workplace Support Levels")
            st.plotly_chart(fig3, use_container_width=True)

    # ========= BIVARIATE ANALYSIS =========
            st.header("⿢ Bivariate Analysis")
            st.subheader("Treatment Seeking by Gender")
            fig4 = px.bar(df, x="Gender", color="treatment", title="Treatment by Gender")
            st.plotly_chart(fig4, use_container_width=True)

            st.subheader("Age vs Treatment")
            fig5 = px.box(df, x="treatment", y="Age", color="treatment",
                    title="Age Distribution by Treatment Status")
            st.plotly_chart(fig5, use_container_width=True)

            st.subheader("Effect of Leave Policies on Treatment Seeking")
            leave_treatment = df.groupby("leave")["treatment"].mean().reset_index()
            fig6 = px.bar(leave_treatment, x="leave", y="treatment",
                    labels={"treatment": "Proportion Seeking Treatment"},
                    title="Leave Policy Impact on Treatment")
            st.plotly_chart(fig6, use_container_width=True)

    # ========= MULTIVARIATE ANALYSIS =========
            st.header("⿣ Multivariate Analysis")
            st.subheader("Age, Workplace Support, and Treatment")
            fig7 = px.scatter(df, x="Age", y="workplace_support",
                        color="treatment", size="social_support",
                        title="Age vs Workplace Support (Colored by Treatment, Sized by Social Support)",
                        hover_data=["no_employees", "leave"])
            st.plotly_chart(fig7, use_container_width=True)

            st.subheader("Correlation Heatmap (Numerical Features)")
            import seaborn as sns
            import matplotlib.pyplot as plt
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr = numeric_df.corr()
            fig8, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
            st.pyplot(fig8)

            st.subheader("Stacked Bar: Leave Policy & Workplace Support by Treatment")
            grouped = df.groupby(["leave", "workplace_support"])["treatment"].mean().reset_index()
            fig9 = px.bar(grouped, x="leave", y="treatment", color="workplace_support",
                    title="Leave Policy & Workplace Support Impact",
                    labels={"treatment": "Proportion Seeking Treatment"},
                    barmode="stack")
            st.plotly_chart(fig9, use_container_width=True)
            st.markdown("""
            <hr>
            <p style='text-align: center; color: #888888; font-size: 14px;'>
            🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
                </p>
        """, unsafe_allow_html=True)
    
    elif menu == "🧠 Treatment Prediction":
        st.title("Treatment Prediction Form")
        st.title("🧾 Predict Treatment Seeking")
        age = st.slider("Age", 18, 70, 30)
        gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
        self_employed = st.selectbox("Self-employed", [0, 1])
        family_history = st.selectbox("Family History", [0, 1])
        work_interfere = st.selectbox("Work Interference", [0, 1, 2, 3])
        no_employees = st.selectbox("Company Size", [1, 2, 3, 4, 5])
        remote_work = st.selectbox("Remote Work", [0, 1])
        tech_company = st.selectbox("Tech Company", [0, 1])
        leave = st.selectbox("Ease of Taking Leave", [0, 1, 2])
        mental_vs_physical = st.selectbox("Mental vs Physical Health", [0, 1, 2])
        workplace_support = st.selectbox("Workplace Support", [0, 1, 2])
        health_interview = st.selectbox("Mental Health Interview", [0, 1, 2])
        health_consequence = st.selectbox("Health Consequence", [0, 1, 2])
        social_support = st.selectbox("Social Support", [0, 1, 2])

        Gender_Female = 1 if gender == "Female" else 0
        Gender_Male = 1 if gender == "Male" else 0
        Gender_Non_binary = 1 if gender == "Non-binary" else 0

        if st.button("Predict Treatment"):
            user_input = pd.DataFrame([[
                age, self_employed, family_history, work_interfere, no_employees,
                remote_work, tech_company, leave, mental_vs_physical, workplace_support,
                health_interview, health_consequence, social_support,
                Gender_Female, Gender_Male, Gender_Non_binary
            ]], columns=[
                'Age', 'self_employed', 'family_history', 'work_interfere',
                'no_employees', 'remote_work', 'tech_company', 'leave',
                'mental_vs_physical', 'workplace_support', 'health_interview',
                'health_consequence', 'social_support',
   
                'Gender_Female', 'Gender_Male', 'Gender_Non-binary'
            ])
            pred = treatment_model.predict(user_input)[0]
            prob = treatment_model.predict_proba(user_input)[0][pred]
            st.success(f"Prediction: {'Will Seek Treatment' if pred==1 else 'Will Not Seek Treatment'}")
            st.info(f"Model Confidence: {prob*100:.2f}%")



            st.markdown("""
            <hr>
            <p style='text-align: center; color: #888888; font-size: 14px;'>
            🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
            </p>
        """, unsafe_allow_html=True)
    
    elif menu == "🎯 Age Prediction":
        st.title("Age Prediction Model")
        st.title("🧮 Age Prediction")
        st.write("Fill in the details below and get your predicted age.")
        try:
            feature_names = model.feature_names_in_
        except AttributeError:
    # If pipeline doesn't store feature names, load from dataset
            df = pd.read_csv("processed_dataset.csv")
            feature_names = [col for col in df.columns if col != "Age"]  # Assuming 'Age' is target
         #Create form for user input
        with st.form("age_form"):
            st.subheader("Enter the required details:")
    
            user_data = {}
            for feature in feature_names:
        # Here assuming numeric inputs, you can customize for categorical features
                user_data[feature] = st.number_input(f"{feature}", value=0.0)
            submitted = st.form_submit_button("Predict Age")
    # Predict when form is submitted
        if submitted:
            input_df = pd.DataFrame([user_data])  # One row dataframe
            prediction = model.predict(input_df)[0]
            st.success(f"🎯 Predicted Age: **{prediction:.1f} years**")
            st.markdown("""
        <hr>
        <p style='text-align: center; color: #888888; font-size: 14px;'>
        🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
        </p>
    """, unsafe_allow_html=True)
    
    elif menu == "📌 Cluster Visualization":
        st.title("Cluster Visualization")

        # ====== Persona Descriptions & Recommendations ======
        persona_map = {
    "Open Advocates": "Comfortable discussing mental health, perceive strong workplace support.",
    "Under-Supported Professionals": "Open to help but lack adequate workplace support.",
    "Silent Strugglers": "Low openness, perceive little workplace support."
        }
        recommendations_map = {
        "Open Advocates": [
        "Empower as mental health ambassadors.",
        "Co-design awareness and support programs.",
        "Keep access to resources easy and visible."
        ],
        "Under-Supported Professionals": [
        "Improve access to mental health benefits.",
        "Offer clear communication on leave and support policies.",
        "Encourage managers to proactively offer help."
        ],
        "Silent Strugglers": [
        "Provide anonymous counseling and mental health checks.",
        "Promote a culture of openness without fear of consequences.",
        "Train managers to identify and support at-risk employees."
             ]
}

# ====== Streamlit UI ======
        st.header("🧩 Employee Cluster Visualizer & Insights")

# --- Scatter Plot ---
        st.subheader("📊 Cluster Visualization")
        fig = px.scatter(
            cluster_df, x="PCA1", y="PCA2",
            color="Cluster_Name",
            hover_data=["Age", "Gender", "Country_top"],
            title="Mental Health Personas"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Persona Descriptions ---
        st.subheader("🧭 Persona Descriptions")
        for cname, desc in persona_map.items():
            st.markdown(f"**{cname}:** {desc}")

# --- Data Summary ---
        st.subheader("📈 Data Summary")
        st.write("**Cluster Counts:**")
        st.dataframe(cluster_df["Cluster_Name"].value_counts())

        st.write("**Cluster Means (numeric features):**")
        numeric_cols = cluster_df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(cluster_df.groupby("Cluster_Name")[numeric_cols].mean().round(2))

# --- Recommendations ---
        st.subheader("🧪 Recommendations by Cluster")
        selected_persona = st.selectbox("Select persona:", list(persona_map.keys()))
        st.markdown(f"**Persona:** {persona_map[selected_persona]}")
        st.write("**Recommendations:**")
        for i, rec in enumerate(recommendations_map[selected_persona], start=1):
            st.write(f"{i}. {rec}")

# --- Model Confidence ---
         # --- Model Confidence ---
        st.subheader("📏 Model Confidence")

# Get only the columns the model was trained on
        try:
    # If pipeline has a 'preprocessor' or similar step with column names
            if "preprocessor" in pipeline.named_steps:
                trained_features = pipeline.named_steps["preprocessor"].feature_names_in_
            else:
              trained_features = pipeline.feature_names_in_
        except AttributeError:
            st.error("Could not find training feature names in pipeline. Using numeric columns as fallback.")
            trained_features = cluster_df.select_dtypes(include=np.number).columns.tolist()

# Select exactly those columns from your dataset
        X = cluster_df[trained_features]

# Compute confidence
        if hasattr(pipeline.named_steps["kmeans"], "transform"):
            dists = pipeline.named_steps["kmeans"].transform(X)
            min_dists = np.min(dists, axis=1)
            second_min_dists = np.sort(dists, axis=1)[:, 1]
            confidence = 1 - (min_dists / (min_dists + second_min_dists + 1e-9))
            cluster_df["Confidence"] = confidence
            st.dataframe(
                cluster_df[["Cluster_Name", "Confidence"]]
                .groupby("Cluster_Name").mean().round(3)
            )
        else:
            st.info("KMeans transform not available in pipeline. Cannot compute confidence.")


# --- Predict New User ---
        st.subheader("🧍 Predict Cluster for a New User")
        new_user_data = {}
        for col in feature_cols:
            if cluster_df[col].dtype in [np.float64, np.int64]:
                new_user_data[col] = st.number_input(col, value=float(cluster_df[col].mean()))
            else:
                new_user_data[col] = st.text_input(col, "")

        if st.button("Predict Cluster"):
            new_df = pd.DataFrame([new_user_data])
            pred_cluster = pipeline.predict(new_df)[0]
            persona = cluster_df[cluster_df["Cluster"] == pred_cluster]["Cluster_Name"].iloc[0]
            st.success(f"Predicted Cluster: {pred_cluster} ({persona})")
            st.markdown(f"**Persona:** {persona_map.get(persona, 'No description available')}")
            recs = recommendations_map.get(persona, [])
            if recs:
                st.write("**Recommended actions:**")
                for i, rec in enumerate(recs, start=1):
                    st.write(f"{i}. {rec}")
                st.markdown("""
                <hr>
                <p style='text-align: center; color: #888888; font-size: 14px;'>
                    🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
            </p>
            """, unsafe_allow_html=True)












# # =========================
# # Cluster Visualization
# # =========================
# elif menu == "👥 Cluster Visualization":
#     st.title("👥 Cluster Visualization")
#     st.warning("Clustering visualization is under development.")

# # =========================
# # Data Summary & Recommendations
# # =========================
# elif menu == "📄 Data Summary & Recommendations":
#     st.title("📄 Data Summary & Recommendations")
#     st.dataframe(df.describe())
#     st.info("Recommendations will be generated based on model insights.")

# # =========================
# # Footer
# # =========================
# st.markdown("""
#     <hr>
#     <p style='text-align: center; color: #888888; font-size: 14px;'>
#     🚀 Built with ❤ using Streamlit | © 2025 Team Lancer
#     </p>
# """, unsafe_allow_html=True)