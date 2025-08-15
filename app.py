import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

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
st.markdown("""
    <style>
        
        .main {background-color: #0D0D2B; color: white;}
        [data-testid="stSidebar"] {background-color: #1E1E3F; color: white;}
        h1, h2, h3 {color: #B266FF;}
        p, li {color: #EAEAEA;}
        div.stButton > button {
            background-color: #B266FF;
            color: white;
            border-radius: 10px;
            font-size: 16px;
        }
        div.stButton > button:hover {background-color: #A64DFF;}
    </style>
""", unsafe_allow_html=True)

# =========================
# Load Data & Model
# =========================
df = pd.read_csv("data\processed_dataset.csv")
treatment_model = joblib.load("models/clf_pipeline2.joblib")

# =========================
# Sidebar Navigation
# =========================
menu = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 EDA Visualizations", "🧾 Treatment Prediction Form",
     "👥 Cluster Visualization", "📄 Data Summary & Recommendations"]
)

# =========================
# Home Page
# =========================
if menu == "🏠 Home":
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

# =========================
# EDA Page
# =========================
elif menu == "📊 EDA Visualizations":
    st.title("📊 EDA Visualizations")
    st.write("Explore survey data with univariate, bivariate, and multivariate analysis.")

    # ========= UNIVARIATE ANALYSIS =========
    st.header("1️⃣ Univariate Analysis")
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
    st.header("2️⃣ Bivariate Analysis")
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
    st.header("3️⃣ Multivariate Analysis")
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

    
# =========================
# Treatment Prediction Form
# =========================
elif menu == "🧾 Treatment Prediction Form":
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

# =========================
# Cluster Visualization
# =========================
elif menu == "👥 Cluster Visualization":
    st.title("👥 Cluster Visualization")
    st.warning("Clustering visualization is under development.")

# =========================
# Data Summary & Recommendations
# =========================
elif menu == "📄 Data Summary & Recommendations":
    st.title("📄 Data Summary & Recommendations")
    st.dataframe(df.describe())
    st.info("Recommendations will be generated based on model insights.")

# =========================
# Footer
# =========================
st.markdown("""
    <hr>
    <p style='text-align: center; color: #888888; font-size: 14px;'>
    🚀 Built with ❤️ using Streamlit | © 2025 Team Lancer
    </p>
""", unsafe_allow_html=True)
