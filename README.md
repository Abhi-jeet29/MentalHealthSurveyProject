This repository contains the code, data, models, and application for the OpenLearn Cohort 1.0 Capstone Project 2025: Mental Wellness Analysis and Support Strategy Design for the Tech Workforce. The project analyzes mental health survey data from tech professionals to predict treatment-seeking behavior, estimate age for targeted interventions, and segment employees into clusters for tailored HR policies. It addresses rising issues like burnout, disengagement, and attrition in the tech industry through data-driven insights.
The project is structured as a case study for NeuronInsights Analytics, contracted by tech companies (CodeLab, QuantumEdge, SynapseWorks) to develop proactive strategies. Key questions include: Who is likely to avoid treatment? How do factors like remote work and benefits influence well-being? Can employee profiles be segmented for targeted HR interventions?
Project Summary
Objectives

Classification Task: Predict if an individual is likely to seek mental health treatment (e.g., using logistic regression or similar models, achieving ~82% accuracy).
Regression Task: Predict an individual's age based on personal and workplace attributes to support age-targeted interventions (e.g., Ridge regression with RMSE ~6.8).
Unsupervised Task: Segment tech employees into distinct clusters (e.g., 3 clusters via KMeans) based on mental health indicators like support scores and work interference, aiding in personalized HR policies.

Dataset

Source: OSMI (Open Sourcing Mental Illness) Mental Health in Tech Survey (2014-2016).
Size: ~1,259 entries.
Key Features: Demographics (age, gender, country), workplace environment (benefits, leave policies, remote work), personal experiences (mental illness, family history), attitudes (perceived consequences, social support).
Data Files:

data/survey.csv: Raw dataset before EDA.
data/processed_dataset.csv: Cleaned and encoded dataset after EDA (numerical scales for categoricals, composite scores like workplace_support).
data/clustered_personas.csv: Dataset with cluster labels and personas after unsupervised learning.



Methodology

EDA: Performed in notebook/01_EDA_Merged.ipynb (distributions, correlations, visuals like heatmaps and boxplots).
Modeling:

Classification: notebook/Classification.ipynb (model saved as models/clf1_pipeline.joblib).
Regression: notebook/03_Regression.ipynb (model saved as models/ridge_model_pipeline_v2.joblib).
Clustering: notebook/03_clustering.ipynb (model saved as models/kmeans_pipeline.joblib).


Application: Interactive Streamlit dashboard (app.py) for visualizations, predictions, and recommendations.
Tools: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Plotly, Joblib, Streamlit.

Results & Insights

Strong predictors: Family history (correlation ~0.26 with treatment), work interference, workplace support.
Clusters: E.g., young high-interference group vs. older family-history prone.
Business Value: Recommendations for enhanced benefits, managerial training, and segmented interventions to reduce attrition by 15-20%.

Setup Instructions
Prerequisites

Python 3.8+
Git
Virtual environment (recommended: venv or conda)

Installation

Clone the Repository:
textgit clone https://github.com/Abhi-jeet29/MentalHealthSurveyProject.git
cd MentalHealthSurveyProject

Create and Activate Virtual Environment:
textpython -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

Install Dependencies:
textpip install -r requirements.txt
(If requirements.txt is not present, install core packages manually: pip install pandas numpy scikit-learn matplotlib seaborn plotly streamlit joblib.)
Run Notebooks (Optional, for exploration/model training):

Open in Jupyter: jupyter notebook or VS Code.
Execute notebook/01_EDA_Merged.ipynb for data cleaning and visuals.
Run notebook/Classification.ipynb, notebook/03_Regression.ipynb, and notebook/03_clustering.ipynb to train/reproduce models.


Run the Streamlit App:
textstreamlit run app.py

Access at http://localhost:8501.
Features: Data exploration, model predictions (treatment, age, cluster), visualizations (e.g., PCA plots, confidence scores), persona-based recommendations.



Model Usage

Load models directly: e.g., model = joblib.load('models/clf1_pipeline.joblib') for predictions.
Ensure input data matches processed format (numerical encodings).

Troubleshooting

If models fail to load: Check Joblib version compatibility.
Data paths: All relative to repo root.
For clustering predictions: Input must align with training features (e.g., drop non-numeric columns).

Feature Description
The dataset includes ~16 key features post-processing (from original 27). Below is a description of main features (based on processed_dataset.csv):

Age: Numerical (e.g., 18-60, mean ~32.8). Capped for outliers. Used in regression target.
Gender: Categorical (e.g., "Male", "Female", "Other"). Heavily male-skewed (~96%).
self_employed: Binary (0=No, 1=Yes). Indicates self-employment status.
family_history: Binary (0=No, 1=Yes). Family history of mental illness; strong predictor for treatment.
treatment: Binary (0=No, 1=Yes). Target for classification; ~67% sought treatment.
work_interfere: Numerical scale (0=Never to 1=Often). How often mental health interferes with work.
no_employees: Numerical scale (0=1-5 to 1=1000+). Company size.
remote_work: Binary (0=No, 1=Yes). Remote work status; correlates with interference.
tech_company: Binary (0=No, 1=Yes). Whether employer is a tech company (higher support scores).
leave: Numerical scale (0=Very difficult to 1=Very easy). Ease of taking mental health leave.
mental_vs_physical: Numerical scale (0=No to 1=Yes). If employer treats mental/physical health equally.
Country_top: Categorical (e.g., "United States" ~57%, "United Kingdom", "Canada", "Other").
workplace_support: Composite score (0-1, mean ~0.45). Average of benefits, anonymity, etc.; key for clustering.
health_interview: Composite score (0-1). Willingness to discuss health in interviews.
health_consequence: Composite score (0-1, mean ~0.21). Perceived negative consequences.
social_support: Composite score (0-1, mean ~0.57). Willingness to discuss with coworkers/supervisor.

Additional derived features in clustered_personas.csv:

Cluster: Integer (0-2). Assigned cluster label.
Cluster_Name: Descriptive (e.g., "High Support, Low Interference").
Confidence: Model confidence score (0-1) for cluster assignment.

For full details, refer to EDA notebook or dataset CSVs.
License
MIT License. See LICENSE for details.
Contact

GitHub: Abhi-jeet29
Email: guptabhijit.ag.1@gmail.com

Contributions welcome! Feel free to open issues or PRs.3.8sHow can Grok help?
