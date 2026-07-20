import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.main{
    background-color:#F5F7FA;
}

.title{
    text-align:center;
    color:#0B5394;
    font-size:42px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.prediction{
    padding:20px;
    border-radius:10px;
    background:#D6F5D6;
    text-align:center;
    font-size:28px;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
# joblib.dump(x.columns.tolist(), "feature_names.pkl")
model = joblib.load("Exam_Score_model.pkl")

# If saved during training
try:
    feature_names = joblib.load("feature_names.pkl")
except:
    feature_names = [
        'StudyHours','Attendance','Resources',
        'Extracurricular','Motivation','Internet',
        'Gender','Age','LearningStyle',
        'OnlineCourses','Discussions',
        'AssignmentCompletion',
        'EduTech','StressLevel','FinalGrade'
    ]

# -----------------------------
# Header
# -----------------------------
st.markdown('<p class="title">🎓 Student Performance Prediction System</p>',
            unsafe_allow_html=True)

st.markdown('<p class="subtitle">Predict Student Exam Score using Machine Learning</p>',
            unsafe_allow_html=True)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Student Details")

study = st.sidebar.slider("Study Hours",0,24,10)

attendance = st.sidebar.slider("Attendance (%)",0,100,80)

resources = st.sidebar.selectbox(
    "Resources Available",
    [0,1],
    format_func=lambda x:"Yes" if x==1 else "No"
)

extra = st.sidebar.selectbox(
    "Extracurricular",
    [0,1],
    format_func=lambda x:"Yes" if x==1 else "No"
)

motivation = st.sidebar.slider("Motivation Level",1,10,5)

internet = st.sidebar.selectbox(
    "Internet Access",
    [0,1],
    format_func=lambda x:"Yes" if x==1 else "No"
)

gender = st.sidebar.selectbox(
    "Gender",
    [0,1],
    format_func=lambda x:"Male" if x==1 else "Female"
)

age = st.sidebar.slider("Age",15,30,20)

learning = st.sidebar.selectbox(
    "Learning Style",
    [0,1,2]
)

courses = st.sidebar.slider("Online Courses",0,20,5)

discussion = st.sidebar.slider("Class Discussions",0,50,10)

assignment = st.sidebar.slider(
    "Assignment Completion (%)",
    0,
    100,
    80
)

edutech = st.sidebar.selectbox(
    "Uses EduTech",
    [0,1],
    format_func=lambda x:"Yes" if x==1 else "No"
)

stress = st.sidebar.slider(
    "Stress Level",
    1,
    10,
    5
)

grade = st.sidebar.selectbox(
    "Previous Final Grade",
    [0,1,2,3]
)

# -----------------------------
# Input DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "StudyHours":[study],
    "Attendance":[attendance],
    "Resources":[resources],
    "Extracurricular":[extra],
    "Motivation":[motivation],
    "Internet":[internet],
    "Gender":[gender],
    "Age":[age],
    "LearningStyle":[learning],
    "OnlineCourses":[courses],
    "Discussions":[discussion],
    "AssignmentCompletion":[assignment],
    "EduTech":[edutech],
    "StressLevel":[stress],
    "FinalGrade":[grade]
})

# Match training features
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -----------------------------
# Layout
# -----------------------------
col1,col2 = st.columns([2,1])

with col1:

    st.subheader("Student Information")

    st.dataframe(
        input_df,
        use_container_width=True
    )

    if st.button("Predict Exam Score", use_container_width=True):

        prediction = model.predict(input_df)[0]

        st.markdown(
            f'<div class="prediction">Predicted Exam Score<br>{prediction:.2f}</div>',
            unsafe_allow_html=True
        )

        st.progress(min(prediction/100,1.0))

with col2:

    st.subheader("Quick Summary")

    st.metric("Study Hours",study)

    st.metric("Attendance",f"{attendance}%")

    st.metric("Assignments",f"{assignment}%")

    st.metric("Stress Level",stress)

    st.metric("Online Courses",courses)

st.divider()

st.subheader("About Project")

st.info("""
This project predicts a student's expected examination score using
Machine Learning (Linear Regression).

**Model Used**
- Linear Regression

**Libraries**
- Scikit-learn
- Pandas
- NumPy
- Streamlit
""")

st.markdown(
    '<p class="footer">Developed using Streamlit • Student Performance Analytics</p>',
    unsafe_allow_html=True
)