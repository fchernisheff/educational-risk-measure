import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Risk Predictor",
    page_icon="🎓",
    layout="wide"
)

@st.cache_resource
def load_assets():
    BASE_DIR = Path(__file__).parent

    MODEL_PATH = BASE_DIR / "model" / "ensemble_model.pkl"
    ENCODERS_PATH = BASE_DIR / "model" / "encoders.pkl"
    TARGET_PATH = BASE_DIR / "model" / "target_encoder.pkl"
    FEATURES_PATH = BASE_DIR / "model" / "features.pkl"

    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    target_encoder = joblib.load(TARGET_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, encoders, target_encoder, features

model, encoders, target_encoder, features = load_assets()

st.markdown(
    """
    <style>
    .main-title{
        text-align:center;
        font-size:42px;
        font-weight:700;
        margin-bottom:10px;
    }

    .sub-title{
        text-align:center;
        color:gray;
        margin-bottom:30px;
    }

    .risk-box{
        padding:20px;
        border-radius:15px;
        text-align:center;
        font-size:24px;
        font-weight:bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🎓 Student Performance Risk Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Predict academic performance risk using machine learning</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    year_class = st.selectbox(
        "Year Class",
        (encoders["year_class"].classes_)
    )

    program_stream = st.selectbox(
        "Program Stream",
        (encoders["program_stream"].classes_)
    )

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        (encoders["gender"].classes_)
    )

    cgpa_category = st.selectbox(
        "CGPA Category",
        (encoders["cgpa_category"].classes_)
    )

    academic_satisfaction = st.selectbox(
        "Academic Satisfaction",
        (encoders["academic_satisfaction"].classes_)
    )

    study_hours_daily = st.selectbox(
        "Study Hours Daily",
        (encoders["study_hours_daily"].classes_)
    )

    daily_productivity = st.slider(
        "Daily Productivity",
        1,
        10,
        5
    )

    revision_frequency = st.selectbox(
    "Revision Frequency",
    encoders["revision_frequency"].classes_.tolist()
    )

    focus_duration = st.selectbox(
        "Focus Duration",
        (encoders["focus_duration"].classes_)
    )

    screen_time_non_study = st.selectbox(
        "Screen Time (Non-study)",
        (encoders["screen_time_non_study"].classes_)
    )

    main_distractor = st.selectbox(
        "Main Distractor",
        (encoders["main_distractor"].classes_)
    )

    study_consistency = st.selectbox(
        "Study Consistency",
        (encoders["study_consistency"].classes_)
    )

    tasks_on_time = st.selectbox(
        "Tasks On Time",
        (encoders["tasks_on_time"].classes_)
    )

    preparation_status = st.selectbox(
        "Preparation Status",
        (encoders["preparation_status"].classes_)
    )

with col2:

    career_goal_clarity = st.selectbox(
        "Career Goal Clarity",
        (encoders["career_goal_clarity"].classes_)
    )

    skills_developing = st.selectbox(
        "Skills Developing",
        (encoders["skills_developing"].classes_)
    )

    energy_level = st.slider(
        "Energy Level",
        1,
        10,
        5
    )

    stress_level = st.slider(
        "Stress Level",
        1,
        10,
        5
    )

    routine_rating = st.slider(
        "Routine Rating",
        1,
        10,
        5
    )

    sleepy_during_study = st.selectbox(
        "Sleepy During Study",
        (encoders["sleepy_during_study"].classes_)
    )

    sleep_hours = st.selectbox(
        "Sleep Hours",
        (encoders["sleep_hours"].classes_)
    )

    career_interest = st.selectbox(
        "Career Interest",
        (encoders["career_interest"].classes_)
    )

    online_courses = st.selectbox(
        "Online Courses",
        (encoders["online_courses"].classes_)
    )

    projects_internships = st.selectbox(
        "Projects / Internships",
        (encoders["projects_internships"].classes_)
    )

    programming_foundation = st.selectbox(
        "Programming Foundation",
        (encoders["programming_foundation"].classes_)
    )

    events_participation = st.selectbox(
        "Events Participation",
        (encoders["events_participation"].classes_)
    )

    assignments_on_time = st.selectbox(
        "Assignments On Time",
        (encoders["assignments_on_time"].classes_)
    )

    attendance_percentage = st.selectbox(
        "Attendance Percentage",
        (encoders["attendance_percentage"].classes_)
    )

    strongest_asset = st.selectbox(
        "Strongest Asset",
        (encoders["strongest_asset"].classes_)
    )

    internal_barrier = st.selectbox(
        "Internal Barrier",
        (encoders["internal_barrier"].classes_)
    )

    external_resources = st.selectbox(
        "External Resources",
        (encoders["external_resources"].classes_)
    )

    external_pressure = st.selectbox(
        "External Pressure",
        (encoders["external_pressure"].classes_)
    )

if st.button("Analyze Academic Risk", use_container_width=True):

    data = pd.DataFrame([{
        "year_class": year_class,
        "program_stream": program_stream,
        "age": age,
        "gender": gender,
        "cgpa_category": cgpa_category,
        "academic_satisfaction": academic_satisfaction,
        "study_hours_daily": study_hours_daily,
        "daily_productivity": daily_productivity,
        "revision_frequency": revision_frequency,
        "focus_duration": focus_duration,
        "screen_time_non_study": screen_time_non_study,
        "main_distractor": main_distractor,
        "study_consistency": study_consistency,
        "tasks_on_time": tasks_on_time,
        "preparation_status": preparation_status,
        "career_goal_clarity": career_goal_clarity,
        "skills_developing": skills_developing,
        "energy_level": energy_level,
        "stress_level": stress_level,
        "routine_rating": routine_rating,
        "sleepy_during_study": sleepy_during_study,
        "sleep_hours": sleep_hours,
        "career_interest": career_interest,
        "online_courses": online_courses,
        "projects_internships": projects_internships,
        "programming_foundation": programming_foundation,
        "events_participation": events_participation,
        "assignments_on_time": assignments_on_time,
        "attendance_percentage": attendance_percentage,
        "strongest_asset": strongest_asset,
        "internal_barrier": internal_barrier,
        "external_resources": external_resources,
        "external_pressure": external_pressure
    }])

    for col in data.columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    risk_label = target_encoder.inverse_transform([prediction])[0]

    st.divider()

    if risk_label == "Low Risk":
        st.success(f"Predicted Risk Level: {risk_label}")

    elif risk_label == "Moderate Risk":
        st.warning(f"Predicted Risk Level: {risk_label}")

    else:
        st.error(f"Predicted Risk Level: {risk_label}")

    st.subheader("Prediction Probabilities")

    for cls, prob in zip(
        target_encoder.classes_,
        probabilities
    ):
        st.write(f"{cls}: {prob:.2%}")
        st.progress(float(prob))