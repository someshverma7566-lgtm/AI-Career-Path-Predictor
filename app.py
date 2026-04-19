from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data import CAREER_ARCHETYPES, INTEREST_COLUMNS, PERSONALITY_COLUMNS, SKILL_COLUMNS
from src.model import analyze_skill_gap, train_models, get_top_predictions


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "career_prediction_dataset.csv"


st.set_page_config(
    page_title="AI Career Path Predictor",
    page_icon="🚀",
    layout="wide",
)

sns.set_theme(style="whitegrid")


@st.cache_resource
def get_training_artifacts():
    return train_models(DATASET_PATH)


def parse_resume_features(uploaded_file) -> dict[str, int]:
    extracted = {column: 0 for column in SKILL_COLUMNS + INTEREST_COLUMNS + PERSONALITY_COLUMNS}
    if uploaded_file is None:
        return extracted

    content = uploaded_file.read().decode("utf-8", errors="ignore").lower()
    keyword_map = {
        "python": ["python", "pandas", "numpy"],
        "java": ["java", "spring"],
        "sql": ["sql", "mysql", "postgres"],
        "communication": ["communication", "presentation", "public speaking"],
        "design": ["design", "figma", "wireframe", "prototype"],
        "leadership": ["leadership", "managed", "team lead"],
        "marketing": ["marketing", "seo", "campaign", "brand"],
        "math": ["mathematics", "statistics", "calculus"],
        "creativity": ["creative", "brainstorm", "storytelling"],
        "analytics": ["analytics", "analysis", "dashboard", "insight"],
        "interest_tech": ["technology", "software", "coding"],
        "interest_business": ["business", "startup", "strategy"],
        "interest_creative": ["art", "creative", "media"],
        "interest_research": ["research", "experiment", "study"],
        "personality_introvert": ["independent", "self-motivated", "focused"],
        "personality_extrovert": ["collaborative", "outgoing", "networking"],
        "personality_analytical": ["analytical", "problem solving", "critical thinking"],
        "personality_empathetic": ["empathetic", "supportive", "user-focused"],
    }

    for feature, keywords in keyword_map.items():
        if any(keyword in content for keyword in keywords):
            extracted[feature] = 1
    return extracted


def render_sidebar() -> dict[str, int | float]:
    st.sidebar.header("Profile Input")
    st.sidebar.caption("Choose your strongest skills, interests, and traits.")

    resume_file = st.sidebar.file_uploader(
        "Upload resume (TXT optional)",
        type=["txt"],
        help="A lightweight keyword scan can auto-detect some skills.",
    )
    resume_features = parse_resume_features(resume_file)

    user_input: dict[str, int | float] = {}

    st.sidebar.subheader("Skills")
    for skill in SKILL_COLUMNS:
        label = skill.replace("_", " ").title()
        user_input[skill] = int(
            st.sidebar.checkbox(label, value=bool(resume_features.get(skill, 0)))
        )

    st.sidebar.subheader("Interests")
    interest_defaults = {
        "interest_tech": resume_features["interest_tech"],
        "interest_business": resume_features["interest_business"],
        "interest_creative": resume_features["interest_creative"],
        "interest_research": resume_features["interest_research"],
    }
    for interest in INTEREST_COLUMNS:
        label = interest.replace("interest_", "").replace("_", " ").title()
        user_input[interest] = int(
            st.sidebar.checkbox(label, value=bool(interest_defaults.get(interest, 0)))
        )

    st.sidebar.subheader("Personality")
    for trait in PERSONALITY_COLUMNS:
        label = trait.replace("personality_", "").replace("_", " ").title()
        user_input[trait] = int(
            st.sidebar.checkbox(label, value=bool(resume_features.get(trait, 0)))
        )

    user_input["gpa"] = st.sidebar.slider("Academic Score (GPA / 10)", 4.0, 10.0, 7.5, 0.1)
    user_input["problem_solving"] = st.sidebar.slider("Problem Solving", 1, 10, 7)

    return user_input


def render_prediction_cards(predictions: list[dict[str, float | str]], user_input: dict[str, int | float]) -> None:
    columns = st.columns(len(predictions))
    for column, prediction in zip(columns, predictions):
        with column:
            st.metric(
                label=str(prediction["career"]),
                value=f'{prediction["confidence"]}%',
            )
            gaps = analyze_skill_gap(user_input, str(prediction["career"]))
            if gaps:
                st.caption(f"Skill gaps: {', '.join(gaps)}")
            else:
                st.caption("Strong alignment across core skills.")


def render_visualizations(dataset: pd.DataFrame, comparison_scores: dict[str, float], predictions: list[dict[str, float | str]]) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Model Comparison")
        comparison_df = pd.DataFrame(
            {"Model": list(comparison_scores.keys()), "Accuracy": list(comparison_scores.values())}
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=comparison_df, x="Model", y="Accuracy", palette="crest", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("")
        plt.xticks(rotation=10)
        st.pyplot(fig)

    with right:
        st.subheader("Top Career Confidence")
        prediction_df = pd.DataFrame(predictions)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=prediction_df, x="confidence", y="career", palette="flare", ax=ax)
        ax.set_xlabel("Confidence (%)")
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Career Distribution in Training Data")
    fig, ax = plt.subplots(figsize=(10, 4))
    order = dataset["career"].value_counts().index
    sns.countplot(data=dataset, y="career", order=order, palette="viridis", ax=ax)
    ax.set_xlabel("Profiles")
    ax.set_ylabel("")
    st.pyplot(fig)


def main() -> None:
    artifacts = get_training_artifacts()
    user_input = render_sidebar()

    st.title("AI Career Path Predictor")
    st.write(
        "Predict suitable career paths from skills, interests, academic strength, and personality traits."
    )

    with st.expander("Supported Career Tracks", expanded=False):
        for career, profile in CAREER_ARCHETYPES.items():
            highlighted = [
                key.replace("_", " ").title()
                for key, value in profile.items()
                if key in SKILL_COLUMNS and value == 1
            ]
            st.write(f"**{career}**: {', '.join(highlighted)}")

    if st.button("Predict Career Paths", type="primary"):
        predictions = get_top_predictions(artifacts.model, user_input)
        st.subheader("Top 3 Career Matches")
        render_prediction_cards(predictions, user_input)

        best_match = predictions[0]["career"]
        st.success(f"Best fit right now: {best_match}")

        st.subheader("Skill Gap Analysis")
        gaps = analyze_skill_gap(user_input, str(best_match))
        if gaps:
            st.write("Recommended areas to strengthen:")
            for gap in gaps:
                st.write(f"- {gap}")
        else:
            st.write("You already match the core skill pattern for this path.")

        render_visualizations(artifacts.dataset, artifacts.comparison_scores, predictions)
    else:
        st.info("Fill in the profile and click Predict Career Paths to see recommendations.")


if __name__ == "__main__":
    main()
