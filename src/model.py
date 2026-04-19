from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.data import ALL_FEATURE_COLUMNS, CAREER_ARCHETYPES, SKILL_COLUMNS, load_or_create_dataset


@dataclass
class TrainingArtifacts:
    dataset: pd.DataFrame
    model: RandomForestClassifier
    comparison_scores: dict[str, float]


def train_models(dataset_path: Path) -> TrainingArtifacts:
    dataset = load_or_create_dataset(dataset_path)
    x = dataset[ALL_FEATURE_COLUMNS]
    y = dataset["career"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    random_forest = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42,
    )
    logistic_regression = LogisticRegression(max_iter=1000)
    decision_tree = DecisionTreeClassifier(max_depth=8, random_state=42)

    random_forest.fit(x_train, y_train)
    logistic_regression.fit(x_train, y_train)
    decision_tree.fit(x_train, y_train)

    comparison_scores = {
        "Random Forest": accuracy_score(y_test, random_forest.predict(x_test)),
        "Logistic Regression": accuracy_score(y_test, logistic_regression.predict(x_test)),
        "Decision Tree": accuracy_score(y_test, decision_tree.predict(x_test)),
    }

    return TrainingArtifacts(
        dataset=dataset,
        model=random_forest,
        comparison_scores=comparison_scores,
    )


def build_input_frame(user_input: dict[str, float | int]) -> pd.DataFrame:
    row = {column: user_input.get(column, 0) for column in ALL_FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)


def get_top_predictions(model: RandomForestClassifier, user_input: dict[str, float | int], top_k: int = 3) -> list[dict[str, float | str]]:
    input_frame = build_input_frame(user_input)
    probabilities = model.predict_proba(input_frame)[0]
    labels = model.classes_

    ranked = sorted(
        zip(labels, probabilities),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]

    return [
        {"career": career, "confidence": round(float(score) * 100, 2)}
        for career, score in ranked
    ]


def analyze_skill_gap(user_input: dict[str, float | int], target_career: str) -> list[str]:
    target = CAREER_ARCHETYPES[target_career]
    missing = []
    for skill in SKILL_COLUMNS:
        if target[skill] == 1 and user_input.get(skill, 0) == 0:
            missing.append(skill.replace("_", " ").title())
    return missing[:5]
