from __future__ import annotations

from itertools import cycle
from pathlib import Path

import pandas as pd


SKILL_COLUMNS = [
    "python",
    "java",
    "sql",
    "communication",
    "design",
    "leadership",
    "marketing",
    "math",
    "creativity",
    "analytics",
]

INTEREST_COLUMNS = [
    "interest_tech",
    "interest_business",
    "interest_creative",
    "interest_research",
]

PERSONALITY_COLUMNS = [
    "personality_introvert",
    "personality_extrovert",
    "personality_analytical",
    "personality_empathetic",
]

NUMERIC_COLUMNS = ["gpa", "problem_solving"]

ALL_FEATURE_COLUMNS = SKILL_COLUMNS + INTEREST_COLUMNS + PERSONALITY_COLUMNS + NUMERIC_COLUMNS


CAREER_ARCHETYPES = {
    "Data Scientist": {
        "python": 1,
        "java": 0,
        "sql": 1,
        "communication": 1,
        "design": 0,
        "leadership": 0,
        "marketing": 0,
        "math": 1,
        "creativity": 0,
        "analytics": 1,
        "interest_tech": 1,
        "interest_business": 0,
        "interest_creative": 0,
        "interest_research": 1,
        "personality_introvert": 1,
        "personality_extrovert": 0,
        "personality_analytical": 1,
        "personality_empathetic": 0,
        "gpa": [7.5, 8.2, 8.8, 9.1],
        "problem_solving": [7, 8, 9, 9],
    },
    "Software Engineer": {
        "python": 1,
        "java": 1,
        "sql": 1,
        "communication": 1,
        "design": 0,
        "leadership": 0,
        "marketing": 0,
        "math": 1,
        "creativity": 0,
        "analytics": 1,
        "interest_tech": 1,
        "interest_business": 0,
        "interest_creative": 0,
        "interest_research": 0,
        "personality_introvert": 1,
        "personality_extrovert": 0,
        "personality_analytical": 1,
        "personality_empathetic": 0,
        "gpa": [6.8, 7.4, 8.0, 8.6],
        "problem_solving": [7, 8, 8, 9],
    },
    "Product Manager": {
        "python": 0,
        "java": 0,
        "sql": 1,
        "communication": 1,
        "design": 1,
        "leadership": 1,
        "marketing": 1,
        "math": 0,
        "creativity": 1,
        "analytics": 1,
        "interest_tech": 1,
        "interest_business": 1,
        "interest_creative": 0,
        "interest_research": 0,
        "personality_introvert": 0,
        "personality_extrovert": 1,
        "personality_analytical": 1,
        "personality_empathetic": 1,
        "gpa": [6.7, 7.2, 7.8, 8.4],
        "problem_solving": [6, 7, 8, 8],
    },
    "UX Designer": {
        "python": 0,
        "java": 0,
        "sql": 0,
        "communication": 1,
        "design": 1,
        "leadership": 0,
        "marketing": 0,
        "math": 0,
        "creativity": 1,
        "analytics": 0,
        "interest_tech": 0,
        "interest_business": 0,
        "interest_creative": 1,
        "interest_research": 1,
        "personality_introvert": 0,
        "personality_extrovert": 1,
        "personality_analytical": 0,
        "personality_empathetic": 1,
        "gpa": [6.5, 7.0, 7.4, 8.0],
        "problem_solving": [5, 6, 7, 7],
    },
    "Digital Marketer": {
        "python": 0,
        "java": 0,
        "sql": 0,
        "communication": 1,
        "design": 1,
        "leadership": 1,
        "marketing": 1,
        "math": 0,
        "creativity": 1,
        "analytics": 1,
        "interest_tech": 0,
        "interest_business": 1,
        "interest_creative": 1,
        "interest_research": 0,
        "personality_introvert": 0,
        "personality_extrovert": 1,
        "personality_analytical": 0,
        "personality_empathetic": 1,
        "gpa": [6.0, 6.6, 7.1, 7.8],
        "problem_solving": [5, 6, 6, 7],
    },
    "Business Analyst": {
        "python": 0,
        "java": 0,
        "sql": 1,
        "communication": 1,
        "design": 0,
        "leadership": 1,
        "marketing": 1,
        "math": 1,
        "creativity": 0,
        "analytics": 1,
        "interest_tech": 0,
        "interest_business": 1,
        "interest_creative": 0,
        "interest_research": 1,
        "personality_introvert": 0,
        "personality_extrovert": 1,
        "personality_analytical": 1,
        "personality_empathetic": 1,
        "gpa": [6.8, 7.3, 7.8, 8.3],
        "problem_solving": [6, 7, 8, 8],
    },
}


def create_sample_dataset(output_path: Path) -> pd.DataFrame:
    rows = []
    for career, profile in CAREER_ARCHETYPES.items():
        gpa_values = cycle(profile["gpa"])
        ps_values = cycle(profile["problem_solving"])
        for variant in range(18):
            row = {column: profile[column] for column in SKILL_COLUMNS + INTEREST_COLUMNS + PERSONALITY_COLUMNS}
            row["gpa"] = round(next(gpa_values) + ((variant % 3) * 0.1), 2)
            row["problem_solving"] = min(10, next(ps_values) + (variant % 2))

            if variant % 5 == 0:
                row["communication"] = 1
            if career in {"Data Scientist", "Business Analyst"} and variant % 4 == 0:
                row["interest_business"] = 1
            if career == "Software Engineer" and variant % 3 == 0:
                row["creativity"] = 1
            if career == "UX Designer" and variant % 3 == 1:
                row["analytics"] = 1
            if career == "Digital Marketer" and variant % 4 == 2:
                row["interest_tech"] = 1
            if career == "Product Manager" and variant % 4 == 3:
                row["python"] = 1

            row["career"] = career
            rows.append(row)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def load_or_create_dataset(dataset_path: Path) -> pd.DataFrame:
    if dataset_path.exists():
        return pd.read_csv(dataset_path)
    return create_sample_dataset(dataset_path)
