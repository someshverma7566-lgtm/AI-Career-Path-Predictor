# AI Career Path Predictor

An intelligent machine learning web application that predicts suitable career paths based on user skills, interests, academic performance, and personality traits.

## Features

- Predicts the top 3 career options
- Shows confidence scores
- Includes skill gap analysis
- Supports optional resume upload from `.txt`
- Offers an interactive Streamlit UI
- Displays model and dataset visualizations

## Tech Stack

- Python
- scikit-learn
- Pandas and NumPy
- Streamlit
- Matplotlib and Seaborn

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
`-- src/
    |-- __init__.py
    |-- data.py
    `-- model.py
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

The app creates a starter dataset automatically at `data/career_prediction_dataset.csv` the first time it runs. You can later replace that file with a Kaggle dataset that uses similar features.

## Models

- Random Forest: primary prediction model
- Logistic Regression: baseline model
- Decision Tree: comparison model
