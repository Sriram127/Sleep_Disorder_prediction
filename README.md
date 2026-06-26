# Sleep Disorder Prediction

A Streamlit web app that estimates the probability of a sleep issue and predicts whether the likely issue is **insomnia** or **sleep apnea**.

> **Disclaimer:** This project is for education and portfolio demonstration only. It is not medical advice and should not be used for diagnosis.

## Features

- Streamlit prediction UI with dark theme
- Pre-trained scikit-learn SVC models
- Sleep issue probability estimate
- Sleep issue type prediction (insomnia vs sleep apnea)
- BMI-based risk input
- Precaution guidance cards for each condition
- Docker support for containerised deployment

## Tech Stack

- Python 3
- Streamlit
- scikit-learn
- pandas / NumPy
- Docker

## Project Structure

```text
.
├── README.md
├── requirements.txt
└── sleepissuepredictor/
    ├── app.py
    ├── modeling.py
    ├── requirements.txt
    ├── Dockerfile
    ├── Sleep_health_and_lifestyle_dataset.csv
    ├── NoSystolic_ScaledModelSVC.pkl
    ├── SleepIssueType_ModelScaled.pkl
    ├── female_silhouette.png
    ├── male_silhouette.png
    ├── insomnia.jpeg
    └── sleep_apnea.jpeg
```

## Setup — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Sriram127/Sleep_Disorder_prediction.git
cd Sleep_Disorder_prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run sleepissuepredictor/app.py
```

The app opens at `http://localhost:8501`.

## Setup — Docker

```bash
cd sleepissuepredictor
docker build -t sleep-disorder-predictor .
docker run -p 8501:8501 sleep-disorder-predictor
```

## Dataset

Uses the [Sleep Health and Lifestyle dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset).  
Inputs: age, sleep duration, heart rate, daily steps, gender, BMI category, occupation type.

## Future Improvements

- Add a `train.py` script to regenerate models from raw data
- Add model metrics and confusion matrix to the app UI
- Add SHAP feature importance explanations
