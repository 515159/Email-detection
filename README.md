# AI-Powered Phishing Email Detection System

A Flask web app that classifies pasted email text as phishing or legitimate using a scikit-learn text classification pipeline.

## Features

- TF-IDF text vectorization
- Logistic Regression classifier
- Reusable text-cleaning function for training and prediction
- Flask form for pasting email content
- Color-coded prediction result with confidence score
- Sample CSV dataset included for a quick local demo

## Project Structure

```text
app/
  app.py
  ml.py
  static/styles.css
  templates/index.html
data/
  sample_emails.csv
models/
  phishing_model.joblib
train_model.py
requirements.txt
README.md
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python -m app.app
```

Then open:

```text
http://127.0.0.1:5000
```

## Using Your Own Dataset

Replace `data/sample_emails.csv` with a larger labeled dataset. The CSV must contain:

- `text`: the email body or subject/body text
- `label`: `phishing` or `legitimate`

Then retrain:

```powershell
python train_model.py --data data/your_dataset.csv
```

## Notes

The included sample dataset is intentionally small so the project can run immediately. For real use, train with a larger public dataset such as Kaggle or UCI phishing email datasets, then review precision, recall, F1 score, and confusion matrix before deployment.
