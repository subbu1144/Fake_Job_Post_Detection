import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from pathlib import Path

from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

# Download stopwords
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

# =========================
# LOAD DATASET
# =========================

BASE_DIR = Path(__file__).resolve().parent
DATASET_FILE = BASE_DIR / 'fake_job_postings.csv'
MODEL_DIR = BASE_DIR / 'saved_model'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    df = pd.read_csv(DATASET_FILE, encoding='latin-1')
    print('Dataset loaded successfully')
    print('Dataset Shape:', df.shape)

except Exception as e:
    print('Error loading dataset:', e)
    exit()

# =========================
# DEFINE COLUMNS
# =========================

TEXT_COLUMNS = [
    'title',
    'company_profile',
    'description',
    'requirements',
    'benefits'
]

TARGET_COLUMN = 'fraudulent'

# Keep only existing columns
TEXT_COLUMNS = [col for col in TEXT_COLUMNS if col in df.columns]

# Check target column
if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

# =========================
# CLEAN DATA
# =========================

for col in TEXT_COLUMNS:
    df[col] = df[col].fillna('').astype(str)

# Combine text columns
df['combined_text'] = df[TEXT_COLUMNS].agg(' '.join, axis=1)

# Convert target labels
if df[TARGET_COLUMN].dtype == object:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(
        lambda x: 1 if str(x).strip().lower() in [
            '1',
            'true',
            'yes',
            'fraud',
            'fraudulent'
        ] else 0
    )

X = df['combined_text']
y = df[TARGET_COLUMN].astype(int)

# =========================
# SPLIT DATA
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print('Training Size:', len(X_train))
print('Testing Size:', len(X_test))

# =========================
# PIPELINE FUNCTION
# =========================


def build_pipeline(clf):
    return Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                lowercase=True,
                stop_words=list(STOPWORDS),
                max_df=0.85,
                min_df=3,
                ngram_range=(1, 2),
                max_features=5000
            )
        ),
        ('clf', clf)
    ])

# =========================
# LOGISTIC REGRESSION MODEL
# =========================

log_model = build_pipeline(
    LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )
)

log_model.fit(X_train, y_train)

# Predictions
log_preds = log_model.predict(X_test)

print('\n===== LOGISTIC REGRESSION RESULTS =====')
print('Accuracy:', accuracy_score(y_test, log_preds))
print(classification_report(y_test, log_preds))

# =========================
# RANDOM FOREST MODEL
# =========================

rf_model = build_pipeline(
    RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print('\n===== RANDOM FOREST RESULTS =====')
print('Accuracy:', accuracy_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))

# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_test, log_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
plt.close()

# =========================
# ROC CURVE
# =========================

log_probs = log_model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, log_probs)
roc_auc = roc_auc_score(y_test, log_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
plt.close()

# =========================
# SAVE MODEL
# =========================

joblib.dump(log_model, MODEL_DIR / 'fake_job_detector.joblib')
print('\nModel saved successfully to', MODEL_DIR / 'fake_job_detector.joblib')

# =========================
# TEST CUSTOM TEXT
# =========================

loaded_model = joblib.load('saved_model/fake_job_detector.joblib')

examples = [
    "We're looking for an entry-level software engineer with good salary and benefits.",
    "Earn $5000 weekly from home with no experience required."
]

predictions = loaded_model.predict(examples)
probabilities = loaded_model.predict_proba(examples)

for text, pred, prob in zip(examples, predictions, probabilities):
    print('\nTEXT:', text)
    print('Prediction:', 'Fraudulent Job' if pred == 1 else 'Real Job')
    print('Confidence:', round(max(prob) * 100, 2), '%')