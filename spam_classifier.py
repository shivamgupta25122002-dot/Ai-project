"""
Spam Email Classifier in Python
--------------------------------
A machine learning system for classifying email messages as 'spam' or 'ham' (not spam).
Features:
 - Text preprocessing (lowercasing, cleaning, stemming)
 - TF-IDF Feature Extraction with unigrams and bigrams
 - Multiple Model Support (Multinomial Naive Bayes, Logistic Regression, Support Vector Classifier)
 - Comprehensive Evaluation Metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix)
 - Model Persistence (Save & Load trained models using joblib)
 - Interactive CLI & API interface
"""

import os
import re
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Preprocesses raw text messages for NLP feature extraction."""
    def __init__(self):
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"a", "an", "the", "in", "on", "at", "and", "or", "is", "to", "for", "of", "with", "this", "that", "you", "your"}

    def clean_text(self, text: str) -> str:
        """
        Cleans input string:
        - Lowercases
        - Normalizes URLs and Email Addresses
        - Removes punctuation and digits
        - Removes stopwords
        - Applies stemming
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        # Replace URLs with placeholder token
        text = re.sub(r'http[s]?://\S+|www\.\S+', ' urltoken ', text)
        # Replace email addresses with placeholder token
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' emailtoken ', text)
        # Replace currency symbols
        text = re.sub(r'[\$£€]', ' moneytoken ', text)
        # Remove special characters and digits, keep spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Tokenize and remove stopwords + apply stemming
        tokens = text.split()
        cleaned_tokens = [self.stemmer.stem(word) for word in tokens if word not in self.stop_words and len(word) > 1]
        
        return " ".join(cleaned_tokens)


class SpamClassifier:
    """Spam Email Classifier pipeline utilizing TF-IDF and Machine Learning."""

    def __init__(self, model_type: str = 'naive_bayes'):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        )
        self.model_type = model_type
        self.model = self._build_model(model_type)
        self.is_trained = False

    def _build_model(self, model_type: str):
        if model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        elif model_type == 'logistic_regression':
            return LogisticRegression(C=1.0, max_iter=1000)
        elif model_type == 'svm':
            return SVC(kernel='linear')
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Choose 'naive_bayes', 'logistic_regression', or 'svm'.")

    def fit(self, texts: list, labels: list):
        """Preprocesses dataset, fits vectorizer and trains the classifier model."""
        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        # Map labels to numeric if strings are passed ('spam' -> 1, 'ham' -> 0)
        y = np.array([1 if str(lbl).lower() == 'spam' else 0 for lbl in labels])
        
        self.model.fit(X, y)
        self.is_trained = True
        return self

    def evaluate(self, texts: list, labels: list) -> dict:
        """Evaluates trained classifier performance on test set."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before calling evaluate().")

        cleaned_texts = [self.preprocessor.clean_text(t) for t in texts]
        X_test = self.vectorizer.transform(cleaned_texts)
        y_true = np.array([1 if str(lbl).lower() == 'spam' else 0 for lbl in labels])
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])
        }
        return metrics

    def predict(self, text: str) -> dict:
        """
        Classifies a single message text.
        Returns label ('spam' or 'ham'), confidence score, and top word features.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call fit() or load_model() first.")

        cleaned = self.preprocessor.clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        
        prediction = self.model.predict(vectorized)[0]
        label = "spam" if prediction == 1 else "ham"
        
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(vectorized)[0]
            confidence = float(probabilities[prediction])
        elif hasattr(self.model, "decision_function"):
            # For linear SVM, convert decision function score to pseudo-probability via sigmoid
            df_val = self.model.decision_function(vectorized)[0]
            prob = 1.0 / (1.0 + np.exp(-df_val))
            confidence = float(prob if prediction == 1 else (1.0 - prob))
        else:
            confidence = 1.0

        # Extract top feature tokens found in input
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        non_zero_indices = vectorized.nonzero()[1]
        tfidf_scores = vectorized.data
        
        top_words = []
        if len(non_zero_indices) > 0:
            word_score_tuples = sorted(zip(feature_names[non_zero_indices], tfidf_scores), key=lambda x: x[1], reverse=True)
            top_words = [w[0] for w in word_score_tuples[:5]]

        return {
            "raw_text": text,
            "cleaned_text": cleaned,
            "label": label,
            "is_spam": bool(prediction == 1),
            "confidence": round(confidence, 4),
            "top_features": top_words
        }

    def save_model(self, model_dir: str = "."):
        """Saves vectorizer and trained model to disk."""
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"spam_model_{self.model_type}.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"[+] Saved model to {model_path} and vectorizer to {vectorizer_path}")

    def load_model(self, model_type: str = 'logistic_regression', model_dir: str = "."):
        """Loads vectorizer and trained model from disk."""
        model_path = os.path.join(model_dir, f"spam_model_{model_type}.pkl")
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Model files not found in {model_dir}")
            
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.model_type = model_type
        self.is_trained = True
        print(f"[+] Successfully loaded trained model from {model_path}")


def load_dataset(file_path: str = "dataset.csv") -> pd.DataFrame:
    """Loads CSV dataset containing 'label' and 'message' columns."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
    df = pd.read_csv(file_path)
    if 'label' not in df.columns or 'message' not in df.columns:
        raise ValueError("Dataset CSV must contain 'label' and 'message' columns.")
    return df


def main():
    print("=" * 60)
    print("          SPAM EMAIL CLASSIFIER - PYTHON SYSTEM")
    print("=" * 60)

    # 1. Load Dataset
    dataset_path = "dataset.csv"
    print(f"\n[1] Loading dataset from '{dataset_path}'...")
    df = load_dataset(dataset_path)
    print(f"    - Total Samples: {len(df)}")
    print(f"    - Class Distribution: {df['label'].value_counts().to_dict()}")

    # 2. Split Dataset into Train and Test
    X = df['message'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"    - Train Samples: {len(X_train)}, Test Samples: {len(X_test)}")

    # 3. Train Classifiers & Compare
    models_to_test = ['naive_bayes', 'logistic_regression', 'svm']
    classifiers = {}
    best_classifier = None
    best_f1 = -1.0

    print("\n[2] Training and Evaluating ML Algorithms...")
    print("-" * 60)

    for m_type in models_to_test:
        clf = SpamClassifier(model_type=m_type)
        clf.fit(X_train, y_train)
        metrics = clf.evaluate(X_test, y_test)
        classifiers[m_type] = (clf, metrics)

        print(f"\nModel: {m_type.upper().replace('_', ' ')}")
        print(f"  * Accuracy:  {metrics['accuracy'] * 100:.2f}%")
        print(f"  * Precision: {metrics['precision'] * 100:.2f}%")
        print(f"  * Recall:    {metrics['recall'] * 100:.2f}%")
        print(f"  * F1-Score:  {metrics['f1_score'] * 100:.2f}%")

        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_classifier = clf

    # 4. Save Best Model
    print(f"\n[3] Saving Best Performing Model ({best_classifier.model_type.upper()})...")
    best_classifier.save_model()

    # 5. Interactive Demo / Testing
    print("\n" + "=" * 60)
    print("                  CLASSIFICATION DEMO")
    print("=" * 60)

    sample_emails = [
        "Hey Shivam, let's schedule our project review call for tomorrow at 4 PM.",
        "URGENT! You have won $10,000 cash. Click here to claim your reward immediately: http://win-cash-now.com",
        "Your Amazon order #402-9981 has been shipped and will arrive on Thursday.",
        "Verify your online banking details now or your account will be permanently closed!"
    ]

    for email in sample_emails:
        res = best_classifier.predict(email)
        status_tag = "[SPAM]" if res['is_spam'] else "[HAM/NOT SPAM]"
        print(f"\nEmail Content: \"{res['raw_text']}\"")
        print(f"Result: {status_tag} (Confidence: {res['confidence']*100:.1f}%)")
        print(f"Keywords Identified: {', '.join(res['top_features']) if res['top_features'] else 'N/A'}")


if __name__ == "__main__":
    main()
