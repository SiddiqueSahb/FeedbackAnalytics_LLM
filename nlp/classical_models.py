"""
Two models trained with TF-IDF features:
  1. Logistic Regression (LR)  — strong classical baseline
  2. Naive Bayes (NB)          — simple probabilistic baseline

These represent pre-transformer NLP approaches (pre-2018).
They are trained on your dataset via: python scripts/train_classical_models.py
Saved to: models/classical/
"""

from __future__ import annotations

import os
import sys
import pickle
import re
from typing import Any

# Add the project root directory to Python's module search path for importing custom packages
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#import logger
from logger import get_logger
log = get_logger("nlp.classical")

#Model store path
MODEL_DIR = "models/classical"
LR_PATH   = os.path.join(MODEL_DIR, "logistic_regression.pkl")
NB_PATH   = os.path.join(MODEL_DIR, "naive_bayes.pkl")
VEC_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

#create label mapping for sentiments
LABEL_MAP: dict[int,str] = {0:"negative",1:"neutral",2:"positive"}

#function to clean input text 
def _clean(text: str) -> str:
    """
    Clean and normalise input text using the same preprocessing
    pipeline applied during model training.

    Steps:
    1. Remove URLs.
    2. Remove user mentions.
    3. Remove the '#' symbol while retaining hashtag words.
    4. Remove HTML tags.
    5. Convert emojis into descriptive text.
    6. Replace underscores with spaces.
    7. Remove non-ASCII characters.
    8. Normalize whitespace.
    9. Convert text to lowercase.
    """

    # Convert the input into a string
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove user mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove the '#' symbol while retaining the hashtag word
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Convert emojis into descriptive text (if emoji package is installed)
    try:
        import emoji
        text = emoji.demojize(text, delimiters=(" ", " "))
    except ImportError:
        pass

    # Replace underscores in emoji names with spaces
    text = text.replace("_", " ")

    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Replace multiple spaces with a single space and trim whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Convert the cleaned text to lowercase
    return text.lower()

#load trained model i.e pickle file
def _load(path:str) -> Any:
    """Deserialise a pickle file and return its contents."""
    with open(path, "rb") as f:
        return pickle.load(f)

# Logistic Regression 
class LogisticRegressionSentiment:
    """
    TF-IDF + Logistic Regression sentiment classifier.
    Classical ML baseline from the pre-transformer era.
    Train with: python scripts/train_classical_models.py
    """

    def __init__(self) -> None:
        # Check whether the trained model and TF-IDF vectorizer exist
        if not os.path.exists(LR_PATH) or not os.path.exists(VEC_PATH):
            log.warning("LR model not found. Run: python scripts/train_classical_models.py")
            self._loaded = False
            return

        # Load the trained TF-IDF vectorizer and Logistic Regression model
        self.vectorizer: Any = _load(VEC_PATH)
        self.model: Any = _load(LR_PATH)

        # Mark the model as successfully loaded
        self._loaded = True
        log.info("Logistic Regression model loaded from %s", LR_PATH)

    def predict(self, text: str) -> dict[str, object]:
        """
        Predict the sentiment of an input review.

        Returns:
            Dictionary containing:
            - predicted sentiment label
            - prediction confidence
            - class probabilities
            - model name
        """

        # Return an informative response if the model has not been trained
        if not self._loaded:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "scores": {
                    "positive": 0.0,
                    "neutral": 0.0,
                    "negative": 0.0,
                },
                "model": "logistic_regression",
                "error": "Model not trained. Run: python scripts/train_classical_models.py",
            }

        # Apply the same preprocessing used during model training
        cleaned: str = _clean(text)

        # Convert the cleaned text into TF-IDF feature vectors
        vec = self.vectorizer.transform([cleaned])

        # Predict class probabilities and sentiment label
        probs: list[float] = list(self.model.predict_proba(vec)[0])
        pred_id: int = int(self.model.predict(vec)[0])
        classes: list[int] = list(self.model.classes_)

        # Map class IDs to sentiment labels with their probabilities
        score_map: dict[str, float] = {
            LABEL_MAP.get(c, str(c)): round(float(p), 4)
            for c, p in zip(classes, probs)
        }

        # Ensure all sentiment classes are included in the output
        for k in ["positive", "neutral", "negative"]:
            score_map.setdefault(k, 0.0)

        # Convert the predicted class ID into its sentiment label
        label: str = LABEL_MAP.get(pred_id, str(pred_id))

        # Return the prediction results
        return {
            "label": label,
            "confidence": round(float(max(probs)), 4),
            "scores": score_map,
            "model": "logistic_regression",
        }


# Naive Bayes
class NaiveBayesSentiment:
    """
    TF-IDF + Multinomial Naive Bayes sentiment classifier.
    Simple probabilistic baseline.
    Train with: python scripts/train_classical_models.py
    """

    def __init__(self) -> None:
        # Check whether the trained model and TF-IDF vectorizer exist
        if not os.path.exists(NB_PATH) or not os.path.exists(VEC_PATH):
            log.warning("NB model not found. Run: python scripts/train_classical_models.py")
            self._loaded = False
            return

        # Load the trained TF-IDF vectorizer and Naive Bayes model
        self.vectorizer: Any = _load(VEC_PATH)
        self.model: Any = _load(NB_PATH)

        # Mark the model as successfully loaded
        self._loaded = True
        log.info("Naive Bayes model loaded from %s", NB_PATH)

    def predict(self, text: str) -> dict[str, object]:
        """
        Predict the sentiment of an input review.

        Returns:
            Dictionary containing:
            - predicted sentiment label
            - prediction confidence
            - class probabilities
            - model name
        """

        # Return an informative response if the model has not been trained
        if not self._loaded:
            return {
                "label": "unknown",
                "confidence": 0.0,
                "scores": {
                    "positive": 0.0,
                    "neutral": 0.0,
                    "negative": 0.0,
                },
                "model": "naive_bayes",
                "error": "Model not trained. Run: python scripts/train_classical_models.py",
            }

        # Apply the same preprocessing used during model training
        cleaned: str = _clean(text)

        # Convert the cleaned text into TF-IDF feature vectors
        vec = self.vectorizer.transform([cleaned])

        # Predict class probabilities and sentiment label
        probs: list[float] = list(self.model.predict_proba(vec)[0])
        pred_id: int = int(self.model.predict(vec)[0])
        classes: list[int] = list(self.model.classes_)

        # Map class IDs to sentiment labels with their probabilities
        score_map: dict[str, float] = {
            LABEL_MAP.get(c, str(c)): round(float(p), 4)
            for c, p in zip(classes, probs)
        }

        # Ensure all sentiment classes are included in the output
        for k in ["positive", "neutral", "negative"]:
            score_map.setdefault(k, 0.0)

        # Convert the predicted class ID into its sentiment label
        label: str = LABEL_MAP.get(pred_id, str(pred_id))

        # Return the prediction results
        return {
            "label": label,
            "confidence": round(float(max(probs)), 4),
            "scores": score_map,
            "model": "naive_bayes",
        }