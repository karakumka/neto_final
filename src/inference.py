import joblib
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from scipy.sparse import csr_matrix, hstack
from sentence_transformers import SentenceTransformer


SBERT_MODEL_PATH = "models/model_sbert"
CATBOOST_MODEL_PATH = "models/model_classification.cbm"
OHE_MODEL_PATH = "models/model_ohe.pkl"


def load_models():

    model_sbert = SentenceTransformer(str(SBERT_MODEL_PATH))
    model_ohe = joblib.load(OHE_MODEL_PATH)
    model_classifier = CatBoostClassifier()
    model_classifier.load_model(str(CATBOOST_MODEL_PATH))

    return model_classifier, model_ohe, model_sbert


def build_features(data: pd.DataFrame, model_ohe, model_sbert):
    text_embeddings = model_sbert.encode(data["text"].astype(str).tolist(), batch_size=32, 
                                         show_progress_bar=True, convert_to_numpy=True)

    company_features = model_ohe.transform(data[["company_name"]])
    features = hstack([csr_matrix(text_embeddings), company_features])

    return features


def predict_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    
    model_classifier, model_ohe, model_sbert = load_models()
    features = build_features(data, model_ohe, model_sbert)

    predictions = model_classifier.predict(features).ravel()
    probabilities = model_classifier.predict_proba(features)

    result = data.copy()
    result["predicted_target"] = predictions

    target_to_sentiment = {-1: "negative", 0: "neutral", 1: "positive"}
    result["predicted_sentiment"] = result["predicted_target"].map(target_to_sentiment)

    classes = list(model_classifier.classes_)

    for class_label in classes:
        sentiment_name = target_to_sentiment[class_label]
        class_index = classes.index(class_label)
        result[f"proba_{sentiment_name}"] = probabilities[:, class_index]

    return result