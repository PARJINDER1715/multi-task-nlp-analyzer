from transformers import pipeline
import torch

# Device configuration
device = 0 if torch.cuda.is_available() else -1

def load_models():
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    emotion_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=device
    )

    toxicity_model = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        device=device
    )

    return sentiment_model, emotion_model, toxicity_model