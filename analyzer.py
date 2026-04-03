def analyze_text(text, sentiment_model, emotion_model, toxicity_model):
    sentiment = sentiment_model(text)[0]
    emotion = emotion_model(text)[0]
    toxicity = toxicity_model(text)[0]

    results = {
        "text": text,
        "sentiment": sentiment["label"],
        "sentiment_score": sentiment["score"],
        "emotion": emotion["label"],
        "emotion_score": emotion["score"],
        "toxicity": toxicity["label"],
        "toxicity_score": toxicity["score"]
    }

    return results