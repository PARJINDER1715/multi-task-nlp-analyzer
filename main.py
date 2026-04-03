from models.model_loader import load_models
from utils.analyzer import analyze_text

if __name__ == "__main__":
    sentiment_model, emotion_model, toxicity_model = load_models()

    text = input("Enter text: ")
    result = analyze_text(text, sentiment_model, emotion_model, toxicity_model)

    print("\nAnalysis Result:")
    print(result)