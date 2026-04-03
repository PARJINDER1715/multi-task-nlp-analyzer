import matplotlib.pyplot as plt

def plot_confidence(scores_dict):
    labels = ["Sentiment", "Emotion", "Toxicity"]
    scores = [
        scores_dict["sentiment_score"],
        scores_dict["emotion_score"],
        scores_dict["toxicity_score"]
    ]

    fig, ax = plt.subplots()
    ax.bar(labels, scores)
    ax.set_title("Model Confidence Scores")
    ax.set_ylabel("Confidence")
    
    return fig