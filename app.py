from flask import Flask, request, jsonify
from transformers import pipeline
import re
from langdetect import detect

app = Flask(__name__)

# Initialize the sentiment analysis pipeline for Arabic
sentiment_pipeline_ar = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment"
)

INSULTS_ar = [
    "غبي", "غبية", "مقرف", "مقرفة", "فاشل", "فاشلة", "كلب", "حمار", "حقير", "حقيرة",
    "سخيف", "سخيفة", "مغفل", "مغفلة", "زبالة", "وسخ", "وسخة", "خول"
]

LAUGHTER_PATTERN_ar = r"خ{3,}"

# Initialize the sentiment analysis pipeline for English
sentiment_pipeline_en = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

INSULTS_en = [
    "stupid", "idiot", "jerk", "loser", "fool", "moron", "trash", "disgusting",
    "pathetic", "dumb", "asshole", "shitty"
]

LAUGHTER_PATTERN_en = r"(?i)\b(lol+|haha+|lmao+|rofl+)\b"

def map_sentiment_to_value(sentiment, confidence):
    """Map sentiment and confidence to a value in the specified range."""
    if sentiment == "positive" or sentiment == "POSITIVE":
        # Map confidence [0,1] to [7.1, 10]
        value = 7.1 + (10 - 7.1) * confidence
    elif sentiment == "negative" or sentiment == "NEGATIVE":
        # Map confidence [0,1] to [4, 1] (high confidence -> 1, low confidence -> 4)
        value = 4 - (4 - 0.5) * confidence
    else:  # neutral
        # Map confidence [0,1] to [4.1, 7]
        value = 4.1 + (7 - 4.1) * confidence
    return round(value, 2)

def analyze_sentiment_ar(text):
    has_insult = any(insult in text for insult in INSULTS_ar)
    has_laughter = bool(re.search(LAUGHTER_PATTERN_ar, text))

    response = sentiment_pipeline_ar(text)
    sentiment = response[0]['label']
    confidence = response[0]['score']

    if has_insult:
        sentiment = "negative"
        confidence = 1.0
    elif has_laughter and sentiment in ["neutral", "negative"]:
        sentiment = "negative"
        confidence = max(confidence, 0.9)

    value = map_sentiment_to_value(sentiment, confidence)
    return sentiment, confidence, value

def analyze_sentiment_en(text):
    has_insult = any(insult.lower() in text.lower() for insult in INSULTS_en)
    has_laughter = bool(re.search(LAUGHTER_PATTERN_en, text, re.IGNORECASE))

    response = sentiment_pipeline_en(text)
    sentiment = response[0]['label']
    confidence = response[0]['score']

    if has_insult:
        sentiment = "NEGATIVE"
        confidence = 0.8
    elif has_laughter and sentiment == "NEGATIVE":
        sentiment = "NEGATIVE"
        confidence = max(confidence, 0.7)

    value = map_sentiment_to_value(sentiment, confidence)
    return sentiment, confidence, value

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    text = data['text']
    language = detect(text)
    try:
        if language == 'ar':
            sentiment, confidence, value = analyze_sentiment_ar(text)
        else:
            sentiment, confidence, value = analyze_sentiment_en(text)
        return jsonify({
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "value": value
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)