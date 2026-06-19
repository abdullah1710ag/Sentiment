import html
import os
import re
import emoji
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================================================
# الـ القواميس والثوابت بتاعتك (سيبها زي ما هي)
# =========================================================
CHAT_WORDS = {
    "u": "you", "ur": "your", "idk": "i do not know", "imo": "in my opinion",
    "imho": "in my humble opinion", "omg": "oh my god", "btw": "by the way",
    "asap": "as soon as possible", "fyi": "for your information", "lol": "laughing",
    "im": "i am", "tmi": "too much information", "r": "are",
}

EMOJI_MAP = {
    "😂": " face_with_tears_of_joy ", "😭": " loudly_crying_face ",
    "😡": " angry_face ", "🤬": " very_angry_face ", "🔥": " fire ",
    "💀": " skull ", "🖕": " middle_finger ",
}

KEEP_EMOJIS = {"😂", "😭", "😡", "🤬", "🔥", "💀", "🖕", "❤️", "❤"}

PROFANITY_PATTERNS = {
    r"f\*+ck": "fuck", r"f\*+king": "fucking", r"sh\*+t": "shit",
    r"b\*+tch": "bitch", r"a\*+hole": "asshole",
}

# =========================================================
# دالات التنظيف والـ Preprocessing
# =========================================================
def split_hashtag(tag):
    tag = tag.replace("#", "")
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)

def replace_chat_words(text):
    words = text.split()
    new_words = [CHAT_WORDS[w.lower()] if w.lower() in CHAT_WORDS else w for w in words]
    return " ".join(new_words)

def process_emojis(text):
    result = []
    for ch in text:
        if ch in EMOJI_MAP:
            result.append(EMOJI_MAP[ch])
        elif ch in KEEP_EMOJIS:
            result.append(ch)
        elif ch in emoji.EMOJI_DATA:
            continue
        else:
            result.append(ch)
    return "".join(result)

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"@\w+", " ", text)
    hashtags = re.findall(r"#\w+", text)
    for h in hashtags:
        text = text.replace(h, split_hashtag(h))
    text = replace_chat_words(text)
    for pattern, repl in PROFANITY_PATTERNS.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = process_emojis(text)
    text = re.sub(r'[\u200b-\u200d\uFEFF]', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'([!?])\1{3,}', r'\1\1\1', text)
    text = re.sub(r"[^\w\s!?.,'_:;()\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()