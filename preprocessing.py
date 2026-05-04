from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(texts):
    processed = []
    for text in texts:
        words = text.lower().split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        processed.append(" ".join(words))
    return processed