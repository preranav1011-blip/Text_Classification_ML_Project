import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import os

# DEFINE THESE HERE
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()


def get_embeddings(texts, save_path=None):

    # create folder if not exists
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # load if already saved
    if save_path and os.path.exists(save_path):
        return np.load(save_path)

    embeddings = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        cls = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls)

    embeddings = np.array(embeddings)

    if save_path:
        np.save(save_path, embeddings)

    return embeddings
