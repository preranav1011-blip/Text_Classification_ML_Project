import numpy as np
from datasets import load_dataset

from config import *
from preprocessing import preprocess
from sparse_features import get_tfidf
from sparse_features import get_bow
from embeddings import get_embeddings
from models import get_models
from evaluation import evaluate
from reduction import apply_svd, apply_pca
#from plots import plot_accuracy, plot_variance
from plots import (
    plot_metric_comparison,
    plot_confusion,
    plot_pca_variance,
    plot_accuracy_vs_components,
    plot_eda
)

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

train_texts = dataset['train']['text'][:TRAIN_SIZE]
train_labels = np.array(dataset['train']['label'][:TRAIN_SIZE])

test_texts = dataset['test']['text'][:TEST_SIZE]
test_labels = np.array(dataset['test']['label'][:TEST_SIZE])

print("Running Exploratory Data Analysis...")
plot_eda(train_texts, train_labels)

print("Preprocessing...")
train_texts = preprocess(train_texts)
test_texts = preprocess(test_texts)

# =========================
# TF-IDF
# =========================
print("TF-IDF...")
X_train_tfidf, X_test_tfidf = get_tfidf(train_texts, test_texts, TFIDF_MAX_FEATURES)

# BoW
print("Bag of Words...")
X_train_bow, X_test_bow = get_bow(train_texts, test_texts, TFIDF_MAX_FEATURES)

# =========================
# BERT
# =========================
print("BERT embeddings...")
X_train_bert = get_embeddings(train_texts[:BERT_TRAIN_SIZE], "embeddings_cache/train.npy")
X_test_bert = get_embeddings(test_texts[:BERT_TEST_SIZE], "embeddings_cache/test.npy")

y_train_small = train_labels[:BERT_TRAIN_SIZE]
y_test_small = test_labels[:BERT_TEST_SIZE]

# =========================
# TRAIN MODELS
# =========================
results = []
models = get_models()

# def run_all(X_train, X_test, y_train, y_test, name):
#     for model_name, model in models.items():
#         model.fit(X_train, y_train)
#         acc, f1 = evaluate(model, X_test, y_test)
#         results.append((model_name, name, acc, f1))

def run_all(X_train, X_test, y_train, y_test, feature_name):
    models = get_models()

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        acc, f1, prec, rec, report, preds = evaluate(model, X_test, y_test)

        results.append({
            "model": model_name,
            "feature": feature_name,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        })

        print(f"\n{model_name} - {feature_name}")
        print(report)

        plot_confusion(y_test, preds, f"{model_name} - {feature_name}")

print("Training TF-IDF models...")
run_all(X_train_tfidf, X_test_tfidf, train_labels, test_labels, "TF-IDF")

print("Training BoW models...")
run_all(X_train_bow, X_test_bow, train_labels, test_labels, "BoW")

print("Training BERT models...")
run_all(X_train_bert, X_test_bert, y_train_small, y_test_small, "BERT")

# =========================
# DIM REDUCTION
# =========================
print("Applying SVD...")
X_train_svd, X_test_svd, _ = apply_svd(X_train_tfidf, X_test_tfidf, N_COMPONENTS)
run_all(X_train_svd, X_test_svd, train_labels, test_labels, "TF-IDF+SVD")

# print("Applying PCA...")
# X_train_pca, X_test_pca, pca = apply_pca(X_train_bert, X_test_bert, N_COMPONENTS)
# run_all(X_train_pca, X_test_pca, y_train_small, y_test_small, "BERT+PCA")

print("\n=== PCA ANALYSIS ===")

components_list = [10, 50, 100, 200]
pca_scores = []

for n in components_list:
    print(f"\nPCA with {n} components")

    X_train_pca, X_test_pca, pca = apply_pca(X_train_bert, X_test_bert, n)

    # Train ONE model for analysis (LogReg recommended)
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000)

    model.fit(X_train_pca, y_train_small)

    acc, f1, prec, rec, report, preds = evaluate(model, X_test_pca, y_test_small)

    print(report)

    pca_scores.append(acc)

# Plot performance vs components
plot_accuracy_vs_components(components_list, pca_scores, "PCA Components vs Accuracy")

# Plot variance
plot_pca_variance(pca)


# =========================
# PLOTS
# =========================
plot_metric_comparison(results)
plot_pca_variance(pca)
