import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_metric_comparison(results):
    labels = [f"{r['model']}-{r['feature']}" for r in results]
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(12,6))
    plt.bar(x - 0.2, accs, width=0.4, label='Accuracy')
    plt.bar(x + 0.2, f1s, width=0.4, label='F1 Score')

    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.title("Model Performance Comparison")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_pca_variance(pca):
    cumulative = np.cumsum(pca.explained_variance_ratio_)

    plt.figure()
    plt.plot(cumulative)
    plt.title("Cumulative Explained Variance")
    plt.xlabel("Components")
    plt.ylabel("Variance")
    plt.show()


def plot_accuracy_vs_components(components, scores, title):
    plt.figure()
    plt.plot(components, scores, marker='o')
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")
    plt.show()

def plot_eda(texts, labels, title="Exploratory Data Analysis"):
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Class Distribution
    plt.subplot(1, 2, 1)
    unique_classes, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_classes, counts, color='skyblue', edgecolor='black')
    plt.title("Class Distribution in Training Set")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.xticks(unique_classes)
    
    # Plot 2: Text Length Distribution
    plt.subplot(1, 2, 2)
    lengths = [len(str(t).split()) for t in texts]
    plt.hist(lengths, bins=40, color='lightcoral', edgecolor='black')
    plt.title("Text Length Distribution (Word Count)")
    plt.xlabel("Number of Words per Article")
    plt.ylabel("Frequency")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
