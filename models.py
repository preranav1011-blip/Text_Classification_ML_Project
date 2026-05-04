from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

def get_models():
    return {
        "LogReg": LogisticRegression(max_iter=1000),
        "SVM-Linear": SVC(kernel='linear'),
        "SVM-RBF": SVC(kernel='rbf'),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "KMeans": KMeans(n_clusters=4, random_state=42)
    }