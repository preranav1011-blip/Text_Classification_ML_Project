from sklearn.decomposition import TruncatedSVD, PCA

def apply_svd(X_train, X_test, n_components):
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(X_train), svd.transform(X_test), svd

def apply_pca(X_train, X_test, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X_train), pca.transform(X_test), pca