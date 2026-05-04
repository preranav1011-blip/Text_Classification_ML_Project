from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow(train, test, max_features):
    vec = CountVectorizer(max_features=max_features)
    return vec.fit_transform(train), vec.transform(test)

def get_tfidf(train, test, max_features):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=max_features)
    return vec.fit_transform(train), vec.transform(test)