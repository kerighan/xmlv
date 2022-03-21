import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def fit_transform(series, min_df=1, tfidf=False, tokenize=False, svd=None):
    if not tokenize:
        if tfidf:
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x)
        else:
            vectorizer = CountVectorizer(
                min_df=min_df,
                preprocessor=lambda x: x,
                tokenizer=lambda x: x)
    else:
        if tfidf:
            vectorizer = TfidfVectorizer(min_df=min_df)
        else:
            vectorizer = CountVectorizer(min_df=min_df)
    if svd is not None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        vectorizer = Pipeline(
            [("vectorizer", vectorizer),
             ("svd", TruncatedSVD(n_components=svd))])

    X = vectorizer.fit_transform(series)
    try:
        X = X.todense()
    except AttributeError:
        pass
    return X, vectorizer


def link_fit_transform(series, tfidf=False, svd=None, min_df=1):
    if tfidf:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            tokenizer=lambda x: re.split(r'[\/\_\-\?\.\:]', x),
            preprocessor=lambda x: x)
    else:
        vectorizer = CountVectorizer(
            min_df=min_df,
            tokenizer=lambda x: re.split(r'[\/\_\-\?\.\:]', x),
            preprocessor=lambda x: x)
    if svd is not None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.pipeline import Pipeline
        vectorizer = Pipeline(
            [("vectorizer", vectorizer),
             ("svd", TruncatedSVD(n_components=svd))])
    X = vectorizer.fit_transform(series)
    try:
        X = X.todense()
    except AttributeError:
        pass
    return X, vectorizer
