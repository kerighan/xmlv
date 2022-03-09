from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re


def fit_transform(series, min_df=1, tfidf=False, tokenize=False):
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
    X = vectorizer.fit_transform(series).todense()
    return X, vectorizer


def link_fit_transform(series, tfidf=False, min_df=1):
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
    X = vectorizer.fit_transform(series).todense()
    return X, vectorizer
