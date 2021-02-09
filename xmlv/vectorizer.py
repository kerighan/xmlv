from sklearn.feature_extraction.text import CountVectorizer


def fit_transform(series, min_df=1, tokenize=False):
    if not tokenize:
        vectorizer = CountVectorizer(
            min_df=min_df,
            preprocessor=lambda x: x, tokenizer=lambda x: x)
    else:
        vectorizer = CountVectorizer(min_df=min_df)

    X = vectorizer.fit_transform(series).todense()
    return X, vectorizer


def link_fit_transform(series, min_df=1):
    vectorizer = CountVectorizer(
        min_df=min_df,
        preprocessor=lambda x: x.split("/"), tokenizer=lambda x: x)
    X = vectorizer.fit_transform(series).todense()
    return X, vectorizer

