from io import StringIO

import numpy as np
import requests
from lxml import html

from .preprocessing import to_networkx
from .vectorizer import fit_transform, link_fit_transform


class XMLV:
    def __init__(
        self,
        min_df=5,
        vectorize_text=False,
        vectorize_link=False,
        attr_min_df=2,
        text_min_df=5,
        scale=False,
        structural=True,
        structural_dim=50,
        walk_len=3,
        tfidf=True
    ):
        self.vectorizers = {}
        self.min_df = min_df
        self.vectorize_text = vectorize_text
        self.vectorize_link = vectorize_link
        self.attr_min_df = attr_min_df
        self.text_min_df = text_min_df
        self.robust_scaler = scale
        self.structural = structural
        self.structural_dim = structural_dim
        self.walk_len = walk_len
        self.tfidf = tfidf

    # =========================================================================
    # model persistence
    # =========================================================================

    def save(self, filename):
        import dill
        with open(filename, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def load(filename):
        import dill
        with open(filename, "rb") as f:
            xmlv = dill.load(f)
        return xmlv

    # =========================================================================
    # utils
    # =========================================================================

    def get(self, url):
        """retrieve attributes and graph from url"""
        r = requests.get(url)
        tree = html.parse(StringIO(r.text))
        root = tree.getroot()
        attributes, G = to_networkx(root)
        return attributes, G

    # =========================================================================
    # training and inference
    # =========================================================================

    def add_classifier(self, clf):
        self.clf = clf

    def fit_transform(self, attributes, G=None, target=None):
        X = []
        # vectorize tag, class and property
        for col in ["tag", "id", "class", "property"]:
            if isinstance(attributes[col].iloc[0], str):
                attributes[col] = attributes[col].apply(eval)
            x, vectorizer = fit_transform(
                attributes[col], min_df=self.attr_min_df, tfidf=self.tfidf)
            self.vectorizers[col] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_text:
            x, vectorizer = fit_transform(
                attributes["text"].fillna(""),
                min_df=self.text_min_df,
                tfidf=self.tfidf,
                tokenize=True)
            self.vectorizers["text"] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_link:
            x, vectorizer = link_fit_transform(
                attributes["href"].fillna(""),
                min_df=self.attr_min_df)
            self.vectorizers["href"] = vectorizer
            X.append(x)

        if "position" not in attributes.columns:
            X.append(np.zeros((len(attributes), 1,), dtype=np.float32))
        else:
            pos = attributes["position"].apply(
                lambda x: x if isinstance(x, float)
                else float(x.replace(",", "."))).values
            np.nan_to_num(pos, copy=False)
            X.append(pos[:, None])

        if G is not None and self.structural:
            from rolewalk import rolewalk
            X.append(rolewalk(G, walk_len=self.walk_len,
                              dim=self.structural_dim))

        # concatenate all features vectors
        X = np.concatenate(X, axis=-1)

        if target in attributes.columns:
            # create category mapping
            categories = set(attributes[target])
            category2id = {
                category: i
                for i, category in enumerate(categories)
            }

            # create sparse categorical target
            Y = []
            for category in attributes[target]:
                Y.append(category2id[category])
            Y = np.array(Y)

            # robustscale if needed
            if self.robust_scaler:
                from sklearn.preprocessing import RobustScaler
                self.scaler = RobustScaler()
                X = self.scaler.fit_transform(X, Y)

            # fit classifier if exists
            if hasattr(self, "clf"):
                print("Training classifier")
                if G is None:
                    self.clf.fit(X, Y)
                    print("accuracy: ", self.clf.score(X, Y))
                else:
                    self.clf.fit(G, X, Y)
            return X, Y

        # robustscale if needed
        if self.robust_scaler:
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler(unit_variance=False)
            X = self.scaler.fit_transform(X)
        return X

    def vectorize(self, attributes):
        # serialize data if needed
        for col in ["tag", "id", "class", "property"]:
            if isinstance(attributes[col].iloc[0], str):
                attributes[col] = attributes[col].apply(eval)

        # create feature vectors
        X = []
        for col, vectorizer in self.vectorizers.items():
            if col in ["text", "href"]:
                data = vectorizer.transform(
                    attributes[col].fillna("")).todense()
            else:
                data = vectorizer.transform(attributes[col]).todense()
            X.append(data)

        if "position" not in attributes.columns:
            X.append(np.zeros((len(attributes), 1), dtype=np.float32))
        else:
            pos = attributes["position"].astype(np.float32).values
            np.nan_to_num(pos, copy=False)
            X.append(pos[:, None])

        X = np.concatenate(X, axis=-1)

        if self.robust_scaler:
            X = self.scaler.transform(X)
        return X

    def predict(self, attributes, G=None):
        X = self.vectorize(attributes)
        if G is None:
            # use sklearn classifier
            return self.clf.predict(X)
        else:
            # use GCN if graph is provided
            return self.clf.predict(G, X)
