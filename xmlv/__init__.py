from io import StringIO

import numpy as np
import requests
from lxml import html
from rolewalk import RoleWalk

from .preprocessing import to_networkx
from .vectorizer import fit_transform, link_fit_transform

headers = {
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/webp,*/*;q=0.8"),
    "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
    "Connection": "keep-alive",
    "User-Agent": ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:73.0) "
                   "Gecko/20100101 Firefox/73.0"),
    "TE": "Trailers",
    "Cache-Control": "max-age=0"
}

class XMLV:
    def __init__(
        self,
        min_df=5,
        vectorize_text=False,
        vectorize_link=False,
        svd_text=100,
        svd_link=50,
        svd_property=50,
        attr_min_df=2,
        text_min_df=5,
        scale=False,
        use_structural=True,
        structural_dim=50,
        walk_len=3,
        tfidf=True,
        max_features = None
    ):
        self.vectorizers = {}
        self.min_df = min_df
        self.vectorize_text = vectorize_text
        self.vectorize_link = vectorize_link
        self.svd_text = svd_text
        self.svd_link = svd_link
        self.svd_property = svd_property
        self.attr_min_df = attr_min_df
        self.text_min_df = text_min_df
        self.robust_scaler = scale
        self.use_structural = use_structural
        self.structural_dim = structural_dim
        self.walk_len = walk_len
        self.tfidf = tfidf
        if max_features is None :
            self.max_features  = {"text": 10000, "id": 500, "tag":500, "class":500, "property":500, "href":500}
        else :
            self.max_features = max_features

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
        r = requests.get(url, headers=headers)
        return self.parse(r.text, url)

    def parse(self, text, url=None):
        tree = html.parse(StringIO(text))
        root = tree.getroot()
        attributes, G = to_networkx(root)
        G.url = url
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
            attributes[col] = attributes[col].apply(
                lambda x: [y for y in x if isinstance(y, str)] if isinstance(x, list) else x)
            x, vectorizer = fit_transform(
                attributes[col], min_df=self.attr_min_df, tfidf=self.tfidf, svd=self.svd_property, max_features = self.max_features[col])
            self.vectorizers[col] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_text:
            x, vectorizer = fit_transform(
                attributes["text"].fillna(""),
                min_df=self.text_min_df,
                tfidf=self.tfidf,
                svd=self.svd_text,
                tokenize=True,
                max_features = self.max_features["text"])
            self.vectorizers["text"] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_link:
            x, vectorizer = link_fit_transform(
                attributes["href"].fillna(""),
                svd=self.svd_link,
                min_df=self.attr_min_df, 
                max_features = self.max_features["href"])
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

        if G is not None and self.use_structural:
            rolewalk = RoleWalk(walk_len=self.walk_len,
                                embedding_dim=self.structural_dim)
            X_structural = rolewalk.transform(G)
            X.append(X_structural)

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

    def transform(self, attributes, G=None):
        # serialize data if needed
        for col in ["tag", "id", "class", "property"]:
            if isinstance(attributes[col].iloc[0], str):
                attributes[col] = attributes[col].apply(eval)
            attributes[col] = attributes[col].apply(
                lambda x: [y.lower() for y in x if isinstance(y, str)] if isinstance(x, list) else x.lower())

        # create feature vectors
        X = []
        for col, vectorizer in self.vectorizers.items():
            if col in ["text", "href"]:
                data = vectorizer.transform(
                    attributes[col].fillna(""))
            else:
                data = vectorizer.transform(attributes[col])
            try:
                data = data.todense()
            except AttributeError:
                pass
            X.append(data)

        if "position" not in attributes.columns:
            X.append(np.zeros((len(attributes), 1), dtype=np.float32))
        else:
            pos = attributes["position"].astype(np.float32).values
            np.nan_to_num(pos, copy=False)
            X.append(pos[:, None])

        if self.robust_scaler:
            X = np.hstack(X)
            X = self.scaler.transform(X)

            if G is not None and self.structural:
                rolewalk = RoleWalk(walk_len=self.walk_len,
                                    embedding_dim=self.structural_dim)
                X_2 = rolewalk.transform(G)
                X = np.hstack([X, X_2])
        else:
            if G is not None and self.use_structural:
                rolewalk = RoleWalk(walk_len=self.walk_len,
                                    embedding_dim=self.structural_dim)
                X.append(rolewalk.transform(G))
            X = np.hstack(X)
        return X

    def predict(self, attributes, G=None):
        X = self.vectorize(attributes)
        if G is None:
            # use sklearn classifier
            return self.clf.predict(X)
        else:
            # use GCN if graph is provided
            return self.clf.predict(G, X)

    def store(self, attr, G, filename):
        import os
        import pickle

        data = {"attr": attr, "G": G, "url": G.url}
        if not os.path.exists(filename):
            os.makedirs(filename)
        with open(f"{filename}/data.p", "wb") as f:
            pickle.dump(data, f)

        attr["content"] = attr["content"].apply(lambda x: str(x)[:15])
        attr["href"] = attr["href"].apply(lambda x: str(x)[:10])
        attr.to_csv(f"{filename}.csv")
