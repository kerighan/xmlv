import re
import requests
import pickle
import pandas as pd
from lxml import html, etree
import networkx as nx
from io import StringIO
import numpy as np
from .vectorizer import fit_transform, link_fit_transform
from sklearn.feature_extraction.text import CountVectorizer


class XMLV:
    def __init__(
        self,
        min_df=5,
        vectorize_text=False,
        vectorize_link=False,
        attr_min_df=2
    ):
        self.vectorizers = {}
        self.min_df = min_df
        self.vectorize_text = vectorize_text
        self.vectorize_link = vectorize_link
        self.attr_min_df = attr_min_df
    
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

    def get(self, url):
        r = requests.get(url)
        tree = html.parse(StringIO(r.text))
        root = tree.getroot()
        attributes, G = to_networkx(root)
        return attributes, G
    
    def fit_transform(self, attributes, target="category"):
        X = []
        # vectorize tag, class and property
        for col in ["tag", "class", "property"]:
            x, vectorizer = fit_transform(
                attributes[col], min_df=self.attr_min_df)
            self.vectorizers[col] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_text:
            x, vectorizer = fit_transform(
                attributes["text"],
                min_df=self.min_df,
                tokenize=True)
            self.vectorizers["text"] = vectorizer
            X.append(x)

        # vectorize text if necessary
        if self.vectorize_link:
            x, vectorizer = link_fit_transform(
                attributes["href"],
                min_df=self.attr_min_df)
            self.vectorizers["href"] = vectorizer
            X.append(x)

        X = np.concatenate(X, axis=-1)
        
        if target in attributes.columns:
            categories = set(attributes[target])
            n_categories = len(categories)
            category2id = {
                category: i
                for i, category in enumerate(categories)
            }

            Y = []
            for category in attributes[target]:
                vec = np.zeros((n_categories,), dtype=np.bool_)
                vec[category2id[category]] = 1
                Y.append(vec)
            Y = np.array(Y)
            return X, Y
        return X
    
    def vectorize(self, attributes):
        X = []
        for col, vectorizer in self.vectorizers.items():
            data = vectorizer.transform(attributes[col]).todense()
            X.append(data)
        X = np.concatenate(X, axis=-1)
        return X


def get_attributes(element_id, element):
    class_ = none_to_empty(element.get("class", "_"))
    class_ = list(set(
        elem.lower() for elem in re.split(r"[\s₋_]+", class_) if len(elem) != 0))

    ppty = none_to_empty(element.get("property", "_"))
    ppty = ppty.split(":")
    ppty = [elem.lower() for elem in ppty if len(elem) > 0]

    return [
        element_id,
        [element.tag],
        none_to_empty(element.get("id", "_")).split(),
        class_,
        element.text_content().strip()[:20],
        ppty,
        element.get("content", ""),
        element.get("href", "")
    ]


def to_networkx(root):
    elements = set([
        elem
        for elem in root.iter()
        if not isinstance(elem, html.HtmlComment)
    ])
    elements_id = set()

    attributes = []
    for elem in elements:
        element_id = str(elem)
        elements_id.add(element_id)
        attributes.append(get_attributes(element_id, elem))
    attributes = pd.DataFrame(attributes)
    attributes.columns = [
        "index", "tag", "id",
        "class", "text", "property",
        "content", "href"]

    G = nx.Graph()
    G.add_nodes_from(elements_id)
    edges = []
    for source in elements:
        source_id = str(source)
        for target in source.getchildren():
            target_id = str(target)
            if target_id in elements_id and source_id in elements_id:
                edges.append((source_id, target_id))
        
        parent = source.getparent()
        parent_id = str(parent)
        if parent_id in elements_id and source_id in elements_id:
            edges.append((parent_id, source_id))
    G.add_edges_from(set(edges))

    # neighbors = {}
    # for node in G.nodes:
    #     neighbors[node] = list(G.neighbors(node))
    # return attributes, neighbors
    return attributes, G


def none_to_empty(x):
    if x is None:
        return ""
    return x
