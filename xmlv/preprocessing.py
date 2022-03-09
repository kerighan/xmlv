from lxml import html, etree
from lxml.etree import tostring
import networkx as nx
import pandas as pd
import re


def get_attributes(element_id, element):
    class_ = none_to_empty(element.get("class", "_"))
    class_ = list(set(
        elem.lower() for elem in re.split(r"[\s₋_]+", class_) if len(elem) != 0))

    ppty = none_to_empty(element.get("property", "_"))
    ppty = ppty.split(":")
    ppty = [elem.lower() for elem in ppty if len(elem) > 0]

    text = tostring(element,
                    method="text",
                    encoding="utf8").decode("utf8").strip()
    return [
        element_id,
        [element.tag],
        none_to_empty(element.get("id", "_")).split(),
        class_,
        text,
        ppty,
        element.get("content", ""),
        element.get("href", "")
    ]


def to_networkx(root):
    # list all html elements
    elements = sorted(list(set([
        elem
        for elem in root.iter()
        if not isinstance(elem, html.HtmlComment)
    ])), key=str)

    # build attributes from lxml
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

    # create html graph
    G = nx.Graph()
    G.add_nodes_from(attributes["index"].tolist())
    edges = []
    id2pos = {}
    for source in elements:
        source_id = str(source)

        children = source.getchildren()
        n_children = len(children)
        for i, target in enumerate(children):
            target_id = str(target)
            # add positional indexing
            if n_children == 1:
                id2pos[target_id] = 0
            else:
                id2pos[target_id] = round(i / (n_children - 1), 2)

            if target_id in elements_id and source_id in elements_id:
                edges.append((source_id, target_id))
        
        parent = source.getparent()
        parent_id = str(parent)
        if parent_id in elements_id and source_id in elements_id:
            edges.append((parent_id, source_id))
    G.add_edges_from(set(edges))
    # add positional indexing
    attributes["position"] = attributes["index"].map(id2pos)
    return attributes, G


def none_to_empty(x):
    if x is None:
        return ""
    return x
