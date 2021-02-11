from lxml import html, etree
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

    return attributes, G


def none_to_empty(x):
    if x is None:
        return ""
    return x
