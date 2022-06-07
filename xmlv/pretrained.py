import os

models_dir = os.path.join(os.path.dirname(__file__), f"models")


class Article:
    def __init__(self):
        from convectors import load_model

        from xmlv import XMLV
        self.xmlv = XMLV.load(os.path.join(models_dir, "article/xmlv.p"))
        self.model = load_model(os.path.join(models_dir, "article/mlp.p"))

    def get(self, url):
        # from dateparser import parse

        attr, G = self.xmlv.get(url)
        X = self.xmlv.transform(attr, G)
        attr["out"] = self.model(X)
        attr["content"] = attr.apply(
            lambda x: x["text"] if len(x["content"]) == 0 else x["content"],
            axis=1)

        attr = attr.sort_values("position", ascending=True)
        content = "\n".join(attr[attr["out"] == "content"].text).strip()

        description = attr[attr["out"] == "description"].content
        if len(description) > 0:
            description = max(description, key=len).strip()
        else:
            description = None

        title = attr[attr["out"] == "title"].content
        title = title[~title.str.contains("\|")]
        if len(title) > 0:
            title = max(title, key=len).strip()
        else:
            return None

        authors = attr[attr["out"] ==
                       "author"].content.drop_duplicates().tolist()
        authors = [item.strip() for item in authors if len(item.strip()) > 0]

        tags = attr[attr["out"] == "tag"].content.drop_duplicates().tolist()
        if len(tags) == 1 and "," in tags[0]:
            tags = [t.strip() for t in tags[0].split(",")]

        section = attr[attr["out"] == "section"].content
        if len(section) != 0:
            section = max(section, key=len).strip()
        else:
            section = None

        date = attr[attr["out"] == "date"].content
        if len(date) != 0:
            date = max(date, key=len)
        else:
            date = None
        return {
            "title": title,
            "description": description,
            "content": content,
            "date": date,
            "tags": tags,
            "section": section,
            "authors": authors
        }
