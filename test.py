from xmlv import XMLV

xmlv = XMLV()
attr, G = xmlv.get("https://en.wikipedia.org/wiki/Lists_of_ambassadors")
X = xmlv.fit_transform(attr, G)
