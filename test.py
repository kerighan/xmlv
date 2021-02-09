import time
import pandas as pd
from xmlv import XMLV
from glob import glob


xmlv = XMLV(vectorize_text=True, vectorize_link=True)
attr, G = xmlv.get("https://en.wikipedia.org/wiki/Lists_of_ambassadors")

start = time.time()
X = xmlv.fit_transform(attr)
print(time.time() - start)
print(X.shape)

# start = time.time()
# attr, G = xmlv.get("https://en.wikipedia.org/wiki/List_of_ambassadors_of_Afghanistan")
# X = xmlv.vectorize(attr)
# print(time.time() - start)
# print(X.shape)
