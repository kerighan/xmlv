XMLVectorizer
=============

XMLVectorizer brings web pages to deep learning.
Given a web page, this library returns a vector for each element as well as the xml graph structure. This library allows for elements to be passed to a graph neural network.


```python
from xmlv import XMLV


xmlv = XMLV(vectorize_text=True, vectorize_link=True)
# attr is a dataframe where each line is an element and columns properties
# G is the networkx graph extracted from the web page
attr, G = xmlv.get("https://en.wikipedia.org/wiki/Lists_of_ambassadors")

# this fits the model using the attributes and vectorizes the properties
# X is a numpy matrix
X = xmlv.fit_transform(attr)
# G and X can the be used to fit a graph neural network

# model persistence
xmlv.save("xmlv.p")
xmlv = XMLV.load("xmlv.p")
```
