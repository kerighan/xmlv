import setuptools


setuptools.setup(
    name="xmlv",
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="XMLVectorizer turn elements of a web pages to numpy vectors",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=["scikit-learn", "networkx", "pandas", "numpy"]
)
