from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [Extension("ididi.graph", ["ididi/graph.pyx"]), "ididi/_node.py", "ididi/_ds.py"]
setup(
    name="ididi",
    ext_modules=cythonize(extensions),
)
