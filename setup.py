from Cython.Build import cythonize
from setuptools import setup

from ididi import VERSION

extensions = ["ididi/graph.pyx", "ididi/_node.py", "ididi/_ds.py"]
setup(
    name="ididi",
    version=VERSION,
    ext_modules=cythonize(extensions),
)
