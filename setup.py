from Cython.Build import cythonize
from setuptools import setup

__version__ = "1.5.4"
VERSION = __version__

#TODO: write version in a file

extensions = ["ididi/graph.pyx", "ididi/_node.py", "ididi/_ds.py"]

setup(
    name="ididi",
    ext_modules=cythonize(extensions),
    version=VERSION,
)
