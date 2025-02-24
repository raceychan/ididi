from Cython.Build import cythonize
from setuptools import setup

extensions = ["ididi/resolve.pyx", "ididi/_node.py", "ididi/_ds.py"]
setup(
    name="ididi",
    ext_modules=cythonize(extensions),
)
