"""
specifically for test driven development for a new feature of ididi;
or for debug of the feature.
run test with: make feat
"""

from contextlib import asynccontextmanager

from ididi import AsyncResource, DependencyGraph


class ACM:
    def __init__(self):
        self._closed = False

    def __aenter__(self):
        return self

    def __aexit__(self, *args):
        self._closed = True

