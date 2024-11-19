
# Open Closed Principle

## Original meaning

> software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification

- A module will be said to be open if it is still available for extension. For example, it should be possible to add fields to the data structures it contains, or new elements to the set of functions it performs.

- A module will be said to be closed if it is available for use by other modules. This assumes that the module has been given a well-defined, stable description (the interface in the sense of information hiding).

Interpretation:

Open for extension: you can add new functionality.
Closed for modification: you can't change existing code that others dependent on

In short: don't modify existing software entiies, add new software entities with the changes needed.

Example:

In the core of my lib ididi, is a class `DependencyGraph`, where user use it likethis

`factory.py`

```python
from ididi import DependencyGraph

dg = DependencyGraph()
```

Since users are now dependent on DependencyGraph, it is considered as `closed`.
We should no longer modify the existing interface of `DependencyGraph` that as it might create breaking changes.

e.g.

```python
class DependencyGraph:
    def __init__(self, name: str): ...


    def get_name(self) -> str: return self.name
```

Now this breaks the client code that calls `DependencyGraph()`, since it now requires a `name` parameter.

what we should do instead, it create a new subclass inherit from `DependencyGraph`, with changes needed.

```python
class NamedGraph(DependencyGraph):
    def __init__(self, name: str): ...

    def get_name(self) -> str: return self.name
```

Now, user's code that call `DependnecyGraph()` won't break, and if they need the new added functionality,
they call `NamedGraph`.

## An evolvement in the 90s

During the 1990s, the open–closed principle became popularly redefined to refer to the use of abstracted interfaces, where the implementations can be changed and multiple implementations could be created and polymorphically substituted for each other.

In contrast to Meyer's usage, this definition advocates inheritance from abstract base classes. Interface specifications can be reused through inheritance but implementation need not be. The existing interface is closed to modifications and new implementations must, at a minimum, implement that interface.

```python
class AbstractDependencyGraph:
    def get_name(self) -> str: ...
```