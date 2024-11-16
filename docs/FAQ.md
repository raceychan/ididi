## FAQ

<details>
  <summary>
    How do I override, or provide a default value for a dependency?
   </summary>
<hr>


you can use `dg.node` to create a factory to override the value.
you can also have dependencies in your factory, and they will be resolved recursively.

```python
class Config:
    def __init__(self, env: str = "prod"):
        self.env = env

@dg.node
def config_factory() -> Config:
    return Config(env="test")
```

</details>

<details>
  <summary>
    How do i override a dependent in test?
   </summary>
  <hr>

you can use `dg.node` with a factory method to override the dependent resolution.

```python
class Cache: ...

class RedisCache(Cache):
    ...

class MemoryCache(Cache):
    ...

@dg.node
def cache_factory(...) -> Cache:
    return RedisCache() 

in your conftest.py:

@dg.node
def memory_cache_factory(...) -> Cache:
    return MemoryCache()    
```

as this follows LSP, it works both with ididi and type checker.

</details>

<details>
  <summary> How do I make ididi reuse a dependencies across different dependent? </summary>
  <hr>

by default, ididi will reuse the dependencies across different dependent,
you can change this behavior by setting `reuse=False` in `dg.node`.

```python
@dg.node(reuse=False) # True by default
class AuthService: ...
```

</details>
