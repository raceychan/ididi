# Ididi

Genius simplicity, unmathced power.

ididi is 100% test covered and strictly typed.

**No Container, No Provider, No Wiring, just *Python***

## Information

**Documentation**: <a href="https://raceychan.github.io/ididi" target="_blank"> https://raceychan.github.io/ididi </a>

**Source Code**: <a href=" https://github.com/raceychan/ididi" target="_blank">  https://github.com/raceychan/ididi</a>

## Install

<div class="termy">

```console
$ pip install ididi
```

To view viusal dependency graph, install `graphviz`

```console
$ pip install "ididi[graphviz]"
```

</div>

## Usage

### Quick Start

```python
import ididi

class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

user_service = ididi.resolve(UserService) 
```

## Why Ididi?

ididi helps you do this while stays out of your way, you do not need to create additional classes like `Container`, `Provider`, `Wire`, nor adding lib-specific annotation like `Closing`, `Injectable`, etc.

ididi provides unique powerful features that most alternatives don't have, such as support to inifinite number of context-specific nested sopce, lazydependent, advanced circular dependency detection, plotting, etc.

Let's now explore these fantastic feaures in the `Features` section.

## To di, or not to di, this is my question

If you

- Are not familar with dependency injection, or OOP in general.
- Can't come up with cases where it would be useful.
- You fee like `it's java` and you have `javaphobia`.

**Let's discuss in the `Introduction` section.**
