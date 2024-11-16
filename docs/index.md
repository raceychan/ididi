# Ididi

Genius simplicity, unmathced power.

ididi is 100% test covered and strictly typed.

**No Container, No Provider, No Wiring, just *Python***

## Information

**Documentation**: <a href="https://raceychan.github.io/ididi" target="_blank"> https://raceychan.github.io/ididi </a>

**Source Code**: https://github.com/raceychan/ididi

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