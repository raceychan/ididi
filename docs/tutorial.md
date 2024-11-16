
### Usage with FastAPI

```python title="app.py"
from fastapi import FastAPI
from ididi import DependencyGraph

app = FastAPI()
dg = DependencyGraph()

def auth_service_factory(db: DataBase) -> AuthService:
    async with dg.scope() as scope
        yield dg.resolve(AuthService)

Service = ty.Annotated[AuthService, Depends(dg.factory(auth_service_factory))]

@app.get("/")
def get_service(service: Service):
    return service
```
