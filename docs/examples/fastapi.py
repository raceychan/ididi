import typing as ty

from fastapi.routing import APIRoute, APIRouter
from starlette.types import ASGIApp, Receive, Scope, Send

from ididi import DependencyGraph


class GraphedScope(ty.TypedDict):
    dg: DependencyGraph


# ========== global state =========


@asynccontextmanager
async def lifespan(app: FastAPI | None = None) -> ty.AsyncIterator[GraphedScope]:
    async with DependencyGraph() as dg:
        yield {"dg": dg}


@app.post("/users")
async def signup_user(request: Request):
    dg = request.state.dg
    service = dg.resolve(UserService)
    user_id = await service.signup_user(...)
    return user_id


# ============== Route Level ===========


class UserRoute(APIRoute):
    def get_route_handler(self) -> ty.Callable[[Request], ty.Awaitable[Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:

            dg = DependencyGraph()
            request.scope["dg"] = dg

            async with dg.scope() as user_scope:
                response = await original_route_handler(request)
                return response

        return custom_route_handler


user_router = APIRouter(route_class=UserRoute)


# ==================== request level ============


class GraphedMiddleware:
    def __init__(self, app, dg: DependencyGraph):
        self.app = app
        self.dg = dg

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        """
        Manipulate request scope to add request_id
        """
        # NOTE: remove follow three lines would break lifespan
        # as startlette would pass lifespan event here
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        scope["dg"] = self.dg
        await self.app(scope, receive, send)


middleware = [GraphedMiddleware]
