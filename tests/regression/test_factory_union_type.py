import typing as ty

from ididi import Graph

dg = Graph()


class TokenBucketFactory:
    def __init__(
        self,
        redis: str,
        script: int,
        keyspace: ty.Union[str, None] = None,
    ):
        self.cache = redis
        self.script = script
        self.keyspace = keyspace


@dg.node
def token_bucket_factory(settings: str = "", aiocache: str = "") -> TokenBucketFactory:
    keyspace = settings.redis.keyspaces.THROTTLER / "user_request"
    script = settings.redis.TOKEN_BUCKET_SCRIPT
    script_func = aiocache.load_script(script)
    return TokenBucketFactory(
        redis=aiocache,
        script=script_func,
        keyspace=keyspace,
    )
