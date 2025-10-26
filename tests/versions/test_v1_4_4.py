from ididi import Graph, use


def test_scope_graph_share_data():
    dg = Graph()
    with dg.scope("s1") as s1:
        assert s1.name == "s1"
        assert dg.type_registry is s1.type_registry


def test_create_scope_from_scope():
    dg = Graph()

    class User:
        ...

    dg.node(User)

    with dg.scope() as s1:
        assert s1.use_scope() is s1
        assert User in s1
        with s1.scope() as s2:
            assert User in s2
            u2 = s2.resolve(User)
        u1 = s1.resolve(User)
        assert u1 is not u2


def test_scope_resolve_inheritance():
    dg = Graph()

    class User:
        ...

    class Cache:
        ...

    dg.node(use(User, reuse=True))
    dg.node(use(Cache, reuse=True))
    dg_u = dg.resolve(User)

    with dg.scope() as s1:
        s1_u = s1.resolve(User)
        assert s1_u is dg_u
        s1_cache = s1.resolve(Cache)
        dg_cache = dg.resolve(Cache)
        assert s1_cache is not dg_cache

        with s1.scope() as s12:
            assert s12.resolve(User) is dg_u
            s12_cache = s12.resolve(Cache)
            assert s12_cache is s1_cache
            assert s12_cache is not dg_cache

    with dg.scope() as s2:
        s2_u = s2.resolve(User)
        assert s2_u is dg_u
        s2_cache = s2.resolve(Cache)
        assert s2_cache is not s1_cache
