# import pytest

# from ididi import AsyncScope, Graph, SyncScope
# from ididi._node import Dependencies, Dependency, DependentNode
# from ididi.graph import AsyncScopeSlots, GraphSlots, ScopeSlots


# def test_graph_valid_slots():
#     dg = Graph()
#     with pytest.raises(AttributeError):
#         assert dg.__slots__ == GraphSlots
#         for mem in GraphSlots:
#             assert mem in dg.__class__.__dict__
#         dg.__dict__


# def test_sync_scope_valid_slots():
#     dg = Graph()
#     sc = dg.scope().__enter__()

#     with pytest.raises(AttributeError):
#         assert isinstance(sc, SyncScope)
#         assert sc.__slots__ == ScopeSlots
#         for mem in ScopeSlots:
#             assert mem in sc.__class__.__dict__
#         sc.__dict__


# async def test_async_scope_valid_slots():
#     dg = Graph()
#     asc = await dg.ascope().__aenter__()

#     with pytest.raises(AttributeError):
#         assert isinstance(asc, AsyncScope)
#         assert asc.__slots__ == AsyncScopeSlots
#         for mem in AsyncScopeSlots:
#             assert mem in asc.__class__.__dict__
#         asc.__dict__


# def test_node_ds_slots():
#     class User:
#         ...

#     n = DependentNode.from_node(User)

#     with pytest.raises(AttributeError):
#         n.__dict__

#     def t() -> str:
#         ...

#     dep = Dependency("name", t, 3, 4)
#     with pytest.raises(AttributeError):
#         dep.__dict__

#     deps = Dependencies(dict(d=dep))

#     with pytest.raises(AttributeError):
#         deps.__dict__
