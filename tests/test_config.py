from dataclasses import FrozenInstanceError

import pytest

from ididi.config import GraphConfig


def test_graph_config_equality_and_hash():
    cfg_from_list = GraphConfig(self_inject=True, ignore=["a", "b"])
    cfg_from_tuple = GraphConfig(self_inject=True, ignore=("a", "b"))
    cfg_other = GraphConfig(self_inject=False, ignore=("a", "b"))

    assert cfg_from_list == cfg_from_tuple
    assert hash(cfg_from_list) == hash(cfg_from_tuple)
    assert cfg_from_list != cfg_other
    assert cfg_from_list != object()
    assert cfg_from_list.ignore == ("a", "b")


def test_graph_config_repr_uses_slots():
    cfg = GraphConfig(self_inject=False, ignore="skip")

    assert repr(cfg) == "GraphConfig(self_inject=False, ignore=('skip',))"


def test_graph_config_is_frozen():
    cfg = GraphConfig(self_inject=True, ignore=("x",))

    with pytest.raises(FrozenInstanceError):
        cfg.self_inject = False
