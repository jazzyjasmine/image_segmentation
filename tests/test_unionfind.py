import pytest

import segment


@pytest.mark.slow
def test_unionfind_0():
    vertices = ["a", "b", "c", "d", "e"]
    values = [1, 2, 3, 4, 5]
    uf = segment.UnionFind(vertices, values)
    for k in vertices:
        assert uf.find(k) == k
        assert uf.size(k) == 1
        assert uf.max_diff(k) == 0

    uf.union("a", "c")
    for k in vertices:
        if k in ("a", "c"):
            continue
        assert uf.find(k) == k
        assert uf.size(k) == 1
        assert uf.max_diff(k) == 0
    assert uf.find("a") == uf.find("c")
    assert uf.size("a") == uf.size("c") == 2
    assert uf.max_diff("a") == uf.max_diff("c") == 2

    uf.union("c", "d")
    for k in vertices:
        if k in ("a", "c", "d"):
            continue
        assert uf.find(k) == k
        assert uf.size(k) == 1
        assert uf.max_diff(k) == 0
    assert uf.find("a") == uf.find("c") == uf.find("d")
    assert uf.size("a") == uf.size("c") == 3
    assert uf.max_diff("a") == uf.max_diff("c") == 3

    uf.union("b", "e")
    assert uf.find("a") == uf.find("c") == uf.find("d")
    assert uf.size("a") == uf.size("c") == uf.size("d") == 3
    assert uf.max_diff("a") == uf.max_diff("c") == uf.size("d") == 3
    assert uf.find("b") == uf.find("e")
    assert uf.size("b") == uf.size("e") == 2
    assert uf.max_diff("b") == uf.max_diff("e") == 3

    uf.union("a", "e")
    for k in vertices:
        assert uf.find(k) == uf.find("a")
        assert uf.size(k) == 5
        assert uf.max_diff(k) == 4


@pytest.mark.slow
def test_unionfind_1():
    vertices = ["a", "b", "c", "d", "e"]
    values = [1, 2, 3, 4, 5]
    uf = segment.UnionFind(vertices, values)
    uf.union("a", "c")
    uf.union("b", "e")
    assert uf.find("a") == uf.find("c")
    assert uf.size("a") == uf.size("c") == 2
    assert uf.max_diff("a") == uf.max_diff("c") == 2
    assert uf.find("b") == uf.find("e")
    assert uf.size("b") == uf.size("e") == 2
    assert uf.max_diff("b") == uf.max_diff("e") == 3
    uf.union("a", "e")
    assert uf.find("a") == uf.find("b") == uf.find("c") == uf.find("e")
    assert uf.size("a") == uf.size("b") == uf.size("c") == uf.size("e") == 4
    assert uf.max_diff("a") == uf.max_diff("b") == uf.size("c") == uf.size("e") == 4
    assert uf.find("d") == "d"
    assert uf.size("d") == 1
    assert uf.max_diff("d") == 0
