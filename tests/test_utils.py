from mobula.utils import get_git_hash


def test_get_git_hash():
    git_hash = get_git_hash()
    assert type(git_hash) == str, (git_hash, type(git_hash))
    assert len(git_hash) == 7, git_hash


def test_edict():
    from mobula.edict import edict
    data = edict(a=3, b=4)
    assert 'a' in data
    assert hasattr(data, 'a')
    assert 'b' in data
    assert hasattr(data, 'b')
    assert len(data) == 2
    assert data['a'] == 3
    assert data['b'] == 4

    data.a = 5
    assert data['a'] == 5
    data.a += 3
    assert data['a'] == 8

    data.update(dict(c=6))
    assert 'c' in data
    assert data['c'] == 6
    data['c'] += 1
    assert data['c'] == 7

    del data.b
    assert 'b' not in data
    assert not hasattr(data, 'b')
    assert len(data) == 2

    del data['a']
    assert 'a' not in data
    assert len(data) == 1
