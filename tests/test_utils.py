from mobula.utils import get_git_hash


def test_get_git_hash():
    git_hash = get_git_hash()
    assert type(git_hash) == str, (git_hash, type(git_hash))
    assert len(git_hash) == 7, git_hash
