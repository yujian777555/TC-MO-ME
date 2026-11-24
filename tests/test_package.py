from importlib.util import find_spec


def test_package_is_importable() -> None:
    assert find_spec("tc_mo_me") is not None
