import pytest

from tc_mo_me.trajectory import Trajectory

SENSITIVE_KEYS = (
    "api_key",
    "access_key",
    "auth_token",
    "authorization",
    "client_secret",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
)

ASSIGNMENTS = (
    ('"{key}": "unit-test-marker"', '"{key}": "<redacted>"'),
    ("{key} = 'unit-test-marker'", "{key} = '<redacted>'"),
    ("{key}: unit-test-marker", "{key}: <redacted>"),
    ("{key}=unit-test-marker", "{key}=<redacted>"),
)


def record_text(kind: str, text: str) -> str | None:
    trajectory = Trajectory()
    if kind == "code":
        return trajectory.record_code(0, text).code
    return trajectory.record_exception(0, RuntimeError(text)).message


@pytest.mark.parametrize("kind", ["code", "exception"])
@pytest.mark.parametrize("key", SENSITIVE_KEYS)
@pytest.mark.parametrize(("template", "expected"), ASSIGNMENTS)
def test_text_redaction_covers_every_sensitive_key(kind, key, template, expected):
    text = template.format(key=key)

    assert record_text(kind, text) == expected.format(key=key)


@pytest.mark.parametrize("key", [key for key in SENSITIVE_KEYS if "_" in key])
def test_text_redaction_accepts_hyphenated_keys(key):
    key = key.replace("_", "-")

    assert record_text("code", f"{key}=unit-test-marker") == f"{key}=<redacted>"


def test_unquoted_redaction_preserves_json_delimiters():
    assert record_text("code", '{"token": unit-test-marker}') == ('{"token": <redacted>}')


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bearer abcDEF123._~+/=-", "Bearer <redacted>"),
        ("sk-abcd1234", "<redacted>"),
    ],
)
def test_existing_token_redaction_is_preserved(text, expected):
    assert record_text("exception", text) == expected


def test_redaction_leaves_ordinary_content_unchanged():
    text = (
        "token_count = 4\npasswordless = True\nsecret_sauce = 'public'\nprint('ordinary content')"
    )

    assert record_text("code", text) == text
