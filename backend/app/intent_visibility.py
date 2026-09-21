"""Remove internal routing markers from persisted or returned assistant text."""

import re


_INTENT_JSON = re.compile(r"\[INTENT_JSON\].*?\[/INTENT_JSON\]", re.DOTALL)
_INTENT_LINE = re.compile(r"\[INTENT:(?:food|exercise)\][^\r\n]*")


def visible_assistant_text(value: str) -> str:
    text = _INTENT_JSON.sub("", value or "")
    return _INTENT_LINE.sub("", text).strip()
