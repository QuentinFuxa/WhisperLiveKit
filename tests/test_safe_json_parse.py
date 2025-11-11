from src.gpt_postproc.daily_summary import safe_json_parse


def test_safe_json_parse_valid():
    out = safe_json_parse('[{"type": "expense", "amount": 10}]')
    assert isinstance(out, list)
    assert out[0]["amount"] == 10


def test_safe_json_parse_single_object():
    out = safe_json_parse('{"type": "note", "content": "ok"}')
    assert out[0]["type"] == "note"


def test_safe_json_parse_with_fences():
    out = safe_json_parse("```json\n[{\"key\":1}]\n```")
    assert out[0]["key"] == 1


def test_safe_json_parse_invalid_returns_error():
    out = safe_json_parse("not valid json")
    assert out[0]["error"] == "parse_failed"
