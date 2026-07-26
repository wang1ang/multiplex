import json

import pytest

from try_engine import load_prompt_file, parse_args


def test_defaults_to_dynamic_depth_three():
    args = parse_args([])

    assert args.depth == 3
    assert args.dynamic_depth is True


def test_dynamic_depth_can_be_disabled():
    args = parse_args(["--no-dynamic-depth"])

    assert args.depth == 3
    assert args.dynamic_depth is False


def test_load_prompt_file_reads_plain_text(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("\nhello from a file\n", encoding="utf-8")

    assert load_prompt_file(str(path)) == "hello from a file"


def test_load_prompt_file_reads_first_jsonl_prompt(tmp_path):
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"id": "first", "prompt": "first prompt"},
        {"id": "second", "prompt": "second prompt"},
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    assert load_prompt_file(str(path)) == "first prompt"


def test_load_prompt_file_rejects_json_without_prompt(tmp_path):
    path = tmp_path / "prompt.json"
    path.write_text('{"text": "missing prompt key"}', encoding="utf-8")

    with pytest.raises(ValueError, match="'prompt' field"):
        load_prompt_file(str(path))
