import json
from pathlib import Path
from typing import Any

type JSONData = dict[str, Any] | list[Any]


def read_jsonl(filepath: Path) -> list[dict]:
    """Read a JSONL file and return a list of dictionaries."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSONL file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(filepath: Path, data: list[dict[str, Any]]) -> None:
    """Write a list of dictionaries to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_json(filepath: Path) -> Any:
    """Read a standard JSON file."""
    if not filepath.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(filepath: Path, data: JSONData, indent: int = 2) -> None:
    """Write data to a standard JSON file with pretty formatting."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
