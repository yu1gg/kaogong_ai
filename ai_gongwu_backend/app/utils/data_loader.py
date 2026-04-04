"""Helpers for loading local JSON data files."""

import json
from pathlib import Path
from typing import Any


def load_json_data(file_path: str | Path) -> Any:
    """Load JSON content from a local file path."""

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"数据文件不存在: {path.resolve()}")

    try:
        with path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON 格式解析失败 [{path}]: {exc}") from exc
