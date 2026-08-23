from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """加载 YAML，并拒绝非字典顶层，避免静默使用错误配置。"""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"配置顶层必须是 mapping: {config_path}")
    return config


def require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"配置缺少路径: {label}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path

