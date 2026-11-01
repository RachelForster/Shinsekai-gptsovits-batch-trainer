"""Read only the host's HF token scalar; never log or persist its value."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


def resolve_hf_token(config_path: Path, environment: Mapping[str, str]) -> str:
    # An explicit process override takes precedence; the HF library handles its
    # saved login when neither this variable nor the app config supplies a token.
    token = environment.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        lines = config_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return ""
    except OSError:
        raise ValueError("无法读取新世界 API 配置中的 Hugging Face Token。") from None
    for line in lines:
        if not line.startswith("hugging_face_access_token:"):
            continue
        import yaml

        try:
            value = yaml.safe_load(line.split(":", 1)[1].strip())
        except yaml.YAMLError:
            # YAML parse errors can include the original line with the secret.
            raise ValueError("Hugging Face Token 配置格式无效，请在新世界 API 配置中重新保存。") from None
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError("Hugging Face Token 配置应为文本。")
        return value.strip()
    return ""
