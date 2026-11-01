from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path


ASR_MODEL_SIZE = "large-v3"
FASTER_WHISPER_REPO = "Systran/faster-whisper-large-v3"
def local_faster_whisper_dir(root: Path, size: str = ASR_MODEL_SIZE) -> Path:
    return root / "tools" / "asr" / "models" / f"faster-whisper-{size}"


def faster_whisper_model_ready(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    names = {path.name.lower() for path in directory.iterdir() if path.is_file()}
    has_config = "config.json" in names
    has_weights = "model.bin" in names or any(name.endswith(".bin") for name in names)
    has_tokenizer = "tokenizer.json" in names or "vocabulary.json" in names
    return has_config and has_weights and has_tokenizer


def default_huggingface_cache_roots(environment: Mapping[str, str] | None = None) -> list[Path]:
    env = environment if environment is not None else os.environ
    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        raw = str(env.get(key) or "").strip()
        if raw:
            roots.append(Path(raw).expanduser())
    roots.append(Path.home() / ".cache" / "huggingface")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve(strict=False)
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def discover_cached_faster_whisper(cache_roots: Sequence[Path]) -> Path | None:
    repo_folder = "models--" + FASTER_WHISPER_REPO.replace("/", "--")
    candidates: list[Path] = []
    for root in cache_roots:
        snapshots = root / "hub" / repo_folder / "snapshots"
        if not snapshots.is_dir():
            snapshots = root / repo_folder / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if snapshot.is_dir() and faster_whisper_model_ready(snapshot):
                candidates.append(snapshot)
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def link_or_copy_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if faster_whisper_model_ready(destination) or destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"目标目录已存在且不是完整 Faster Whisper 模型：{destination}")
    try:
        os.symlink(source, destination, target_is_directory=True)
        return
    except OSError:
        pass
    if os.name == "nt":
        import subprocess

        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode == 0 and destination.exists():
            return
    import shutil

    shutil.copytree(source, destination)


def prepare_local_faster_whisper(
    root: Path,
    *,
    cache_roots: Sequence[Path] | None = None,
    link: bool = True,
) -> Path | None:
    local_dir = local_faster_whisper_dir(root)
    if faster_whisper_model_ready(local_dir):
        return local_dir
    cached = discover_cached_faster_whisper(cache_roots or default_huggingface_cache_roots())
    if cached is None:
        return None
    if link:
        link_or_copy_model(cached, local_dir)
        if faster_whisper_model_ready(local_dir):
            return local_dir
        return cached
    return cached


def asr_local_hub_env(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Point Faster Whisper at the real Hugging Face cache and forbid Hub sync."""
    env = environment if environment is not None else os.environ
    chosen: Path | None = None
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        raw = str(env.get(key) or "").strip()
        if not raw:
            continue
        candidate = Path(raw).expanduser()
        if discover_cached_faster_whisper([candidate]) is not None:
            chosen = candidate
            break
    if chosen is None:
        chosen = Path.home() / ".cache" / "huggingface" / "hub"
    hf_home = chosen if chosen.name != "hub" else chosen.parent
    hf_hub = hf_home / "hub"
    return {
        "HF_HOME": str(hf_home),
        "HF_HUB_CACHE": str(hf_hub),
        "HUGGINGFACE_HUB_CACHE": str(hf_hub),
        "TRANSFORMERS_CACHE": str(hf_home),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def missing_faster_whisper_hint(directory: Path) -> str:
    return (
        f"未找到本地 Faster Whisper {ASR_MODEL_SIZE}。"
        f"GPT-SoVITS WebUI 使用的路径是 {directory}，"
        f"也会复用本机 Hugging Face 缓存中的 {FASTER_WHISPER_REPO}。"
    )
