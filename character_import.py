from __future__ import annotations

import hashlib
import re
import wave
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


REF_MIN_SECONDS = 3.0
REF_MAX_SECONDS = 10.0
SLICE_SAMPLE_RATE = 32_000
GPT_WEIGHT_DIR = "GPT_weights_v2ProPlus"
SOVITS_WEIGHT_DIR = "SoVITS_weights_v2ProPlus"
SLICER_DIR_NAMES = ("slicer_opt", "slice_opt")
AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}

_GPT_WEIGHT_RE = re.compile(r"^(.+)-e(\d+)\.ckpt$", re.IGNORECASE)
_SOVITS_WEIGHT_RE = re.compile(r"^(.+)_e(\d+)_s(\d+)\.pth$", re.IGNORECASE)
_SLICE_SPAN_RE = re.compile(r"_(\d+)_(\d+)\.[A-Za-z0-9]+$")
_ASCII_PREFIX_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_PROMPT_LANGS = {
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "zh": "zh",
    "cn": "zh",
    "yue": "zh",
    "chinese": "zh",
    "all_zh": "zh",
    "en": "en",
    "english": "en",
}


@dataclass(frozen=True)
class CharacterVoiceBinding:
    name: str
    created: bool
    gpt_model_path: str
    sovits_model_path: str
    refer_audio_path: str
    prompt_text: str
    prompt_lang: str


@dataclass(frozen=True)
class CharacterVoiceSkip:
    name: str
    reason: str


@dataclass(frozen=True)
class CharacterVoiceImportResult:
    imported: tuple[CharacterVoiceBinding, ...]
    skipped: tuple[CharacterVoiceSkip, ...]
    characters: list[dict[str, Any]]

    @property
    def summary(self) -> str:
        imported = len(self.imported)
        skipped = len(self.skipped)
        created = sum(1 for item in self.imported if item.created)
        updated = imported - created
        parts = [f"已写入 {imported} 个人物配置（新建 {created}，更新 {updated}）"]
        if skipped:
            parts.append(f"跳过 {skipped} 个")
        return "，".join(parts) + "。"


def import_trained_voices(
    gpt_sovits_root: str | Path,
    *,
    source_root: str | Path | None = None,
    characters: Sequence[Mapping[str, Any]] | None = None,
    min_seconds: float = REF_MIN_SECONDS,
    max_seconds: float = REF_MAX_SECONDS,
) -> CharacterVoiceImportResult:
    """Fill Shinsekai character voice fields from GPT-SoVITS v2ProPlus outputs."""
    root = Path(gpt_sovits_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"GPT-SoVITS 目录不存在：{root}")

    names = _character_names(root, source_root)
    if not names:
        raise ValueError("没有找到可导入的人物。请先完成训练，或填写包含人物子目录的语音总目录。")

    documents = [_copy_character(item) for item in characters or ()]
    imported: list[CharacterVoiceBinding] = []
    skipped: list[CharacterVoiceSkip] = []
    for name in names:
        try:
            binding = _bind_character(
                root,
                name,
                documents,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
            )
        except ValueError as exc:
            skipped.append(CharacterVoiceSkip(name, str(exc)))
            continue
        imported.append(binding)
    return CharacterVoiceImportResult(tuple(imported), tuple(skipped), documents)


def load_character_documents(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"无法读取人物配置：{exc}") from exc
    if loaded in (None, ""):
        return []
    if not isinstance(loaded, list):
        raise ValueError("characters.yaml 必须是角色列表。")
    return [_copy_character(item) for item in loaded if isinstance(item, Mapping)]


def save_character_documents(path: Path, characters: Sequence[Mapping[str, Any]]) -> None:
    documents = [_stringify(_copy_character(item)) for item in characters]
    if _save_with_config_manager(path, documents):
        return
    _write_yaml_list(path, documents)


def character_config_path(project_root: Path) -> Path:
    return project_root / "data" / "config" / "characters.yaml"


def _character_names(root: Path, source_root: str | Path | None) -> list[str]:
    cleaned = str(source_root or "").strip().strip('"')
    if cleaned:
        source = Path(cleaned).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"语音总目录不存在：{source}")
        names = [
            path.name
            for path in sorted(source.iterdir(), key=lambda item: item.name.casefold())
            if path.is_dir() and not path.name.startswith(".") and path.name != "_needs_review"
        ]
        if not names:
            raise ValueError(f"{source} 下没有人物子目录。")
        return names
    return _names_from_weights(root)


def _names_from_weights(root: Path) -> list[str]:
    gpt_names = {match.group(1) for match, _path in _iter_gpt_weights(root / GPT_WEIGHT_DIR)}
    sovits_names = {match.group(1) for match, _path in _iter_sovits_weights(root / SOVITS_WEIGHT_DIR)}
    return sorted(gpt_names & sovits_names, key=str.casefold)


def _bind_character(
    root: Path,
    name: str,
    documents: list[dict[str, Any]],
    *,
    min_seconds: float,
    max_seconds: float,
) -> CharacterVoiceBinding:
    gpt_path = _select_gpt_weight(root / GPT_WEIGHT_DIR, name)
    if gpt_path is None:
        raise ValueError(f"未找到 GPT 权重（{GPT_WEIGHT_DIR}/{name}-e*.ckpt）")
    sovits_path = _select_sovits_weight(root / SOVITS_WEIGHT_DIR, name)
    if sovits_path is None:
        raise ValueError(f"未找到 SoVITS 权重（{SOVITS_WEIGHT_DIR}/{name}_e*_s*.pth）")

    transcripts = _load_transcripts(root, name)
    audio_path, prompt_text, prompt_lang = _select_reference(
        root,
        name,
        transcripts,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )

    index = _find_character_index(documents, name)
    created = index is None
    if created:
        documents.append(_new_character_document(name))
        index = len(documents) - 1
    target = documents[index]
    target["gpt_model_path"] = str(gpt_path)
    target["sovits_model_path"] = str(sovits_path)
    target["refer_audio_path"] = str(audio_path)
    target["prompt_text"] = prompt_text
    target["prompt_lang"] = prompt_lang
    return CharacterVoiceBinding(
        name=str(target["name"]),
        created=created,
        gpt_model_path=str(gpt_path),
        sovits_model_path=str(sovits_path),
        refer_audio_path=str(audio_path),
        prompt_text=prompt_text,
        prompt_lang=prompt_lang,
    )


def _select_gpt_weight(directory: Path, name: str) -> Path | None:
    matches = [
        (int(match.group(2)), path)
        for match, path in _iter_gpt_weights(directory)
        if match.group(1) == name
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1].stat().st_mtime, item[1].name))
    return matches[-1][1]


def _select_sovits_weight(directory: Path, name: str) -> Path | None:
    matches = [
        (int(match.group(2)), int(match.group(3)), path)
        for match, path in _iter_sovits_weights(directory)
        if match.group(1) == name
    ]
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], item[1], item[2].stat().st_mtime, item[2].name))
    return matches[-1][2]


def _iter_gpt_weights(directory: Path) -> Iterable[tuple[re.Match[str], Path]]:
    yield from _iter_named_weights(directory, _GPT_WEIGHT_RE)


def _iter_sovits_weights(directory: Path) -> Iterable[tuple[re.Match[str], Path]]:
    yield from _iter_named_weights(directory, _SOVITS_WEIGHT_RE)


def _iter_named_weights(
    directory: Path,
    pattern: re.Pattern[str],
) -> Iterable[tuple[re.Match[str], Path]]:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            yield match, path


def _slicer_dirs(root: Path, name: str) -> list[Path]:
    found: list[Path] = []
    for folder in SLICER_DIR_NAMES:
        candidate = root / "output" / folder / name
        if candidate.is_dir():
            found.append(candidate)
    return found


def _load_transcripts(root: Path, name: str) -> dict[str, tuple[str, str]]:
    list_path = root / "output" / "asr_opt" / f"{name}.list"
    if not list_path.is_file():
        return {}
    transcripts: dict[str, tuple[str, str]] = {}
    try:
        lines = list_path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return {}
    for raw in lines:
        line = raw.strip()
        if not line or line.count("|") < 3:
            continue
        audio_path, _speaker, lang, text = line.split("|", 3)
        filename = Path(audio_path.strip()).name
        prompt = text.strip()
        if not filename or not prompt:
            continue
        transcripts[filename] = (prompt, _normalize_prompt_lang(lang))
    return transcripts


def _select_reference(
    root: Path,
    name: str,
    transcripts: Mapping[str, tuple[str, str]],
    *,
    min_seconds: float,
    max_seconds: float,
) -> tuple[Path, str, str]:
    audio_files: list[Path] = []
    for directory in _slicer_dirs(root, name):
        audio_files.extend(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
    audio_files.sort(key=lambda path: path.name)
    if not audio_files:
        raise ValueError(f"未找到参考音频目录 output/slicer_opt/{name} 或 output/slice_opt/{name}")

    fallback: tuple[Path, float] | None = None
    for path in audio_files:
        duration = audio_duration_seconds(path)
        if duration is None or duration < min_seconds or duration > max_seconds:
            continue
        transcript = transcripts.get(path.name)
        if transcript:
            return path, transcript[0], transcript[1]
        if fallback is None:
            fallback = (path, duration)
    if fallback is not None:
        return fallback[0], "", ""
    raise ValueError(f"未找到 {min_seconds:g}–{max_seconds:g} 秒的参考音频")


def audio_duration_seconds(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                frames = handle.getnframes()
            if rate > 0 and frames > 0:
                return frames / rate
        except (OSError, wave.Error):
            pass
    match = _SLICE_SPAN_RE.search(path.name)
    if match is None:
        return None
    start = int(match.group(1))
    end = int(match.group(2))
    if end <= start:
        return None
    return (end - start) / SLICE_SAMPLE_RATE


def _normalize_prompt_lang(raw: str) -> str:
    key = str(raw or "").strip().lower()
    return _PROMPT_LANGS.get(key, key if key in {"en", "ja", "zh"} else "ja")


def _find_character_index(documents: Sequence[Mapping[str, Any]], name: str) -> int | None:
    key = name.casefold()
    for index, item in enumerate(documents):
        if str(item.get("name") or "").casefold() == key:
            return index
    return None


def _new_character_document(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "color": _color_for_name(name),
        "sprite_prefix": _sprite_prefix_for_name(name),
        "sprites": [],
        "character_brief": "",
        "character_setting": "",
        "sprite_scale": 1.0,
        "emotion_tags": "",
        "gpt_model_path": "",
        "sovits_model_path": "",
        "refer_audio_path": "",
        "prompt_text": "",
        "prompt_lang": "",
        "speech_speed": 1.0,
        "speech_volume": 1.0,
        "pronunciation_map": {},
    }


def _sprite_prefix_for_name(name: str) -> str:
    if _ASCII_PREFIX_RE.fullmatch(name):
        return name[:80]
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    return f"char_{digest}"


def _color_for_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    red = 72 + digest[0] % 150
    green = 72 + digest[1] % 150
    blue = 72 + digest[2] % 150
    return f"#{red:02x}{green:02x}{blue:02x}"


def _copy_character(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _copy_value(value) for key, value in item.items()}


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _stringify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify(item) for item in value]
    return value


def _write_yaml_list(path: Path, characters: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = yaml.safe_dump(
        list(characters),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _save_with_config_manager(path: Path, characters: Sequence[Mapping[str, Any]]) -> bool:
    try:
        from config.config_manager import ConfigManager, character_name_key
        from config.schema import Character
    except ImportError:
        return False

    instance = getattr(ConfigManager, "_instance", None)
    config = getattr(instance, "_config", None) if instance is not None else None
    if instance is None or config is None:
        return False

    configured = Path(ConfigManager._CHARACTERS_CONFIG_PATH)
    if not configured.is_absolute():
        configured = Path.cwd() / configured
    try:
        if configured.resolve(strict=False) != path.resolve(strict=False):
            return False
    except OSError:
        return False

    by_key = {character_name_key(str(item.get("name") or "")): item for item in characters}
    remaining = dict(by_key)
    for character in config.characters:
        item = remaining.pop(character_name_key(character.name), None)
        if item is None:
            continue
        _apply_voice_fields(character, item)
    for item in remaining.values():
        config.characters.append(
            Character(
                name=str(item.get("name") or ""),
                color=str(item.get("color") or "#88c0d0"),
                sprite_prefix=str(item.get("sprite_prefix") or ""),
                gpt_model_path=str(item.get("gpt_model_path") or ""),
                sovits_model_path=str(item.get("sovits_model_path") or ""),
                refer_audio_path=str(item.get("refer_audio_path") or ""),
                prompt_text=str(item.get("prompt_text") or ""),
                prompt_lang=str(item.get("prompt_lang") or ""),
                sprites=[],
                sprite_scale=float(item.get("sprite_scale") or 1.0),
                emotion_tags=str(item.get("emotion_tags") or ""),
                character_brief=str(item.get("character_brief") or ""),
                character_setting=str(item.get("character_setting") or ""),
                speech_speed=float(item.get("speech_speed") or 1.0),
                speech_volume=float(item.get("speech_volume") or 1.0),
                pronunciation_map=dict(item.get("pronunciation_map") or {}),
            )
        )
    instance.save_characters_config()
    return True


def _apply_voice_fields(character: Any, item: Mapping[str, Any]) -> None:
    character.gpt_model_path = str(item.get("gpt_model_path") or "")
    character.sovits_model_path = str(item.get("sovits_model_path") or "")
    character.refer_audio_path = str(item.get("refer_audio_path") or "")
    character.prompt_text = str(item.get("prompt_text") or "")
    character.prompt_lang = str(item.get("prompt_lang") or "")
