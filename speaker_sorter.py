"""Per-episode diarization and conservative cross-episode speaker clustering.

Heavy dependencies are imported only inside the worker subprocess, which uses
Shinsekai's Python and shares its installed PyTorch stack.
"""
from __future__ import annotations

import csv
import hashlib
import heapq
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL = "pyannote/speaker-diarization-3.1"
MEDIA_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".mp4", ".mkv", ".webm"}
CACHE_VERSION = 1
REVIEW_LABEL = "_needs_review"


@dataclass(frozen=True)
class SortOptions:
    source: Path
    output: Path
    model: str = DEFAULT_MODEL
    device: str = "auto"
    threshold: float = 0.72
    min_seconds: float = 1.5
    max_seconds: float = 12.0
    edge_trim: float = 0.05
    reanalyze: bool = False
    max_speakers: int = 0

    def validate(self) -> list[Path]:
        if not self.source.is_dir():
            raise ValueError(f"输入目录不存在：{self.source}")
        if self.output == self.source or self.output in self.source.parents:
            raise ValueError("输出目录不能等于输入目录或包含输入目录。")
        if self.output.exists() and not self.output.is_dir():
            raise ValueError("输出路径必须是目录。")
        if not math.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("跨集合并阈值必须大于 0 且不超过 1。")
        if type(self.max_speakers) is not int or self.max_speakers < 0:
            raise ValueError("最多说话人数必须为非负整数，0 表示不限。")
        if not all(math.isfinite(v) for v in (self.min_seconds, self.max_seconds, self.edge_trim)):
            raise ValueError("切片时长必须是有限数值。")
        if not 0 < self.min_seconds <= self.max_seconds or self.edge_trim < 0:
            raise ValueError("切片时长应满足 0 < 最短时长 <= 最长时长，边缘裁剪不能为负数。")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("设备必须是 auto、cpu 或 cuda。")
        # Only the local model is supported: never accidentally send audio to a
        # hosted precision-* pipeline through a user-supplied repository name.
        if self.model != DEFAULT_MODEL and not Path(self.model).is_file() and not Path(self.model).is_dir():
            raise ValueError("模型应为 speaker-diarization-3.1，或本地模型配置文件/目录。")
        files = discover_media(self.source, self.output)
        if not files:
            raise ValueError("输入目录内没有支持的音频或视频文件。")
        return files


def discover_media(source: Path, output: Path) -> list[Path]:
    """Recurse without following directory symlinks, excluding our output tree."""
    found = []
    for root, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(root) / d).is_symlink()
                         and (Path(root) / d).resolve() != output)
        for name in files:
            path = Path(root) / name
            if path.suffix.lower() in MEDIA_SUFFIXES and not path.is_symlink():
                found.append(path)
    return sorted(found, key=lambda p: p.relative_to(source).as_posix().casefold())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def clean_turns(turns: list[dict], duration: float, options: SortOptions) -> tuple[list[dict], float]:
    """Keep only spans with exactly one distinct speaker; never bridge overlap.

    A sweep handles nested overlaps and duplicate tracks without quadratic work.
    The regular annotation must be used, not exclusive_speaker_diarization,
    which would incorrectly assign overlapping voices to a single person.
    """
    events: dict[float, list[tuple[str, int]]] = defaultdict(list)
    for turn in turns:
        start, end = max(0.0, float(turn["start"])), min(duration, float(turn["end"]))
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        events[start].append((turn["speaker"], 1))
        events[end].append((turn["speaker"], -1))
    active: Counter = Counter()
    previous = 0.0
    spans: list[dict] = []
    overlap_seconds = 0.0
    for position in sorted(events):
        labels = [label for label, count in active.items() if count > 0]
        if position > previous and len(labels) == 1:
            if spans and spans[-1]["speaker"] == labels[0] and spans[-1]["end"] == previous:
                spans[-1]["end"] = position
            else:
                spans.append({"start": previous, "end": position, "speaker": labels[0]})
        elif len(labels) > 1:
            overlap_seconds += position - previous
        for label, delta in events[position]:
            active[label] += delta
            if active[label] == 0:
                del active[label]
        previous = position
    clips = []
    for span in spans:
        start, end = span["start"] + options.edge_trim, span["end"] - options.edge_trim
        length = end - start
        if length < options.min_seconds:
            continue
        parts = math.ceil(length / options.max_seconds)
        # Equal parts avoid throwing away a short tail of a long utterance.
        if length / parts < options.min_seconds:
            parts = math.floor(length / options.min_seconds)
            end = min(end, start + parts * options.max_seconds)
        step = (end - start) / parts
        for index in range(parts):
            clips.append({"start": start + index * step, "end": start + (index + 1) * step,
                          "speaker": span["speaker"]})
    return clips, overlap_seconds


def normalize(vector: list[float]) -> list[float] | None:
    if not vector or not all(math.isfinite(x) for x in vector):
        return None
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm > 1e-8 else None


def cluster_speakers(records: list[dict], threshold: float, max_speakers: int = 0) -> dict[tuple[str, str], str]:
    """Complete-link cosine clustering with a same-file cannot-link constraint.

    Complete linkage avoids transitive A~B~C merges when A and C sound different.
    First respect the threshold and same-file constraint. An optional global
    cap then merges the most similar remaining groups, relaxing both constraints.
    Unusable embeddings go to a review folder outside the training input when
    capped: there is no voice evidence for assigning them to a known speaker.
    """
    vectors = [normalize(record["embedding"]) for record in records]
    review = set()
    if max_speakers:
        dimensions = Counter(len(vector) for vector in vectors if vector is not None)
        dimension = dimensions.most_common(1)[0][0] if dimensions else 0
        review = {i for i, vector in enumerate(vectors) if vector is None or len(vector) != dimension}
    groups = {i: [i] for i in range(len(records)) if i not in review}
    episodes = {i: {records[i]["file"]} for i in groups}
    similarities: dict[tuple[int, int], float] = {}
    queue = []
    indices = list(groups)
    for position, i in enumerate(indices):
        for j in indices[position + 1:]:
            left, right = vectors[i], vectors[j]
            similarity = -1.0
            if left is not None and right is not None and len(left) == len(right):
                similarity = sum(a * b for a, b in zip(left, right))
            similarities[i, j] = similarity
            if similarity >= threshold and not episodes[i] & episodes[j]:
                heapq.heappush(queue, (-similarity, i, j))
    next_id = len(records)

    def merge(i: int, j: int, constrained: bool) -> None:
        nonlocal next_id
        members = groups.pop(i) + groups.pop(j)
        files = episodes.pop(i) | episodes.pop(j)
        merged = next_id
        next_id += 1
        for other in groups:
            similarity = min(similarities[tuple(sorted((i, other)))],
                             similarities[tuple(sorted((j, other)))])
            similarities[other, merged] = similarity
            if not constrained or (similarity >= threshold and not files & episodes[other]):
                heapq.heappush(queue, (-similarity, other, merged))
        groups[merged], episodes[merged] = members, files

    while queue:
        _, i, j = heapq.heappop(queue)
        if i in groups and j in groups:
            merge(i, j, constrained=True)
    if max_speakers and len(groups) > max_speakers:
        indices = sorted(groups)
        queue = [(-similarities[i, j], i, j) for position, i in enumerate(indices) for j in indices[position + 1:]]
        heapq.heapify(queue)
        while len(groups) > max_speakers:
            _, i, j = heapq.heappop(queue)
            if i in groups and j in groups:
                merge(i, j, constrained=False)
    mapping = {(records[i]["file"], records[i]["speaker"]): REVIEW_LABEL for i in review}
    for number, members in enumerate(sorted(groups.values(), key=min), 1):
        for index in members:
            mapping[records[index]["file"], records[index]["speaker"]] = f"speaker_{number:03d}"
    return mapping


def cache_signature(path: Path, options: SortOptions, engine_version: str) -> dict:
    stat = path.stat()
    model = options.model
    config = Path(model) if Path(model).is_file() else Path(model) / "config.yaml"
    model_stamp = config.stat().st_mtime_ns if config.is_file() else None
    return {"version": CACHE_VERSION, "file": str(path.resolve()), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "model": model, "model_stamp": model_stamp,
            "engine_version": engine_version}


class PyannoteBackend:
    def __init__(self, options: SortOptions, log: Callable[[str], None]) -> None:
        self.options, self.log = options, log
        self.pipeline = None

    def analyze(self, wav: Path) -> dict:
        import numpy as np
        import soundfile as sf
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.core.task import Problem, Resolution, Specifications
        from torch.torch_version import TorchVersion

        if self.pipeline is None:
            self.log("加载 speaker-diarization-3.1 模型（首次使用需要下载；需先在 Hugging Face 接受模型条件）。")
            device = self.options.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("已选择 CUDA，但新世界当前 Python 环境无法使用 CUDA。")
            model = self.options.model
            if Path(model).is_dir():
                model = str(Path(model) / "config.yaml")
            # pyannote 3.x returns (Annotation, embeddings) and uses the older
            # authentication keyword. It is compatible with the host Torch 2.7.
            # Torch >=2.6 defaults to weights_only=True. Allow only the metadata
            # types in the official pyannote checkpoint, within this load scope.
            with torch.serialization.safe_globals([Specifications, Problem, Resolution, TorchVersion]):
                self.pipeline = Pipeline.from_pretrained(model, use_auth_token=os.environ.get("HF_TOKEN") or None)
            if self.pipeline is None:
                raise RuntimeError("无法加载模型，请确认模型访问授权及 HF_TOKEN 或本机 HF 登录。")
            self.pipeline.to(torch.device(device))
            self.log(f"推理设备：{device}。")
        waveform, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        last_log = [0.0]

        def hook(step_name, step_artifact, file=None, total=None, completed=None):
            now = time.monotonic()
            if now - last_log[0] >= 10 or (total is not None and completed == total):
                self.log(f"  {step_name}" + (f"：{completed}/{total}" if total is not None else ""))
                last_log[0] = now

        annotation, embeddings = self.pipeline(
            {"waveform": torch.from_numpy(waveform.T.copy()), "sample_rate": sample_rate},
            hook=hook, return_embeddings=True,
        )
        labels = annotation.labels()
        if labels and (embeddings is None or len(embeddings) != len(labels)):
            raise RuntimeError("模型未返回匹配的说话人声纹。请使用 pyannote.audio 3.4 的 speaker-diarization-3.1 模型。")
        turns = [{"start": float(turn.start), "end": float(turn.end), "speaker": str(label)}
                 for turn, _, label in annotation.itertracks(yield_label=True)]
        vectors = {}
        for label, vector in zip(labels, embeddings if embeddings is not None else []):
            values = np.asarray(vector).reshape(-1).tolist()
            vectors[str(label)] = normalize(values) or []
        return {"duration": len(waveform) / sample_rate, "turns": turns, "embeddings": vectors}


def resolve_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("找不到 FFmpeg，请安装 requirements-speakers.txt 中的 imageio-ffmpeg。") from exc


def decode_audio(source: Path, target: Path, sample_rate: int, ffmpeg: str) -> None:
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source),
               "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target)]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace",
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    if result.returncode:
        raise RuntimeError(f"音频解码失败：{source.name}\n{result.stderr[-2000:]}")


def run_sort(options: SortOptions, log: Callable[[str], None] = print, *, backend=None,
             ffmpeg: str | None = None, engine_version: str | None = None) -> dict:
    files = options.validate()
    import soundfile as sf
    if engine_version is None:
        from importlib.metadata import version
        engine_version = version("pyannote.audio")
        if engine_version != "3.4.0":
            raise RuntimeError("此脚本需要 pyannote.audio 3.4.0，请通过新世界安装插件依赖，复用现有 PyTorch。")
    ffmpeg = ffmpeg or resolve_ffmpeg()
    backend = backend or PyannoteBackend(options, log)
    options.output.mkdir(parents=True, exist_ok=True)
    # Each export gets its own directory, so regrouping never mixes stale files
    # into a different speaker or overwrites a manually reviewed dataset.
    run_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime("run_%Y%m%d_%H%M%S_"), dir=options.output))
    speakers_dir = run_dir / "speakers"
    speakers_dir.mkdir()
    report = {"status": "running", "source": str(options.source), "run_dir": str(run_dir),
              "speakers_dir": str(speakers_dir), "options": {**asdict(options), "source": str(options.source),
              "output": str(options.output)}, "files": len(files), "engine_version": engine_version}
    write_json(run_dir / "report.json", report)
    analyses = []
    records = []
    try:
        with tempfile.TemporaryDirectory(prefix="speaker-sort-") as temporary:
            wav = Path(temporary) / "episode.wav"
            for index, path in enumerate(files, 1):
                relative = path.relative_to(options.source).as_posix()
                log(f"分析 {index}/{len(files)}：{relative}")
                signature = cache_signature(path, options, engine_version)
                key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
                cache_path = options.output / ".speaker_cache" / f"{key}.json"
                cached = None
                if cache_path.is_file() and not options.reanalyze:
                    try:
                        candidate = json.loads(cache_path.read_text(encoding="utf-8"))
                        if candidate["signature"] == signature:
                            data = candidate["analysis"]
                            if all(key in data for key in ("duration", "turns", "embeddings")):
                                cached = data
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                if cached is None:
                    decode_audio(path, wav, 16000, ffmpeg)
                    cached = backend.analyze(wav)
                    if cache_signature(path, options, engine_version) != signature:
                        raise RuntimeError(f"处理期间源文件发生变化，请重新运行：{relative}")
                    write_json(cache_path, {"signature": signature, "analysis": cached})
                else:
                    log("  复用已完成的本集分析。")
                clips, overlap = clean_turns(cached["turns"], cached["duration"], options)
                labels = sorted({clip["speaker"] for clip in clips})
                for label in labels:
                    embedding = cached["embeddings"].get(label, [])
                    if normalize(embedding) is None:
                        treatment = "放入待核对目录。" if options.max_speakers else "保留为独立编号，不跨集合并。"
                        log(f"  {label} 声纹不足，{treatment}")
                    records.append({"file": relative, "speaker": label, "embedding": embedding})
                analyses.append({"path": path, "file": relative, "clips": clips, "overlap_seconds": overlap,
                                 "signature": signature})
                log(f"  保留 {len(clips)} 段，{len(labels)} 个本集说话人；剔除重叠 {overlap:.1f} 秒。")
            log(f"跨集匹配 {len(records)} 组声纹，合并阈值 {options.threshold:.2f}。")
            if options.max_speakers:
                log(f"最多输出 {options.max_speakers} 个说话人目录；超过上限时按声纹继续合并，可能合并同集编号，请试听核对。")
            mapping = cluster_speakers(records, options.threshold, options.max_speakers)
            write_json(run_dir / "speaker_mapping.json", [
                {"source": record["file"], "local_speaker": record["speaker"],
                 "speaker": mapping[record["file"], record["speaker"]]} for record in records])
            totals: dict[str, dict] = {}
            count = 0
            with (run_dir / "segments.csv").open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["speaker", "source", "local_speaker", "start", "end", "seconds", "clip"])
                writer.writeheader()
                for index, analysis in enumerate(analyses, 1):
                    if not analysis["clips"]:
                        continue
                    log(f"导出 {index}/{len(files)}：{analysis['file']}")
                    if cache_signature(analysis["path"], options, engine_version) != analysis["signature"]:
                        raise RuntimeError(f"源文件已改变，请重新运行：{analysis['file']}")
                    decode_audio(analysis["path"], wav, 32000, ffmpeg)
                    stem = re.sub(r"[^\w\-]", "_", analysis["path"].stem)[:48]
                    identity = hashlib.sha256(analysis["file"].encode()).hexdigest()[:12]
                    with sf.SoundFile(wav) as audio:
                        for clip_number, clip in enumerate(analysis["clips"], 1):
                            speaker = mapping[analysis["file"], clip["speaker"]]
                            begin = math.ceil(clip["start"] * audio.samplerate)
                            end = min(len(audio), math.floor(clip["end"] * audio.samplerate))
                            if end - begin < options.min_seconds * audio.samplerate - 2:
                                continue
                            filename = f"{stem}_{identity}_{clip_number:05d}_{begin}_{end}.wav"
                            destination = (run_dir / REVIEW_LABEL if speaker == REVIEW_LABEL else speakers_dir / speaker) / filename
                            destination.parent.mkdir(exist_ok=True)
                            audio.seek(begin)
                            samples = audio.read(end - begin, dtype="int16")
                            sf.write(destination, samples, audio.samplerate, subtype="PCM_16")
                            seconds = len(samples) / audio.samplerate
                            summary = totals.setdefault(speaker, {"clips": 0, "seconds": 0.0})
                            summary["clips"] += 1
                            summary["seconds"] += seconds
                            count += 1
                            writer.writerow({"speaker": speaker, "source": analysis["file"],
                                             "local_speaker": clip["speaker"], "start": begin / audio.samplerate,
                                             "end": end / audio.samplerate, "seconds": seconds,
                                             "clip": destination.relative_to(run_dir).as_posix()})
            review_totals = totals.pop(REVIEW_LABEL, {"clips": 0, "seconds": 0.0})
            report.update(status="completed", clips=count, speakers=len(totals), speaker_totals=totals,
                          review_clips=review_totals["clips"], review_seconds=review_totals["seconds"],
                          needs_review_dir=str(run_dir / REVIEW_LABEL) if review_totals["clips"] else "",
                          overlap_seconds=sum(a["overlap_seconds"] for a in analyses))
            write_json(run_dir / "report.json", report)
            log(f"完成：{len(totals)} 个说话人目录，{count} 段。训练输入目录：{speakers_dir}")
            if review_totals["clips"]:
                log(f"另有 {review_totals['clips']} 段声纹不足，保存在 {run_dir / REVIEW_LABEL}，请单独核对。")
            if count == 0:
                log("没有可用切片；请检查音频内容和最短切片时长。")
            return report
    except BaseException as exc:
        error = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            error = error.replace(token, "[REDACTED]")
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=error)
        write_json(run_dir / "report.json", report)
        raise
