from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from .asr_model import (
        ASR_MODEL_SIZE,
        asr_local_hub_env,
        faster_whisper_model_ready,
        missing_faster_whisper_hint,
        prepare_local_faster_whisper,
    )
except ImportError:
    from asr_model import (
        ASR_MODEL_SIZE,
        asr_local_hub_env,
        faster_whisper_model_ready,
        missing_faster_whisper_hint,
        prepare_local_faster_whisper,
    )

MODEL_VERSION = "v2ProPlus"
ASR_LANGUAGE = "auto"
SLICER_PARTS = 4
SOVITS_EPOCHS = 8
SOVITS_SAVE_EVERY = 4
GPT_EPOCHS = 15
GPT_SAVE_EVERY = 5

AUDIO_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".ogg",
    ".opus",
    ".wav",
    ".wma",
}
STAGES = ("slice", "asr", "prepare", "sovits", "gpt")
STAGE_LABELS = {
    "slice": "语音切分",
    "asr": "Faster Whisper 语音识别",
    "prepare": "训练集格式化一键三连",
    "sovits": "SoVITS 训练",
    "gpt": "GPT 训练",
}


class TrainingError(RuntimeError):
    """Raised when one GPT-SoVITS stage does not produce a usable result."""


class GptSovitsConfigurationError(TrainingError):
    """Raised when Shinsekai has no usable GPT-SoVITS directory configured."""


class TrainingCancelled(TrainingError):
    """Raised after a user-requested cancellation."""


class CommandFailed(TrainingError):
    """Raised when a GPT-SoVITS subprocess returns a non-zero exit code."""

    def __init__(self, label: str, return_code: int) -> None:
        self.label = label
        self.return_code = return_code
        super().__init__(f"{label} 失败，退出码 {return_code}。")


@dataclass(frozen=True)
class TrainingEvent:
    message: str
    level: str = "info"
    experiment: str = ""
    stage: str = ""


@dataclass(frozen=True)
class RuntimeProfile:
    gpu: str = "0"
    batch_size: int = 1
    use_half: bool = False

    @property
    def asr_precision(self) -> str:
        # Matches webui.py's Faster Whisper default selection.
        return "int8" if self.batch_size <= 4 else "float16" if self.use_half else "float32"


def detect_runtime_profile() -> RuntimeProfile:
    """Mirror the WebUI defaults without importing its heavyweight Gradio module."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        memory_mib = int(result.stdout.splitlines()[0].strip())
        memory_gib = memory_mib / 1024 + 0.4
        return RuntimeProfile(batch_size=max(1, int(memory_gib // 2)), use_half=True)
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        # The bundled NVIDIA build can still decide whether CUDA is available.
        # A batch size of one is the safest fallback.
        return RuntimeProfile()


def discover_experiments(source_root: Path) -> tuple[list[Path], list[Path]]:
    """Return immediate child folders with audio and child folders that were skipped."""
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise TrainingError(f"语音总目录不存在或不是目录：{source_root}")
    candidates = sorted((path for path in source_root.iterdir() if path.is_dir()), key=lambda p: p.name.casefold())
    ready: list[Path] = []
    skipped: list[Path] = []
    for path in candidates:
        if any(item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES for item in path.iterdir()):
            ready.append(path)
        else:
            skipped.append(path)
    if not ready:
        raise TrainingError(f"{source_root} 下没有包含音频文件的一级子目录。")
    return ready, skipped


_CONFIGURE_HINT = (
    "请先到 Shinsekai 的「API 配置 → TTS 语音」中选择 GPT SoVITS，"
    "并配置 GPT-SoVITS 服务启动路径。"
)


def load_configured_gpt_sovits_root(api_config_path: Path, project_root: Path) -> Path:
    """Read only the GPT-SoVITS path scalar from Shinsekai's API config."""
    try:
        lines = api_config_path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise GptSovitsConfigurationError(f"尚未配置 GPT-SoVITS 目录。{_CONFIGURE_HINT}") from exc
    except OSError as exc:
        raise GptSovitsConfigurationError(f"无法读取 GPT-SoVITS 目录配置：{exc}。{_CONFIGURE_HINT}") from exc

    raw_value = ""
    for line in lines:
        if line.startswith("gpt_sovits_api_path:"):
            raw_value = line.split(":", 1)[1].strip()
            break
    if not raw_value:
        raise GptSovitsConfigurationError(f"尚未配置 GPT-SoVITS 目录。{_CONFIGURE_HINT}")

    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise GptSovitsConfigurationError(f"GPT-SoVITS 目录配置格式无效。{_CONFIGURE_HINT}") from exc
    configured = str(value or "").strip()
    if not configured:
        raise GptSovitsConfigurationError(f"尚未配置 GPT-SoVITS 目录。{_CONFIGURE_HINT}")

    root = Path(os.path.expandvars(configured)).expanduser()
    if not root.is_absolute():
        root = project_root / root
    root = root.resolve(strict=False)
    if not root.is_dir():
        raise GptSovitsConfigurationError(
            f"已配置的 GPT-SoVITS 目录不存在：{root}。{_CONFIGURE_HINT}"
        )
    validate_gpt_sovits_root(root)
    return root


def validate_gpt_sovits_root(root: Path) -> None:
    required = (
        root / "runtime" / "python.exe",
        root / "tools" / "slice_audio.py",
        root / "tools" / "asr" / "fasterwhisper_asr.py",
        root / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json",
        root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml",
        root / "GPT_SoVITS" / "pretrained_models" / "chinese-hubert-base" / "config.json",
        root / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large" / "config.json",
        root / "GPT_SoVITS" / "pretrained_models" / "sv" / "pretrained_eres2netv2w24s4ep4.ckpt",
        root / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Gv2ProPlus.pth",
        root / "GPT_SoVITS" / "pretrained_models" / "v2Pro" / "s2Dv2ProPlus.pth",
        root / "GPT_SoVITS" / "pretrained_models" / "s1v3.ckpt",
    )
    missing = [path.relative_to(root) for path in required if not path.is_file()]
    if missing:
        formatted = "、".join(str(path) for path in missing)
        raise GptSovitsConfigurationError(
            f"已配置的 GPT-SoVITS 目录不可用于批量训练，缺少：{formatted}。{_CONFIGURE_HINT}"
        )


class BatchTrainer:
    """Run the same pipeline wired by GPT-SoVITS webui.py, one speaker at a time."""

    def __init__(
        self,
        source_root: str | Path,
        *,
        gpt_sovits_root: str | Path,
        on_event: Callable[[TrainingEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve()
        self.root = Path(gpt_sovits_root).expanduser().resolve()
        self.on_event = on_event or (lambda event: None)
        self.cancel_event = cancel_event or threading.Event()
        self.force = force
        self.dry_run = dry_run
        self.profile = detect_runtime_profile()
        self.python = self.root / "runtime" / "python.exe"
        self._processes: dict[int, subprocess.Popen[str]] = {}
        self._process_lock = threading.RLock()

    def validate(self) -> tuple[list[Path], list[Path]]:
        validate_gpt_sovits_root(self.root)
        return discover_experiments(self.source_root)

    def cancel(self) -> None:
        self.cancel_event.set()
        self._terminate_all_processes()

    def _terminate_all_processes(self) -> None:
        with self._process_lock:
            processes = list(self._processes.values())
        for process in processes:
            self._terminate_process_tree(process)

    def run(self) -> dict[str, Any]:
        experiments, skipped = self.validate()
        for path in skipped:
            self._emit(f"跳过不含直接音频文件的目录：{path.name}", level="warning")
        self._emit(
            f"发现 {len(experiments)} 个人物目录；模型版本 {MODEL_VERSION}，GPU {self.profile.gpu}，"
            f"batch size {self.profile.batch_size}。"
        )
        completed = 0
        for index, voice_dir in enumerate(experiments, start=1):
            self._check_cancelled()
            self._emit(f"开始人物 {index}/{len(experiments)}：{voice_dir.name}", experiment=voice_dir.name)
            self._run_experiment(voice_dir)
            completed += 1
            self._emit(f"人物 {voice_dir.name} 全部训练完成。", experiment=voice_dir.name)
        return {"completed": completed, "total": len(experiments), "skipped": len(skipped)}

    def _run_experiment(self, voice_dir: Path) -> None:
        experiment = voice_dir.name.rstrip(" ")
        if not experiment:
            raise TrainingError(f"人物目录名不可为空：{voice_dir}")
        state = {} if self.force else self._load_state(experiment)
        recorded_root = str(state.get("source_root", "")).strip()
        if recorded_root and Path(recorded_root).resolve() != self.source_root:
            raise TrainingError(
                f"人物 {experiment} 已有来自另一语音总目录的续跑记录：{recorded_root}。"
                "请改名，或使用独立脚本的 --force 明确重做。"
            )
        completed = set(state.get("completed_stages", []))
        for stage in STAGES:
            self._check_cancelled()
            if stage in completed:
                self._emit(
                    f"已完成，跳过：{STAGE_LABELS[stage]}",
                    experiment=experiment,
                    stage=stage,
                )
                continue
            self._emit(f"开始：{STAGE_LABELS[stage]}", experiment=experiment, stage=stage)
            getattr(self, f"_stage_{stage}")(voice_dir, experiment)
            if not self.dry_run:
                completed.add(stage)
                self._save_state(experiment, completed)
            self._emit(f"完成：{STAGE_LABELS[stage]}", experiment=experiment, stage=stage)

    def _stage_slice(self, voice_dir: Path, experiment: str) -> None:
        output_dir = self.root / "output" / "slicer_opt" / experiment
        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        commands = []
        for part in range(SLICER_PARTS):
            commands.append(
                self._python_command(
                    "tools/slice_audio.py",
                    str(voice_dir),
                    str(output_dir),
                    "-34",
                    "4000",
                    "300",
                    "10",
                    "500",
                    "0.9",
                    "0.25",
                    str(part),
                    str(SLICER_PARTS),
                )
            )
        self._run_commands("语音切分", commands, experiment=experiment, stage="slice")
        if not self.dry_run and not any(output_dir.glob("*.wav")):
            raise TrainingError(f"语音切分没有产生 WAV：{output_dir}")

    def _stage_asr(self, voice_dir: Path, experiment: str) -> None:
        _ = voice_dir
        sliced_dir = self.root / "output" / "slicer_opt" / experiment
        asr_dir = self.root / "output" / "asr_opt"
        list_path = asr_dir / f"{experiment}.list"
        if not self.force and not self.dry_run:
            try:
                count = self._validate_asr_output(list_path, sliced_dir)
            except TrainingError:
                pass
            else:
                self._emit(
                    f"复用已完整生成的 ASR 标注（{count} 条）：{list_path}",
                    level="warning",
                    experiment=experiment,
                    stage="asr",
                )
                return
        if not self.dry_run:
            asr_dir.mkdir(parents=True, exist_ok=True)
        model_size = self._faster_whisper_model_size()
        command = self._python_command(
            "tools/asr/fasterwhisper_asr.py",
            "-i",
            str(sliced_dir),
            "-o",
            str(asr_dir),
            "-s",
            model_size,
            "-l",
            ASR_LANGUAGE,
            "-p",
            self.profile.asr_precision,
        )
        try:
            self._run_command(
                "Faster Whisper ASR",
                command,
                extra_env=asr_local_hub_env(),
                experiment=experiment,
                stage="asr",
            )
        except CommandFailed as exc:
            if os.name != "nt" or exc.return_code != 0xC0000409:
                raise
            # ctranslate2/FunASR native libraries can fast-fail while Windows is
            # unloading DLLs, after execute_asr has already flushed the result.
            count = self._validate_asr_output(list_path, sliced_dir)
            self._emit(
                f"ASR 已生成全部 {count} 条标注；忽略原生库退出清理阶段的 0xC0000409。",
                level="warning",
                experiment=experiment,
                stage="asr",
            )
        if not self.dry_run:
            self._validate_asr_output(list_path, sliced_dir)

    def _faster_whisper_model_size(self) -> str:
        expected = self.root / "tools" / "asr" / "models" / f"faster-whisper-{ASR_MODEL_SIZE}"
        prepared = prepare_local_faster_whisper(self.root, link=not self.dry_run)
        if faster_whisper_model_ready(expected):
            self._emit(f"使用本地 Faster Whisper：{expected}", stage="asr")
            return f"{ASR_MODEL_SIZE}-local"
        if prepared is not None:
            self._emit(f"使用本机已缓存的 Faster Whisper：{prepared}", stage="asr")
        elif not self.dry_run:
            self._emit(missing_faster_whisper_hint(expected), level="warning", stage="asr")
        return ASR_MODEL_SIZE

    def _stage_prepare(self, voice_dir: Path, experiment: str) -> None:
        _ = voice_dir
        sliced_dir = self.root / "output" / "slicer_opt" / experiment
        list_path = self.root / "output" / "asr_opt" / f"{experiment}.list"
        exp_dir = self.root / "logs" / experiment
        if not self.dry_run:
            exp_dir.mkdir(parents=True, exist_ok=True)
        common = {
            "inp_text": str(list_path),
            "inp_wav_dir": str(sliced_dir),
            "exp_name": experiment,
            "opt_dir": str(exp_dir),
            "i_part": "0",
            "all_parts": "1",
            "_CUDA_VISIBLE_DEVICES": self.profile.gpu,
            "is_half": str(self.profile.use_half),
            "version": MODEL_VERSION,
        }

        text_env = dict(common)
        text_env["bert_pretrained_dir"] = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self._run_command(
            "1A 文本分词与特征提取",
            self._python_command("GPT_SoVITS/prepare_datasets/1-get-text.py"),
            extra_env=text_env,
            experiment=experiment,
            stage="prepare",
        )
        if not self.dry_run:
            self._merge_text_parts(exp_dir)

        hubert_env = dict(common)
        hubert_env["cnhubert_base_dir"] = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        hubert_env["sv_path"] = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"
        self._run_command(
            "1B Hubert/WAV32K 特征提取",
            self._python_command("GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py"),
            extra_env=hubert_env,
            experiment=experiment,
            stage="prepare",
        )
        self._run_command(
            "1B v2ProPlus 音色特征提取",
            self._python_command("GPT_SoVITS/prepare_datasets/2-get-sv.py"),
            extra_env=hubert_env,
            experiment=experiment,
            stage="prepare",
        )

        semantic_env = dict(common)
        semantic_env.update(
            {
                "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
                "s2config_path": "GPT_SoVITS/configs/s2v2ProPlus.json",
            }
        )
        self._run_command(
            "1C 语义 Token 提取",
            self._python_command("GPT_SoVITS/prepare_datasets/3-get-semantic.py"),
            extra_env=semantic_env,
            experiment=experiment,
            stage="prepare",
        )
        if not self.dry_run:
            self._merge_semantic_parts(exp_dir)
            self._validate_prepared_dataset(exp_dir)

    def _stage_sovits(self, voice_dir: Path, experiment: str) -> None:
        _ = voice_dir
        exp_dir = self.root / "logs" / experiment
        checkpoint_dir = exp_dir / f"logs_s2_{MODEL_VERSION}"
        config = self._read_json(self.root / "GPT_SoVITS" / "configs" / "s2v2ProPlus.json")
        train = config["train"]
        train.update(
            {
                "batch_size": self.profile.batch_size,
                "epochs": SOVITS_EPOCHS,
                "text_low_lr_rate": 0.4,
                "pretrained_s2G": "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth",
                "pretrained_s2D": "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth",
                "if_save_latest": True,
                "if_save_every_weights": True,
                "save_every_epoch": SOVITS_SAVE_EVERY,
                "gpu_numbers": self.profile.gpu,
                "grad_ckpt": False,
                "lora_rank": "32",
                "fp16_run": self.profile.use_half,
            }
        )
        config["model"]["version"] = MODEL_VERSION
        config["data"]["exp_dir"] = str(exp_dir)
        config["s2_ckpt_dir"] = str(exp_dir)
        config["save_weight_dir"] = "SoVITS_weights_v2ProPlus"
        config["name"] = experiment
        config["version"] = MODEL_VERSION
        config_path = self.root / "TEMP" / "batch_trainer" / experiment / "s2.json"
        if not self.dry_run:
            # webui.py/open1Ba creates this directory before s2_train.py.
            # GPT-SoVITS' utils.my_save does not create its destination parent.
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (self.root / "SoVITS_weights_v2ProPlus").mkdir(parents=True, exist_ok=True)
            # Upstream opens this without an explicit encoding. Escaping Chinese
            # characters also makes the config safe for legacy GBK readers.
            self._write_json(config_path, config, ensure_ascii=True)
        self._run_command(
            "SoVITS 训练",
            self._python_command("GPT_SoVITS/s2_train.py", "--config", str(config_path)),
            experiment=experiment,
            stage="sovits",
        )
        if not self.dry_run and not self._has_weight(
            self.root / "SoVITS_weights_v2ProPlus", f"{experiment}_e", ".pth"
        ):
            raise TrainingError(f"SoVITS 训练结束但未找到人物权重：{experiment}")

    def _stage_gpt(self, voice_dir: Path, experiment: str) -> None:
        _ = voice_dir
        exp_dir = self.root / "logs" / experiment
        config_path = self.root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        train = config["train"]
        train.update(
            {
                "batch_size": self.profile.batch_size,
                "epochs": GPT_EPOCHS,
                "save_every_n_epoch": GPT_SAVE_EVERY,
                "if_save_every_weights": True,
                "if_save_latest": True,
                "if_dpo": False,
                "half_weights_save_dir": "GPT_weights_v2ProPlus",
                "exp_name": experiment,
            }
        )
        if not self.profile.use_half:
            train["precision"] = "32"
        config["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        config["train_semantic_path"] = str(exp_dir / "6-name2semantic.tsv")
        config["train_phoneme_path"] = str(exp_dir / "2-name2text.txt")
        config["output_dir"] = str(exp_dir / f"logs_s1_{MODEL_VERSION}")
        temp_path = self.root / "TEMP" / "batch_trainer" / experiment / "s1.yaml"
        if not self.dry_run:
            (self.root / "GPT_weights_v2ProPlus").mkdir(parents=True, exist_ok=True)
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(yaml.safe_dump(config, allow_unicode=False, sort_keys=False), encoding="utf-8")
        self._run_command(
            "GPT 训练",
            self._python_command("GPT_SoVITS/s1_train.py", "--config_file", str(temp_path)),
            extra_env={"_CUDA_VISIBLE_DEVICES": self.profile.gpu, "hz": "25hz"},
            experiment=experiment,
            stage="gpt",
        )
        if not self.dry_run and not self._has_weight(
            self.root / "GPT_weights_v2ProPlus", f"{experiment}-e", ".ckpt"
        ):
            raise TrainingError(f"GPT 训练结束但未找到人物权重：{experiment}")

    def _python_command(self, script: str, *arguments: str) -> list[str]:
        # This embedded runtime's spawn workers ignore PYTHONUTF8. CPython
        # propagates an explicit -X utf8 flag to those workers instead.
        return [str(self.python), "-X", "utf8", "-u", "-s", script, *arguments]

    def _run_commands(
        self,
        label: str,
        commands: Sequence[Sequence[str]],
        *,
        experiment: str,
        stage: str,
    ) -> None:
        if self.dry_run:
            for command in commands:
                self._run_command(label, command, experiment=experiment, stage=stage)
            return
        with ThreadPoolExecutor(max_workers=len(commands), thread_name_prefix="gptsovits-stage") as pool:
            futures = [
                pool.submit(self._run_command, f"{label} {index + 1}/{len(commands)}", command,
                            experiment=experiment, stage=stage)
                for index, command in enumerate(commands)
            ]
            try:
                for future in as_completed(futures):
                    future.result()
            except Exception:
                self._terminate_all_processes()
                for future in futures:
                    try:
                        future.result()
                    except Exception:
                        pass
                if self.cancel_event.is_set():
                    raise TrainingCancelled("训练已停止。")
                raise

    def _run_command(
        self,
        label: str,
        command: Sequence[str],
        *,
        extra_env: dict[str, str] | None = None,
        experiment: str,
        stage: str,
    ) -> None:
        self._check_cancelled()
        rendered = subprocess.list2cmdline([str(part) for part in command])
        self._emit(f"{label}：{rendered}", experiment=experiment, stage=stage)
        if self.dry_run:
            return
        env = self._base_env()
        if extra_env:
            env.update({key: str(value) for key, value in extra_env.items()})
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            list(command),
            cwd=self.root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creationflags,
        )
        with self._process_lock:
            self._processes[process.pid] = process
        output_queue: queue.Queue[str | None] = queue.Queue()

        def _read_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                output_queue.put(line.rstrip())
            output_queue.put(None)

        reader = threading.Thread(target=_read_output, name=f"gptsovits-output-{process.pid}", daemon=True)
        reader.start()
        reader_finished = False
        try:
            while process.poll() is None or not reader_finished:
                if self.cancel_event.is_set():
                    self._terminate_process_tree(process)
                    raise TrainingCancelled("训练已停止。")
                try:
                    line = output_queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    reader_finished = True
                elif line:
                    self._emit(line, experiment=experiment, stage=stage)
            return_code = process.wait()
            if return_code != 0:
                raise CommandFailed(label, return_code)
        finally:
            with self._process_lock:
                self._processes.pop(process.pid, None)
            if process.stdout is not None:
                process.stdout.close()

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        runtime_dir = str(self.root / "runtime")
        env["PATH"] = runtime_dir + os.pathsep + env.get("PATH", "")
        temp_dir = self.root / "TEMP"
        temp_dir.mkdir(parents=True, exist_ok=True)
        env.update(
            {
                "TEMP": str(temp_dir),
                "TMP": str(temp_dir),
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "is_half": str(self.profile.use_half),
                "version": MODEL_VERSION,
            }
        )
        return env

    def _terminate_process_tree(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True,
                    timeout=15,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
            else:
                process.terminate()
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except OSError:
                pass

    def _state_path(self, experiment: str) -> Path:
        return self.root / "output" / "batch_train_state" / f"{experiment}.json"

    def _load_state(self, experiment: str) -> dict[str, Any]:
        path = self._state_path(experiment)
        if not path.is_file():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_state(self, experiment: str, completed: Iterable[str]) -> None:
        path = self._state_path(experiment)
        payload = {
            "experiment": experiment,
            "version": MODEL_VERSION,
            "source_root": str(self.source_root),
            "completed_stages": [stage for stage in STAGES if stage in set(completed)],
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        self._write_json(path, payload)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TrainingError(f"配置文件不是 JSON 对象：{path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any], *, ensure_ascii: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=ensure_ascii, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _merge_text_parts(exp_dir: Path) -> None:
        part = exp_dir / "2-name2text-0.txt"
        output = exp_dir / "2-name2text.txt"
        if part.is_file():
            text = part.read_text(encoding="utf-8").strip()
            output.write_text(text + ("\n" if text else ""), encoding="utf-8")
            part.unlink()
        if not output.is_file() or not output.read_text(encoding="utf-8").strip():
            raise TrainingError(f"文本特征提取没有产生有效结果：{output}")

    @staticmethod
    def _merge_semantic_parts(exp_dir: Path) -> None:
        part = exp_dir / "6-name2semantic-0.tsv"
        output = exp_dir / "6-name2semantic.tsv"
        if part.is_file():
            body = part.read_text(encoding="utf-8").strip()
            text = "item_name\tsemantic_audio\n" + body + ("\n" if body else "")
            output.write_text(text, encoding="utf-8")
            part.unlink()
        if not output.is_file() or len(output.read_text(encoding="utf-8").splitlines()) < 2:
            raise TrainingError(f"语义特征提取没有产生有效结果：{output}")

    @staticmethod
    def _validate_asr_output(path: Path, sliced_dir: Path) -> int:
        if not path.is_file():
            raise TrainingError(f"ASR 未生成标注文件：{path}")
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines or any(len(line.split("|", 3)) != 4 for line in lines):
            raise TrainingError(f"ASR 标注为空或格式错误：{path}")
        expected = {item.name for item in sliced_dir.glob("*.wav") if item.is_file()}
        actual = {Path(line.split("|", 1)[0]).name for line in lines}
        if not expected or len(lines) != len(expected) or actual != expected:
            raise TrainingError(
                f"ASR 标注不完整：切片 {len(expected)} 个，标注 {len(lines)} 条，文件名匹配 {len(actual & expected)} 个。"
            )
        return len(lines)

    @staticmethod
    def _validate_prepared_dataset(exp_dir: Path) -> None:
        required_files = (exp_dir / "2-name2text.txt", exp_dir / "6-name2semantic.tsv")
        required_dirs = (exp_dir / "4-cnhubert", exp_dir / "5-wav32k", exp_dir / "7-sv_cn")
        missing = [str(path) for path in required_files if not path.is_file() or path.stat().st_size == 0]
        missing.extend(str(path) for path in required_dirs if not path.is_dir() or not any(path.iterdir()))
        bert_dir = exp_dir / "3-bert"
        if not bert_dir.is_dir():
            missing.append(str(bert_dir))
        if missing:
            raise TrainingError("训练集格式化结果不完整：\n" + "\n".join(missing))

    @staticmethod
    def _has_weight(directory: Path, prefix: str, suffix: str) -> bool:
        if not directory.is_dir():
            return False
        return any(path.is_file() and path.name.startswith(prefix) and path.name.endswith(suffix) for path in directory.iterdir())

    def _check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise TrainingCancelled("训练已停止。")

    def _emit(self, message: str, *, level: str = "info", experiment: str = "", stage: str = "") -> None:
        self.on_event(TrainingEvent(message=message, level=level, experiment=experiment, stage=stage))
