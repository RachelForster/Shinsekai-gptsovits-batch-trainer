from __future__ import annotations

import json
import re
import threading
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sdk.plugin import PluginBase
from sdk.plugin_host_context import PluginHostContext
from sdk.register import PluginCapabilityRegistry
from sdk.types import FrontendConfigAction, FrontendConfigContribution, FrontendPageContribution
from plugin_system.requirements.install import pip_python_executable

from .character_import import (
    character_config_path,
    import_trained_voices,
    load_character_documents,
    save_character_documents,
)
from .speaker_job import SpeakerSortJob, console_python
from .speaker_sorter import DEFAULT_MODEL, SortOptions
from .trainer import (
    BatchTrainer,
    GptSovitsConfigurationError,
    TrainingCancelled,
    TrainingEvent,
    load_configured_gpt_sovits_root,
)


PLUGIN_ID = "local.gpt_sovits_batch_trainer"
PAGE_ID = "gpt_sovits_batch_trainer.tools"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TrainingController:
    def __init__(self, plugin_root: Path, api_config_path: Path, project_root: Path) -> None:
        self.plugin_root = plugin_root
        self.config_path = plugin_root / "config.json"
        self.api_config_path = api_config_path
        self.project_root = project_root
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._trainer: BatchTrainer | SpeakerSortJob | None = None
        self._cancel_event = threading.Event()
        self._source_root = self._load_source_root()
        self._sort_values = self._load_sort_values()
        self._job_kind = "training"
        self._sorted_root = ""
        self._state = "idle"
        self._current_experiment = ""
        self._current_stage = ""
        self._summary = "等待开始。"
        self._lines: deque[str] = deque(maxlen=240)

    def save_source_root(self, source_root: str) -> None:
        cleaned = str(source_root or "").strip().strip('"')
        with self._lock:
            self._source_root = cleaned
        self.plugin_root.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(".json.tmp")
        with self._lock:
            temporary.write_text(json.dumps({"source_root": cleaned, **self._sort_values}, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.config_path)

    def _load_sort_values(self) -> dict[str, Any]:
        defaults = {
            "sort_source": "", "sort_output": "",
            "sort_model": DEFAULT_MODEL, "sort_device": "auto", "sort_threshold": 0.72,
            "sort_min_seconds": 1.5, "sort_max_seconds": 12.0,
            "sort_max_speakers": 0,
        }
        try:
            saved = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                defaults.update({key: saved[key] for key in defaults if key in saved})
                if defaults["sort_model"] == "pyannote/speaker-diarization-community-1":
                    defaults["sort_model"] = DEFAULT_MODEL
        except (OSError, ValueError):
            pass
        return defaults

    def save_values(self, values: Mapping[str, Any]) -> None:
        with self._lock:
            self._sort_values.update({key: values[key] for key in self._sort_values if key in values})
            self.save_source_root(str(values.get("source_root", self._source_root)))

    def start_sort(self, values: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._state in ("starting", "running", "stopping") or (self._thread and self._thread.is_alive()):
                raise RuntimeError("已有训练或说话人处理任务正在运行。")
            self._state = "starting"
            self._job_kind = "sorting"
            self._sorted_root = ""
            self._lines.clear()
        try:
            self.save_values(values)
            settings = self._sort_values.copy()
            cleaned = str(settings["sort_source"]).strip().strip('"')
            if not cleaned:
                raise ValueError("请先填写全集人声目录。")
            source = Path(cleaned).expanduser().resolve()
            output = str(settings["sort_output"]).strip().strip('"')
            output = Path(output).expanduser().resolve() if output else source.with_name(source.name + "_speakers")
            model = str(settings["sort_model"]).strip().strip('"') or DEFAULT_MODEL
            if Path(model).expanduser().exists():
                model = str(Path(model).expanduser().resolve())
            try:
                max_speakers = int(str(settings["sort_max_speakers"]).strip() or "0")
                if max_speakers < 0:
                    raise ValueError
            except (TypeError, ValueError):
                raise ValueError("最多说话人数必须为非负整数，0 表示不限。") from None
            options = SortOptions(source, output, model, str(settings["sort_device"]),
                                  float(settings["sort_threshold"]), float(settings["sort_min_seconds"]),
                                  float(settings["sort_max_seconds"]), max_speakers=max_speakers)
            job = SpeakerSortJob(options, pip_python_executable(), self._on_event, api_config_path=self.api_config_path)
            files = job.validate()
            with self._lock:
                if self._state == "stopping":
                    job.cancel()
                self._trainer = job
                self._state = "running"
                self._summary = f"已排队 {len(files)} 个音频文件，逐集分析后跨集匹配说话人。"
                self._thread = threading.Thread(target=self._worker, name="speaker-sorter", daemon=True)
                self._thread.start()
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._summary = str(exc)
            raise
        return self.snapshot()

    def start(self, source_root: str) -> dict[str, Any]:
        cleaned = str(source_root or "").strip().strip('"')
        if not cleaned:
            raise ValueError("请先填写语音总目录。")
        with self._lock:
            if self._state in ("starting", "running", "stopping") or (self._thread and self._thread.is_alive()):
                raise RuntimeError("已有训练或说话人处理任务正在运行。")
            self._state = "starting"
            self._job_kind = "training"
            self._summary = "正在检查目录和 GPT-SoVITS 安装……"
            self._lines.clear()
        cancel_event = threading.Event()
        try:
            self.save_source_root(cleaned)
            gpt_sovits_root = self._configured_root()
            trainer = BatchTrainer(
                cleaned,
                gpt_sovits_root=gpt_sovits_root,
                on_event=self._on_event,
                cancel_event=cancel_event,
            )
            experiments, skipped = trainer.validate()
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._summary = str(exc)
            raise
        with self._lock:
            if self._state == "stopping":
                trainer.cancel()
            self._cancel_event = cancel_event
            self._trainer = trainer
            self._state = "running"
            self._summary = f"已排队 {len(experiments)} 个人物，跳过 {len(skipped)} 个空目录。"
            self._thread = threading.Thread(target=self._worker, name="gpt-sovits-batch-trainer", daemon=True)
            self._thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            trainer = self._trainer
            running = self._state in ("starting", "running", "stopping")
            if running:
                self._state = "stopping"
                self._summary = "正在停止当前子进程……"
        if trainer is not None and running:
            trainer.cancel()
        return self.snapshot()

    def apply_trained_voices(self, source_root: str) -> dict[str, Any]:
        cleaned = str(source_root or "").strip().strip('"')
        with self._lock:
            if self._state in ("starting", "running", "stopping") or (self._thread and self._thread.is_alive()):
                raise RuntimeError("训练或说话人处理进行中，请完成或停止后再写入人物配置。")
        try:
            if cleaned:
                self.save_source_root(cleaned)
            gpt_sovits_root = self._configured_root()
            config_path = character_config_path(self.project_root)
            result = import_trained_voices(
                gpt_sovits_root,
                source_root=cleaned,
                characters=load_character_documents(config_path),
            )
            save_character_documents(config_path, result.characters)
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._summary = str(exc)
            raise
        self._append_line(f"[人物配置] {result.summary}")
        for item in result.imported:
            action = "新建" if item.created else "更新"
            self._append_line(
                f"[人物配置] {action} {item.name}：{Path(item.gpt_model_path).name} / "
                f"{Path(item.sovits_model_path).name} / {Path(item.refer_audio_path).name}"
            )
        for item in result.skipped:
            self._append_line(f"[人物配置] 跳过 {item.name}：{item.reason}")
        with self._lock:
            self._state = "completed"
            self._summary = result.summary
        return self.snapshot()

    def shutdown(self) -> None:
        self.stop()

    def snapshot(self) -> dict[str, Any]:
        try:
            configured_root = str(self._configured_root())
            configuration_error = ""
        except GptSovitsConfigurationError as exc:
            configured_root = ""
            configuration_error = str(exc)
        with self._lock:
            running = self._state in ("starting", "running", "stopping")
            summary = self._summary
            if not running and self._state == "idle" and configuration_error:
                summary = configuration_error
            return {
                "source_root": self._source_root,
                **self._sort_values,
                "sort_python": str(console_python(pip_python_executable())),
                "job_kind": self._job_kind,
                "sorted_root": self._sorted_root,
                "gpt_sovits_root": configured_root,
                "configuration_error": configuration_error,
                "can_start": not configuration_error,
                "state": self._state,
                "running": running,
                "current_experiment": self._current_experiment,
                "current_stage": self._current_stage,
                "summary": summary,
                "progress": "\n".join(self._lines),
            }

    def _configured_root(self) -> Path:
        return load_configured_gpt_sovits_root(self.api_config_path, self.project_root)

    def _worker(self) -> None:
        assert self._trainer is not None
        try:
            result = self._trainer.run()
        except TrainingCancelled:
            with self._lock:
                self._state = "cancelled"
                self._summary = ("说话人处理已停止；再次开始会复用已完成的逐集分析。"
                                 if self._job_kind == "sorting" else "训练已停止；再次开始会从成功记录的阶段继续。")
        except Exception as exc:
            self._append_line(f"[失败] {exc}")
            with self._lock:
                self._state = "failed"
                self._summary = f"任务失败：{exc}"
        else:
            with self._lock:
                self._state = "completed"
                if self._job_kind == "sorting":
                    self._sorted_root = result["speakers_dir"] if result["speakers"] else ""
                    self._summary = f"说话人分目录完成：{result['speakers']} 个目录，{result['clips']} 段。请试听核对后训练。"
                    if result.get("review_clips"):
                        self._summary += f"其中 {result['review_clips']} 段声纹不足，单独保存在 _needs_review 目录。"
                else:
                    self._summary = f"批量训练完成：{result['completed']}/{result['total']} 个人物。"
        finally:
            with self._lock:
                self._trainer = None
                self._current_experiment = ""
                self._current_stage = ""

    def _on_event(self, event: TrainingEvent) -> None:
        prefix = ""
        if event.experiment:
            prefix += f"[{event.experiment}]"
        if event.stage:
            prefix += f"[{event.stage}]"
        self._append_line(f"{prefix} {event.message}".strip())
        with self._lock:
            if event.experiment:
                self._current_experiment = event.experiment
            if event.stage:
                self._current_stage = event.stage
            self._summary = event.message

    def _append_line(self, line: str) -> None:
        cleaned = _ANSI_ESCAPE.sub("", line).replace("\r", "").strip()
        if not cleaned:
            return
        with self._lock:
            self._lines.append(cleaned)

    def _load_source_root(self) -> str:
        if not self.config_path.is_file():
            return ""
        try:
            value = json.loads(self.config_path.read_text(encoding="utf-8"))
            return str(value.get("source_root", "")).strip() if isinstance(value, dict) else ""
        except (OSError, json.JSONDecodeError):
            return ""


class GptSovitsBatchTrainerPlugin(PluginBase):
    def __init__(self) -> None:
        self._controller: TrainingController | None = None

    @property
    def plugin_id(self) -> str:
        return PLUGIN_ID

    @property
    def plugin_version(self) -> str:
        return "0.2.5"

    @property
    def plugin_name(self) -> str:
        return "GPT-SoVITS 批量训练"

    @property
    def plugin_description(self) -> str:
        return "全集人声按说话人分目录，并按人物目录完成 GPT-SoVITS v2ProPlus 批量训练。"

    @property
    def plugin_author(self) -> str:
        return "Shinsekai Local"

    @property
    def priority(self) -> int:
        return 95

    def initialize(
        self,
        register: PluginCapabilityRegistry,
        plugin_root: Path,
        host: PluginHostContext,
    ) -> None:
        package_root = Path(__file__).parent
        project_data_dir = host.project_data_dir.expanduser()
        if not project_data_dir.is_absolute():
            project_data_dir = Path.cwd() / project_data_dir
        project_data_dir = project_data_dir.resolve(strict=False)
        self._controller = TrainingController(
            plugin_root,
            project_data_dir / "config" / "api.yaml",
            project_data_dir.parent,
        )

        def _load_values() -> Mapping[str, Any]:
            assert self._controller is not None
            return self._controller.snapshot()

        def _save_values(values: Mapping[str, Any]) -> None:
            assert self._controller is not None
            self._controller.save_values(values)

        def _start(values: Mapping[str, Any]) -> Mapping[str, Any]:
            assert self._controller is not None
            return self._controller.start(str(values.get("source_root", "")))

        def _stop(values: Mapping[str, Any]) -> Mapping[str, Any]:
            _ = values
            assert self._controller is not None
            return self._controller.stop()

        def _sort(values: Mapping[str, Any]) -> Mapping[str, Any]:
            assert self._controller is not None
            return self._controller.start_sort(values)

        def _apply_voices(values: Mapping[str, Any]) -> Mapping[str, Any]:
            assert self._controller is not None
            return self._controller.apply_trained_voices(str(values.get("source_root", "")))

        register.register_frontend_config_page(
            FrontendConfigContribution(
                page_id=PAGE_ID,
                title="GPT-SoVITS 批量训练",
                kind="tools",
                description="先将全集人声按说话人分目录，试听核对后按人物目录批量训练。",
                schema=[
                    {
                        "id": "source",
                        "title": "训练来源",
                        "fields": [
                            {
                                "key": "source_root",
                                "label": "语音总目录",
                                "type": "text",
                                "pathKind": "directory",
                                "placeholder": r"例如 D:\voices（其下每个一级目录代表一个人物）",
                            }
                        ],
                    },
                    {
                        "id": "speaker_sort", "title": "全集人声分目录",
                        "fields": [
                            {"key": "sort_source", "label": "全集人声目录", "type": "text", "pathKind": "directory"},
                            {"key": "sort_output", "label": "输出根目录（可留空）", "type": "text", "pathKind": "directory"},
                            {"key": "sort_model", "label": "说话人模型 3.1 或本地配置", "type": "text"},
                            {"key": "sort_device", "label": "设备（auto / cpu / cuda）", "type": "text"},
                            {"key": "sort_threshold", "label": "跨集合并阈值", "type": "number"},
                            {"key": "sort_min_seconds", "label": "最短切片秒数", "type": "number"},
                            {"key": "sort_max_seconds", "label": "最长切片秒数", "type": "number"},
                        ],
                    }
                ],
                load_values=_load_values,
                save_values=_save_values,
                actions=[
                    FrontendConfigAction(
                        id="sort", label="开始按说话人分目录", variant="primary", order=5, run=_sort,
                        description="逐集分析已去配乐的人声，跨集合并声纹并导出 WAV。",
                    ),
                    FrontendConfigAction(
                        id="start",
                        label="开始批量训练",
                        description="在后台依次训练所有人物。",
                        variant="primary",
                        confirm="将开始耗时较长的模型训练，确定继续吗？",
                        order=10,
                        run=_start,
                    ),
                    FrontendConfigAction(
                        id="apply_voices",
                        label="填入人物配置",
                        description="把已训练的模型、3–10 秒参考音频和 ASR 文本写入 Shinsekai 人物配置。",
                        variant="ghost",
                        confirm="将把已训练的 GPT/SoVITS 模型和参考音频写入人物配置。已有人物只更新语音字段，没有则按目录名新建。确定继续吗？",
                        order=15,
                        run=_apply_voices,
                    ),
                    FrontendConfigAction(
                        id="stop",
                        label="停止当前任务",
                        description="终止当前训练或说话人处理子进程，保留已完成的记录。",
                        variant="danger",
                        confirm="确定停止当前任务吗？",
                        order=20,
                        run=_stop,
                    ),
                ],
                order=30,
            )
        )
        register.register_frontend_page(
            FrontendPageContribution(
                page_id=PAGE_ID,
                title="GPT-SoVITS 批量训练",
                kind="tools",
                description="批量训练人物语音模型。",
                entry=(package_root / "frontend" / "dist" / "index.html").as_posix(),
                order=30,
            )
        )

    def shutdown(self) -> None:
        if self._controller is not None:
            self._controller.shutdown()
