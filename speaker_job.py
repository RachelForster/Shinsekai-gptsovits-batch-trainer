"""Run cancellable diarization in Shinsekai's shared Python runtime."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
from pathlib import Path

from .hf_auth import resolve_hf_token
from .speaker_sorter import SortOptions
from .trainer import TrainingCancelled, TrainingEvent


def console_python(executable: Path) -> Path:
    """Use the same environment's console executable for reliable piped output."""
    if executable.name.casefold() == "pythonw.exe":
        candidate = executable.with_name("python.exe")
        if candidate.is_file():
            return candidate
    return executable


class SpeakerSortJob:
    def __init__(self, options: SortOptions, python: Path, on_event, *, api_config_path: Path | None = None) -> None:
        self.options, self.python, self.on_event = options, console_python(python), on_event
        self.api_config_path = api_config_path or Path(__file__).resolve().parents[2] / "data" / "config" / "api.yaml"
        self.cancelled = threading.Event()
        self.process = None
        self._lock = threading.Lock()

    def validate(self) -> list[Path]:
        if not self.python.is_file():
            raise ValueError("新世界的 Python 运行时不存在，请在新世界的运行环境设置中修复。")
        return self.options.validate()

    def cancel(self) -> None:
        self.cancelled.set()
        with self._lock:
            process = self.process
        if process is not None:
            self._terminate(process)

    @staticmethod
    def _terminate(process) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
            else:
                process.kill()

    def run(self) -> dict:
        self.validate()
        if self.cancelled.is_set():
            raise TrainingCancelled("说话人分目录已停止。")
        options = self.options
        command = [str(self.python), "-X", "utf8", "-u", str(Path(__file__).with_name("batch_diarize.py")),
                   str(options.source), "--output", str(options.output), "--model", options.model,
                   "--device", options.device, "--threshold", str(options.threshold),
                   "--min-seconds", str(options.min_seconds), "--max-seconds", str(options.max_seconds),
                   "--max-speakers", str(options.max_speakers),
                   "--edge-trim", str(options.edge_trim), "--api-config", str(self.api_config_path)]
        env = os.environ.copy()
        token = resolve_hf_token(self.api_config_path, env)
        if token:
            env["HF_TOKEN"] = token
        env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1", PYANNOTE_METRICS_ENABLED="0")
        # Use the interpreter's own site-packages, not shell Python overrides.
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        process = subprocess.Popen(command, cwd=Path(__file__).parent, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                                   start_new_session=os.name != "nt")
        with self._lock:
            self.process = process
        lines = queue.Queue()

        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line.rstrip())
            finally:
                lines.put(None)

        reader = threading.Thread(target=read_output, daemon=True, name="speaker-sort-output")
        reader.start()
        report = None
        finished = False
        last_line = ""
        try:
            while not finished or process.poll() is None:
                if self.cancelled.is_set():
                    self._terminate(process)
                    raise TrainingCancelled("说话人分目录已停止。")
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    finished = True
                elif line.startswith("SPEAKER_SORT_RESULT="):
                    report = json.loads(line.partition("=")[2])
                elif line:
                    token = env.get("HF_TOKEN")
                    if token:
                        line = line.replace(token, "[REDACTED]")
                    last_line = line
                    self.on_event(TrainingEvent(line, stage="说话人分目录"))
            if self.cancelled.is_set():
                raise TrainingCancelled("说话人分目录已停止。")
            if process.wait() != 0 or report is None or report.get("status") != "completed":
                raise RuntimeError(f"说话人处理脚本未完成。{last_line}")
            return report
        finally:
            if process.poll() is None:
                self._terminate(process)
            reader.join(timeout=3)
            process.stdout.close()
            with self._lock:
                self.process = None
