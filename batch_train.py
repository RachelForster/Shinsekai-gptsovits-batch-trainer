from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__:
    from .trainer import (
        BatchTrainer,
        TrainingCancelled,
        TrainingError,
        TrainingEvent,
        load_configured_gpt_sovits_root,
    )
else:
    # GPT-SoVITS ships an isolated embeddable Python whose ._pth file does not
    # automatically add the script directory to sys.path.
    script_dir = str(Path(__file__).resolve().parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from trainer import (
        BatchTrainer,
        TrainingCancelled,
        TrainingError,
        TrainingEvent,
        load_configured_gpt_sovits_root,
    )


def _print_event(event: TrainingEvent) -> None:
    prefix = ""
    if event.experiment:
        prefix += f"[{event.experiment}]"
    if event.stage:
        prefix += f"[{event.stage}]"
    print(f"{prefix} {event.message}".strip(), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="依次为指定目录下的每个人物子目录训练 GPT-SoVITS v2ProPlus 模型。"
    )
    parser.add_argument("source_root", type=Path, help="包含多个人物语音子目录的总目录")
    parser.add_argument("--dry-run", action="store_true", help="只检查目录并打印流程，不启动模型或训练")
    parser.add_argument("--force", action="store_true", help="忽略已完成阶段记录，从语音切分重新执行")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project_root = Path(__file__).resolve().parents[2]
        gpt_sovits_root = load_configured_gpt_sovits_root(
            project_root / "data" / "config" / "api.yaml",
            project_root,
        )
        trainer = BatchTrainer(
            args.source_root,
            gpt_sovits_root=gpt_sovits_root,
            on_event=_print_event,
            force=args.force or args.dry_run,
            dry_run=args.dry_run,
        )
        result = trainer.run()
    except KeyboardInterrupt:
        if "trainer" in locals():
            trainer.cancel()
        print("已请求停止当前训练。", file=sys.stderr)
        return 130
    except TrainingCancelled as exc:
        print(str(exc), file=sys.stderr)
        return 130
    except TrainingError as exc:
        print(f"训练失败：{exc}", file=sys.stderr)
        return 1
    print(f"批量训练完成：{result['completed']}/{result['total']} 个人物。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
