"""Standalone entry point using Shinsekai's shared Python environment."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__:
    from .hf_auth import resolve_hf_token
    from .speaker_sorter import DEFAULT_MODEL, SortOptions, run_sort
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hf_auth import resolve_hf_token
    from speaker_sorter import DEFAULT_MODEL, SortOptions, run_sort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="逐集分析已去配乐的人声，跨集合并同一说话人并导出训练切片。")
    parser.add_argument("source", type=Path, help="全集人声音频目录，递归处理子目录")
    parser.add_argument("--output", type=Path, help="输出根目录，默认是输入目录旁的 <目录名>_speakers")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="speaker-diarization-3.1 模型名或本地配置")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threshold", type=float, default=0.72, help="跨集合并余弦相似度，越高越保守，默认 0.72")
    parser.add_argument("--max-speakers", type=int, default=0, help="最终说话人目录数上限；0 不限，超出时按声纹继续合并")
    parser.add_argument("--min-seconds", type=float, default=1.5)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--edge-trim", type=float, default=0.05, help="两端各裁剪秒数，避免相邻人声泄漏")
    parser.add_argument("--reanalyze", action="store_true", help="忽略逐集分析缓存（如本地模型权重已更换）")
    parser.add_argument("--dry-run", action="store_true", help="仅检查目录和参数，无需安装模型依赖")
    parser.add_argument("--api-config", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "config" / "api.yaml",
                        help="新世界 API 配置路径，用于读取已保存的 Hugging Face Token")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else source.with_name(source.name + "_speakers")
    model = args.model
    if Path(model).expanduser().exists():
        model = str(Path(model).expanduser().resolve())
    options = SortOptions(source, output, model, args.device, args.threshold,
                          args.min_seconds, args.max_seconds, args.edge_trim, args.reanalyze, args.max_speakers)
    # No telemetry is needed for local dataset preparation.
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"
    try:
        files = options.validate()
        if args.dry_run:
            print(f"发现 {len(files)} 个文件；输出：{output}", flush=True)
            for path in files:
                print(path.relative_to(source), flush=True)
            return 0
        token = resolve_hf_token(args.api_config.expanduser().resolve(), os.environ)
        if token:
            os.environ["HF_TOKEN"] = token
        report = run_sort(options, lambda message: print(message, flush=True))
        print("SPEAKER_SORT_RESULT=" + json.dumps(report, ensure_ascii=False), flush=True)
        return 0
    except KeyboardInterrupt:
        print("已停止；已完成的逐集分析可在下次运行时复用。", file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc)
        token = os.environ.get("HF_TOKEN")
        if token:
            message = message.replace(token, "[REDACTED]")
        print(f"说话人分目录失败：{message}\n首次使用请按 README 在新世界环境安装插件依赖，并完成模型 3.1 和 segmentation-3.0 的授权/登录。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
