"""Install into Shinsekai's Python through its shared dependency installer."""
from __future__ import annotations

import sys
from pathlib import Path

# Also works when the embedded runtime does not add the script's directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plugin_system.requirements.install import install_plugin_requirements_txt, pip_python_executable


def main() -> int:
    print(f"安装到新世界 Python：{pip_python_executable()}", flush=True)
    print("使用新世界统一管理的 PyTorch 版本和 CUDA 通道，复用已安装且兼容的依赖。", flush=True)
    code, detail = install_plugin_requirements_txt(
        Path(__file__).parent, on_output_line=lambda line: print(line, flush=True),
    )
    if code != "pip_ok":
        print(f"依赖安装未完成：{code}\n{detail}", file=sys.stderr)
        return 1
    print("依赖已就绪。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
