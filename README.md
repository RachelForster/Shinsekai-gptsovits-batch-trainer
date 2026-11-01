# GPT-SoVITS 批量训练插件

工具页现在包含两个独立步骤：**全集人声按说话人分目录**，以及原有的**按人物目录批量训练**。
已经整理好的人物目录仍可直接训练。

## 全集人声按说话人分目录

输入已去除配乐的人声音频总目录，递归处理所有音频。建议同一部番剧的全集放在一起，
**每集保持独立文件**：脚本逐集推理，再根据声纹跨集聚类，不需要拼接全集。
同一编号尽量对应同一说话人，但编号不代表真实角色名。

实现使用本机运行的 [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)。
根据官方 [return_embeddings 接口](https://github.com/pyannote/pyannote-audio/blob/3.4.0/pyannote/audio/pipelines/speaker_diarization.py)
提取每集说话人的声纹，再以余弦相似度做 complete-link 聚类；默认同一集已分开的说话人不自动互相合并。
「最多说话人数（全集）」可限制最终 `speakers` 内的目录数量，`0` 表示不限。
例如填写 `10`：先按阈值聚类，若仍超过 10 组，再按声纹相似度继续合并到 10 组；
这个阶段允许合并同一集内被拆开的编号，也不再受阈值限制。原本不足 10 组时不会补足。
人数设得太少可能混入不同角色；声纹缺失或无效的片段不强行归给某个角色，单独存入同级 `_needs_review` 目录。
这是一套保守的自动整理流程，动漫喊叫、模仿、变声、同一声优配多个角色都可能影响准确度。
阈值 `0.72` 是初始经验值，未经你的番剧数据校准；不同人物误合并时提高阈值，
同一人物跨集被拆开时可适当降低，或试听后手动合并。

### 首次安装（使用新世界现有 Python）

插件自动跟随新世界正在使用的 Python，无需创建虚拟环境或填写另一个 Python 路径。
旧配置中的独立环境路径会被忽略。依赖通过新世界现有的插件安装器管理：
已安装且版本、CPU/CUDA 通道匹配的 PyTorch 会直接复用，缺失时才安装。
新世界当前统一版本为 PyTorch 2.7.1，因此使用兼容的 `pyannote.audio==3.4.0` 与说话人模型 3.1；
不再要求需要更新 PyTorch 的 pyannote 4.x / community-1。

在插件管理器安装本插件依赖，或者用新世界的 Python 执行安装脚本。
以下以仓库自带的 `runtime\python.exe` 为例；如果新世界选择了其他运行环境，使用工具页显示的 Python 路径：

```powershell
Set-Location C:\AI\Shinsekai
& .\runtime\python.exe -X utf8 .\plugins\gpt_sovits_batch_trainer\install_speakers.py
```

安装脚本调用新世界的安装器，为依赖解析加上宿主 PyTorch 版本约束；
不直接执行无约束的 `pip install pyannote.audio`，以免间接升级 PyTorch。
模型推理在使用同一个 Python 的子进程里进行，仍可停止，不会创建另一套环境。
`auto` 会在 CUDA 不可用时改用 CPU；手动选择 `cuda` 则会明确报错。
FFmpeg 会优先使用系统安装，否则使用 `imageio-ffmpeg` 自带版本。

首次下载模型前，在上述模型 3.1 页面和 [segmentation-3.0 页面](https://huggingface.co/pyannote/segmentation-3.0)
登录并接受访问条件。脚本会自动读取新世界 `data/config/api.yaml` 的
`hugging_face_access_token`，无需重新填写令牌；插件使用宿主实际的数据目录。
也可以用同一个 Python 登录：

```powershell
& .\runtime\python.exe -X utf8 -c "from huggingface_hub import login; login()"
```

令牌优先级：`HF_TOKEN` 环境变量 → 新世界 API 配置 → Hugging Face 本机保存的登录。
插件仅通过子进程环境传递配置令牌，不写入命令行、插件配置或结果文件，日志会脱敏。
独立脚本的 `--api-config` 可指定其他数据目录下的 API 配置文件。
模型下载完成后在本机推理，不上传音频；脚本关闭 pyannote 使用统计。
模型来源还可填写 3.1 的本地 `config.yaml` 或包含它的目录；离线使用时配置中的分割模型、
声纹模型也必须改为已下载的本地路径。不能直接使用 community-1 的配置。

配置里没有 Token 并不一定不能用：本机保存的 HF 登录可继续提供下载凭据；如果完全没有凭据，
首次下载受访问限制的模型会失败。若所有模型文件已经下载、配置中的引用全部指向本地，
则本地推理无需 Token。仅缓存了部分文件不满足此条件；有 Token 也仍需账号接受上述两个模型的访问条件。

### 使用方式

重载插件后，在工具页第 1 部分填写「全集人声目录」，按需要设置「最多说话人数（全集）」；展开设置可查看自动选用的新世界 Python 路径。
点击「开始按说话人分目录」。完成后先试听整理，再点击「将本次结果填入训练目录」，
按需要修改人物目录名，最后手动开始批量训练。分目录功能不依赖 GPT-SoVITS 安装路径配置。
两个任务共用日志和停止按钮，同一时刻只允许运行一个任务。

独立脚本也可运行：

```powershell
& .\runtime\python.exe -X utf8 .\plugins\gpt_sovits_batch_trainer\batch_diarize.py D:\anime\第一季人声

# 自定义输出和匹配阈值
& .\runtime\python.exe -X utf8 .\plugins\gpt_sovits_batch_trainer\batch_diarize.py D:\anime\第一季人声 `
  --output D:\anime\整理结果 --max-speakers 10 --threshold 0.78 --min-seconds 1.5 --max-seconds 12

# 只检查目录；不加载模型、不安装依赖、不创建结果目录
& .\runtime\python.exe -X utf8 .\plugins\gpt_sovits_batch_trainer\batch_diarize.py D:\anime\第一季人声 --dry-run
```

输出默认为输入目录旁的 `<目录名>_speakers`，结构如下：

```text
第一季人声_speakers/
  .speaker_cache/                     # 逐集时间段、声纹缓存
  run_日期_时间_唯一后缀/
    speakers/                         # 把这一层作为批量训练的语音总目录
      speaker_001/*.wav
      speaker_002/*.wav
    _needs_review/*.wav               # 限制人数时无有效声纹的片段，仅在需要时创建，不参与训练
    segments.csv                      # 每段的原文件、起止秒数、归属和输出路径
    speaker_mapping.json              # 本集说话人 → 跨集编号
    report.json                       # 状态、参数、每个说话人的片段数和累计时长
```

- 默认保留 1.5–12 秒切片，两端各裁剪 0.05 秒；输出为 32 kHz 单声道 PCM16 WAV。
  长句按时长均分，不使用转写寻找词边界；训练前请试听检查切点。
- 根据**普通说话人时间线**排除多人重叠区间，不用 exclusive 时间线强制把重叠归给某个人。
  本功能不会从同一时刻的混合声音中还原两条独立人声，也不会再次去配乐或自动排除歌唱。
- 原音频保持不变。相同文件名位于不同子目录时，导出名称仍然唯一。
- 每次启动生成新的 `run_*`，避免改阈值后混入旧切片；不会覆盖人工改名的历史结果。
  新增集数或改阈值后编号可能变化，请以本次 `speaker_mapping.json` 为准。
- 文件路径、大小、修改时间、模型来源和 pyannote 版本匹配时，复用已完成的逐集分析。
  修改最多说话人数、分组阈值或切片时长无需重新推理，但需要重新解码导出。更换本地权重时使用 `--reanalyze`。
- 停止后未完成的一集需重算，已完成的分析缓存可复用。强制停止可能留下不完整 `run_*` 或临时 WAV；
  不要训练其 `speakers` 目录，只使用页面提示的已完成结果（`report.json` 状态为 `completed`）。
- 内存和临时 WAV 按单集使用，逐集串行；预先拼好的数小时单文件仍会增加单次内存占用。
  输出只保留单人切片和分析缓存，不长期缓存整集 PCM。

## 按人物目录批量训练

输入一个总目录，插件会把其中每个含音频的一级子目录当作一个人物，并依次执行：

1. 语音切分到 `output/slicer_opt/<人物名>`；
2. 使用 `Faster Whisper (多语种)`、`large-v3`、自动识别语言生成标注；
   优先 `tools/asr/models/faster-whisper-large-v3`，否则复用本机 Hugging Face 缓存，不再联网下载。
3. 以人物目录名作为实验名，按 `v2ProPlus` 执行训练集格式化一键三连；
4. 训练 SoVITS 8 轮；
5. 训练 GPT 15 轮，然后继续下一个人物。

训练完成后，第二步旁的「填入人物配置」会读取：

- `GPT_weights_v2ProPlus/<人物名>-e15.ckpt` 等 GPT 权重（取最高 epoch）；
- `SoVITS_weights_v2ProPlus/<人物名>_e8_s134.pth` 等 SoVITS 权重（取最高 epoch / step）；
- `output/slicer_opt/<人物名>/`（也兼容 `slice_opt`）中一段 3–10 秒参考音频；
- `output/asr_opt/<人物名>.list` 里该音频对应的转写和语言。

写入 Shinsekai `data/config/characters.yaml`：已有同名人物只更新语音字段，没有则按目录名新建。

插件读取 Shinsekai `API 配置 → TTS 语音` 中已经保存的 GPT-SoVITS 服务启动路径，
不再内置或猜测安装目录。尚未配置、目录不存在或缺少训练所需文件时，工具页会提示先完成配置。
Faster Whisper 使用 GPT-SoVITS 本地模型或本机已缓存的 `large-v3`，不会再访问 Hugging Face。
插件直接调用 `webui.py` 背后的脚本入口，因此无需先运行 `go-webui.bat`，也不要同时在 WebUI 中启动另一项训练。
Windows 上某些 Faster Whisper/FunASR 原生库会在标注写完后的 DLL 清理阶段返回
`0xC0000409`；插件会逐一核对切片和标注，只在全部匹配时将其视为成功。

中文人物名无需改名：所有阶段使用自带 Python 的显式 `-X utf8` 参数，确保训练时
`spawn` 出的子进程也启用 UTF-8。训练 JSON/YAML 额外使用 Unicode 转义，兼容上游未指定
编码的配置读取逻辑；解析后的实验名与文件路径仍是原来的中文。

## 独立脚本

不打开 Shinsekai 也可以运行：

```powershell
& "<已配置的 GPT-SoVITS 目录>\runtime\python.exe" `
  "C:\AI\Shinsekai\plugins\gpt_sovits_batch_trainer\batch_train.py" D:\voices
```

先查看将执行的流程：

```powershell
& "<已配置的 GPT-SoVITS 目录>\runtime\python.exe" `
  "C:\AI\Shinsekai\plugins\gpt_sovits_batch_trainer\batch_train.py" D:\voices --dry-run
```

每个成功阶段会记录到 GPT-SoVITS 的 `output/batch_train_state`。任务中断后重新开始会续跑。
`--force` 会忽略这些阶段记录并重新进入完整流程；GPT-SoVITS 对已经存在的同名训练 checkpoint
仍会沿用其自身的续训机制。
