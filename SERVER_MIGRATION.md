# 双 RTX 4090 D Linux 服务器迁移

## 1. 已确认资源

- 主机：`worker-0`，Ubuntu 22.04.3 LTS (Jammy Jellyfish)，Python 3.10.12。
- CPU/内存：32 logical CPUs、251 GiB RAM、无 swap。
- GPU：2 × NVIDIA GeForce RTX 4090 D，每张 47.37 GiB 可见显存。
- 驱动：595.58.03，`nvidia-smi` 报告 CUDA 13.2。
- 持久化目录：`/public/home/wwwyyycom123_`，NFS 约 13 TiB 可用。
- 容器根目录 `/` 虽有约 806 GiB 可用，但不视为持久化存储，不保存正式数据和环境。
- 系统环境的 PyTorch 2.1.0+cu121 低于本工程 `torch>=2.4` 要求，不直接修改；正式任务使用用户目录下独立 venv。

## 2. 服务器目录约定

```bash
export MMSV_HOME=/public/home/wwwyyycom123_/multimodal_sv_reproduction
export MMSV_CORPORA=/public/home/wwwyyycom123_/datasets/corpora
export MMSV_VENV=/public/home/wwwyyycom123_/venvs/mmsv
mkdir -p "$MMSV_CORPORA" "$(dirname "$MMSV_VENV")"
```

`MMSV_CORPORA` 下保持本机 `corpora` 的相对目录结构，包括：

- `fisher/fisher_eng_tr_sp_LDC2004S13/...`（约 27.98 GiB）；
- `fisher/fe_03_p1_tran_LDC2004T19/...`（约 0.25 GiB）；
- `LibriSpeech/train-clean-360/...`（约 22.25 GiB）。

Fisher 为授权语料，只能传到满足许可证要求的私有存储；不得放入 Git、公开对象存储或公开下载链接。

## 3. 基础工具与隔离环境

服务器当前未检测到 `curl`、`rsync`、`tmux`、`ffmpeg`、`conda`。先检查发行版和可用包管理器：

```bash
cat /etc/os-release
command -v apt-get dnf yum
sudo -n true && echo 'passwordless sudo available' || echo 'sudo may require a password'
```

Debian/Ubuntu 系列：

```bash
sudo apt-get update
sudo apt-get install -y curl rsync tmux ffmpeg libsndfile1 git
```

Rocky/RHEL 系列（`ffmpeg` 可能需要平台额外仓库，当前流水线不以系统 ffmpeg 为启动前置条件）：

```bash
sudo dnf install -y curl rsync tmux libsndfile git
```

创建持久化 venv，并安装与本机成功运行环境一致的核心版本：

```bash
sudo apt-get install -y python3-venv
cd "$MMSV_HOME"
MMSV_HOME="$MMSV_HOME" MMSV_VENV="$MMSV_VENV" \
  bash scripts/setup_linux_server_env.sh
```

脚本将以重试方式安装与本机一致的 PyTorch 2.9.1+cu128 及 StreamVoiceAnon 推理依赖，执行全部 pytest 和 StreamVoice wrapper import，并把 Python、CUDA、双 GPU、依赖版本、Git HEAD 与完成时间写入 `results/runs/server_setup/environment.json`。安装后可再次核验：

正式服务器环境已于 2026-09-01 11:16:24 +08:00 按该脚本验收通过：`25 passed`、`streamvoice_import=true`、`server_environment_ready=true`，两张 47.37 GiB RTX 4090 D 均由 PyTorch 2.9.1+cu128 正确识别。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); [print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
python -m pytest -q
```

## 4. Git 与传输范围

代码、配置、测试及 Markdown 总账由 Git 管理。服务器 clone 后，以下被 `.gitignore` 排除的文件仍需从本机传输：

- `artifacts/anonymization/`；
- `artifacts/metadata/`；
- `artifacts/trials/`；
- `artifacts/anonymized/train/`（切换时的全部断点产物）；
- `artifacts/anonymized/evaluation/`（86,222 条，约 5.20 GiB）；
- `results/runs/audio_corrected_p1/last.pt`；
- `third_party/StreamVoiceAnon/`；
- Hugging Face `models--microsoft--wavlm-large` snapshot。

当前阶段无需复制 `artifacts/cache/fisher_train_all_p1`（约 23.51 GiB）。匿名化从原始 Fisher SPHERE 读取；semi-informed 训练从生成后的匿名 FLAC 读取。

优先通过平台提供的持久化数据集/NFS 导入功能传输。若平台提供 worker SSH 地址和端口，可从支持 rsync 的本机/WSL 执行增量传输；不要用浏览器逐文件上传十万级小文件。

`worker-0` 容器内 SSHD 已确认监听 `0.0.0.0:22` 且启用公钥认证，但容器地址属于平台内部网络；从实验本机直接连接该地址的 TCP/22 测试失败。因此实际传输必须使用平台控制台给出的 SSH gateway/forward host 与映射端口，不能把 `hostname -I` 的容器地址当成外部 SSH 地址。若控制台不提供 SSH 转发，则改用平台的持久化数据集/NFS 导入功能。

安全要求：只使用 SSH 公钥；不要在聊天、日志或实验总账中保存密码、私钥、验证码或控制台 token。若终端诊断意外显示明文凭证，应先在平台侧轮换使其失效，再继续配置传输。

服务器首次 `git clone --recurse-submodules` 曾因外网链路中断报 `GnuTLS recv error (-110)`。这不是仓库或提交错误；主仓库可改用 HTTP/1.1、单请求、浅 fetch 的可重试初始化，避免每次失败都重新创建目录：

```bash
export MMSV_HOME=/public/home/wwwyyycom123_/multimodal_sv_reproduction
mkdir -p "$MMSV_HOME"
git -C "$MMSV_HOME" init
git -C "$MMSV_HOME" remote remove origin 2>/dev/null || true
git -C "$MMSV_HOME" remote add origin https://github.com/wwwYYYcom/multimodal-SV.git

fetched=0
for attempt in 1 2 3 4 5; do
  if git -C "$MMSV_HOME" -c http.version=HTTP/1.1 -c http.maxRequests=1 \
      fetch --depth=1 origin main; then
    fetched=1
    break
  fi
  echo "fetch attempt $attempt failed; retrying"
  sleep 10
done
test "$fetched" -eq 1
git -C "$MMSV_HOME" checkout -B main FETCH_HEAD

GIT_HTTP_MAX_REQUESTS=1 git -C "$MMSV_HOME" -c http.version=HTTP/1.1 \
  submodule update --init --depth=1 --jobs=1
git -C "$MMSV_HOME" rev-parse HEAD
```

预期主仓库 HEAD 至少为记录服务器迁移代码的 `47bc1b3c12f3334de7b411beae131f53bebf5bb4`，实际应取远程 `main` 的更新提交。若 5 次均失败，停止重试并改用平台数据导入或从本机生成 `git bundle`，不要关闭 TLS 校验。

## 5. Windows 路径转换

所有传输完成后，在服务器项目根目录运行一次：

```bash
cd "$MMSV_HOME"
source "$MMSV_VENV/bin/activate"
PROJECT_ROOT="$MMSV_HOME" CORPORA_ROOT="$MMSV_CORPORA" \
  bash scripts/remap_artifacts_for_linux.sh
```

脚本流式、原子地重写 4 个 CSV 中的 Windows 绝对路径，并为每个 CSV 写出 `*.remap.audit.json`。任何未被映射的绝对路径都会使脚本失败，避免任务运行数小时后才发现路径错误。

核验路径：

```bash
head -n 2 artifacts/anonymization/train_all_utterances_plan.csv
find artifacts/anonymized/train -type f -name '*.flac' | wc -l
find artifacts/anonymized/evaluation -type f -name '*.flac' | wc -l
```

## 6. 双 GPU 断点恢复

先做不启动模型的 slice/disk dry run：

```bash
cd "$MMSV_HOME"
source "$MMSV_VENV/bin/activate"
DRY_RUN=1 GPU_IDS=0,1 WORKERS_PER_GPU=4 \
  bash scripts/anonymize_train_multigpu_then_train_semi.sh
```

正式默认配置为 2 GPUs × 4 workers，共 8 个互斥 slice。已有非空 FLAC 会被跳过但仍写入 worker manifest，因此从本机断点产物继续且最终 manifest 保持 572,951 行原始顺序。每个 worker 最多自动重启 20 次；超过 30 秒的 13 条 source 继续使用已验证的分块拼接路径。

在 `tmux` 中启动：

```bash
tmux new -s mmsv
cd "$MMSV_HOME"
source "$MMSV_VENV/bin/activate"
GPU_IDS=0,1 WORKERS_PER_GPU=4 MONITOR_SECONDS=60 \
  bash scripts/anonymize_train_multigpu_then_train_semi.sh \
  2>&1 | tee results/runs/anonymization_train/server_supervisor.log
```

按 `Ctrl+B`、再按 `D` 退出 tmux 而不停止任务；恢复查看：

```bash
tmux attach -t mmsv
```

流水线会依次执行：全 572,951 条 train 匿名化、manifest 合并与完整性校验、从 corrected lazy checkpoint 初始化的 15 epoch semi-informed 训练、original/anonymized evaluation embedding 提取、Mean O-A/A-A N=1/5/10/15 评分。

初次正式启动后先观察 15 分钟。若两张 GPU 稳定、无 OOM 且利用率仍低，再测试 `WORKERS_PER_GPU=6`；不得在同一输出目录同时运行两套 supervisor。

## 7. 切换原则

1. 服务器代码、环境、全部静态数据和模型先准备完成，本机任务保持运行。
2. 服务器完成普通样本和 43.24 秒故障样本 smoke test。
3. 记录本机最后进度后停止本机 supervisor 及两个 worker。
4. 最后一次增量同步 `artifacts/anonymized/train/`。
5. 核对服务器 FLAC 数量、总字节数和抽样 SHA-256。
6. 只在上述核验通过后启动服务器正式 supervisor。
7. 将切换时间、服务器路径、硬件、代码提交、文件数和日志路径追加到 `EXPERIMENT_RESULTS.md`。

## 8. E-File 分包传输

平台未提供外部 SSH 映射端口，但 E-File 的“本地上传”已通过 TAR 探针验证。正式数据不要逐文件上传；在 Windows 本机按稳定相对路径打成约 1.9 GB 输入数据一片的 TAR，每次只生成、上传并验收一片：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File scripts\create_server_transfer_pack.ps1 `
  -Name fisher_audio `
  -BaseDirectory 'D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora' `
  -InputPath 'D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fisher_eng_tr_sp_LDC2004S13' `
  -DestinationRoot '/public/home/wwwyyycom123_/datasets/corpora' `
  -PartNumber 1
```

每片会在 `C:\mmsv_transfer` 输出 `.tar`、`.files.txt`、`.audit.json` 和 `.sha256`。通过 E-File 把 TAR 上传到 `/public/home/wwwyyycom123_/incoming/`，然后在服务器核验并解包：

```bash
export MMSV_HOME=/public/home/wwwyyycom123_/multimodal_sv_reproduction
export MMSV_CORPORA=/public/home/wwwyyycom123_/datasets/corpora
mkdir -p /public/home/wwwyyycom123_/incoming "$MMSV_CORPORA"

cd /public/home/wwwyyycom123_/incoming
sha256sum <archive>.tar
tar -tf <archive>.tar | sed -n '1,3p;$p'
tar -xf <archive>.tar -C "$MMSV_CORPORA"
```

项目内产物使用项目根作为 `DestinationRoot` 和解包目录；corpora 使用 `MMSV_CORPORA`；WavLM snapshot 使用 `/public/home/wwwyyycom123_/.cache/huggingface/hub`。只有 SHA-256 与本机 `.sha256` 完全一致、成员路径正确且解包成功后，才把该片标记完成并生成下一片。本机正式任务在所有静态资源和服务器 smoke 验收前保持运行。

WavLM 缓存只迁移 `models--microsoft--wavlm-large/refs/` 和已完成的 `snapshots/<commit>/`。不得打包 `*.incomplete`、`.no_exist/` 或 lock 文件；这些是下载残留/缓存状态，不是模型运行输入。当前有效 snapshot commit 为 `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c`。
