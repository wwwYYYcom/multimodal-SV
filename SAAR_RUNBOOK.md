# SAAR Phase 1/2 混合运行手册

本文只描述运行方法；实验事实、时间、指标和文件指纹统一记录在
[`EXPERIMENT_RESULTS.md`](EXPERIMENT_RESULTS.md)。当前实现遵守 Fisher Part 1 +
LibriSpeech `train-clean-360` 的数据边界。

## 任务划分

本机已经完成以下低算力工作：

- 构造 5 个固定 seed 的 session-aware nested trials；
- 构造 `call_id + channel -> LibriSpeech reference` 的固定伪说话人映射；
- 使用已有 86,222 条匿名 embedding 计算 utterance-random control；
- 计算 control PCS、隐私退化率、斜率及绘图；
- 测试、输入审计和服务器上传包。

服务器负责以下 GPU 工作：

- 对新协议实际引用的 66,712 条 target utterance 执行 session-fixed 匿名化；
- 提取 66,712 条匿名语音的 WavLM-ECAPA embedding；
- 计算 5 seeds × 5 个 N 的 O-A EER、PCS 和 Gate 1。

原论文复现流水线与 SAAR 使用不同的输出目录。SAAR runner 默认检测并拒绝与任何
仍在运行的原论文复现 supervisor、`anonymize-streamvoice`、`train-audio` 或
`extract-embeddings` 任务争抢 GPU。

## 输入包

本机输入包：

```text
D:\download4browser\mmsv_saar_phase12_inputs_20260904_165532.tar
bytes=133223936
sha256=e7b419ea9256830e22d083da8deb30d5b875d815e483ff37781d0edd7c92d2bd
```

通过 E-File 上传到 `/public/home/wwwyyycom123_/`，然后执行：

```bash
set -euo pipefail
export MMSV_HOME=/public/home/wwwyyycom123_/multimodal_sv_reproduction
cd /public/home/wwwyyycom123_

echo 'e7b419ea9256830e22d083da8deb30d5b875d815e483ff37781d0edd7c92d2bd  mmsv_saar_phase12_inputs_20260904_165532.tar' | sha256sum -c
tar -xf mmsv_saar_phase12_inputs_20260904_165532.tar -C "$MMSV_HOME"
```

## 路径重映射与 dry run

先更新到包含 SAAR 实现的 Git 提交，再执行：

```bash
set -euo pipefail
export MMSV_HOME=/public/home/wwwyyycom123_/multimodal_sv_reproduction
export MMSV_VENV=/public/home/wwwyyycom123_/venvs/mmsv
export MMSV_CORPORA=/public/home/wwwyyycom123_/datasets/corpora
cd "$MMSV_HOME"
source "$MMSV_VENV/bin/activate"

PROJECT_ROOT="$MMSV_HOME" \
SERVER_PROJECT_ROOT="$MMSV_HOME" \
SERVER_CORPORA_ROOT="$MMSV_CORPORA" \
PYTHON_EXE="$MMSV_VENV/bin/python" \
bash scripts/remap_saar_artifacts_for_linux.sh

DRY_RUN=1 GPU_IDS=0,1,2,3 WORKERS_PER_GPU=4 \
PYTHON_EXE="$MMSV_VENV/bin/python" \
bash scripts/anonymize_saar_session_baseline_multigpu.sh
```

预期看到 `saar_linux_path_remap_ready=true`、`rows=66712`、16 个连续互斥
slice，以及 `dry_run_complete=true`。

## 正式启动

只有原论文复现的匿名化进程已经结束时才启动：

```bash
pgrep -af '[m]msv.cli anonymize-streamvoice' || true

tmux new -s saar-baseline
cd "$MMSV_HOME"
source "$MMSV_VENV/bin/activate"
mkdir -p results/runs/saar_session_baseline

GPU_IDS=0,1,2,3 WORKERS_PER_GPU=4 MONITOR_SECONDS=60 \
PYTHON_EXE="$MMSV_VENV/bin/python" \
bash scripts/anonymize_saar_session_baseline_multigpu.sh \
2>&1 | tee results/runs/saar_session_baseline/server_supervisor.log
```

按 `Ctrl+B`、再按 `D` 脱离 tmux。runner 会自动完成生成、合并、格式校验、
embedding、25 项 EER、PCS 和 Gate 1；若服务器已有 matplotlib 也会绘制曲线，否则会
在 summary 中留下本机补图命令。它不会自动启动 SAAR 训练。

## 进度、暂停与恢复

查看整体进度：

```bash
watch -n 60 'COUNT=$(find artifacts/saar/anonymized/session_baseline_evaluation -type f -name "*.flac" | wc -l); awk -v c="$COUNT" "BEGIN {printf \"completed: %d / 66712\\nprogress: %.6f%%\\nremaining: %d\\n\", c, c*100/66712, 66712-c}"; echo; echo "active workers:"; pgrep -fc "[m]msv.cli anonymize-streamvoice"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader'
```

需要暂停时，在 tmux 前台按 `Ctrl+C`。已经原子写完的非空 FLAC 会保留；用完全相同的
正式启动命令重启后会逐条跳过并重建 worker manifest。不要同时启动两套 supervisor。

## Gate 1 输出

主要输出：

```text
artifacts/saar/session_baseline/metrics/gate_1.json
artifacts/saar/session_baseline/metrics/privacy_summary.csv
artifacts/saar/session_baseline/metrics/pcs_summary.csv
artifacts/saar/session_baseline/figures/eer_vs_n.png
artifacts/saar/session_baseline/evaluation_summary.json
results/runs/saar_session_baseline/final.validation.json
```

本工程把 5-seed mean `EER(N=1)-EER(N=15) >= 1.0` 个百分点定义为 Gate 1
“有意义下降”的可执行阈值。通过后才进入 SAAR MVP 训练；失败时先复核协议或修订假设。
