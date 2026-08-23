# Multimodal Speaker Verification 复现工程

本仓库复现 Garg et al. (2026) 的多语句 speaker verification 攻击协议。当前优先完成论文的 Level 1 音频闭环：

`Fisher manifest -> speaker split -> nested trials -> WavLM-Large + ECAPA-TDNN -> mean/query/frame aggregation -> cosine -> EER`

论文链接：<https://arxiv.org/abs/2607.19636>。作者仓库：<https://github.com/Ashigarg123/multimodal-speaker-verification>。

## 当前复现边界

截至 2026-08-22，作者仓库仍只包含“Code and pretrained models will be released soon”。论文还没有公开精确 Fisher speaker split、evaluation trial list、随机种子和部分 ECAPA/query-attention 工程细节。本工程会把这些选择写入 audit JSON，绝不把自选设置称为作者原始设置。

用户于 2026-08-23 明确将本项目的数据范围固定为 Fisher Part 1（5,850 calls / 7,066 speakers）和 LibriSpeech `train-clean-360`。后续不等待、不引入 Fisher Part 2 或 `train-other-500`。因此：

- `configs/local_fisher_p1.yaml` 是全部正式实验的默认数据配置，按论文比例缩放 speaker 数。
- `configs/paper.yaml` 只保留为论文规模参考，不进入当前实验运行链。
- 所有结果都标为“Part 1 + clean-360 范围复现”；其 EER 可与论文做参考比较，但不能冒充论文完整数据规模的原始数值。
- StreamVoiceAnon checkpoint 已下载到 `third_party/StreamVoiceAnon/pretrained_checkpoints/dual_ar_delay_0_8.pth`，后续可在上述固定数据范围内生成匿名化语音。

## 环境

推荐现有 Conda 环境：

```powershell
conda activate pytorch
python -m pip install -e ".[audio,test]"
```

配置默认将 Hugging Face 端点设为 `https://huggingface.co`。本机原有的 `hf-mirror.com` 端点不可达；若你的网络需要其他镜像，请修改配置中的 `model.hf_endpoint`。

Fisher SPHERE 使用 shorten 压缩。Windows 默认使用 `desphere[fast]` 解码（随 `audio` extra 安装）；如另有 `sph2pipe`，可用它按时间范围直接解码，长训练更高效：

```powershell
$env:SPH2PIPE = "D:\path\to\sph2pipe.exe"  # 可选
```

## 可重复运行顺序

```powershell
mmsv prepare-fisher --config configs/local_fisher_p1.yaml --output artifacts/metadata/fisher_manifest.csv
mmsv split-speakers --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --output artifacts/metadata/speaker_splits.csv
mmsv build-trials --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --split evaluation --output artifacts/trials/evaluation.jsonl
mmsv model-smoke
```

短训练验证（下载 WavLM-Large 后，只执行 1 个 optimizer step）：

```powershell
mmsv train-audio --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/audio_lazy --max-steps 1
```

正式训练去掉 `--max-steps`。中断用 `Ctrl+C`；从最后 checkpoint 续跑：

```powershell
mmsv train-audio --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/audio_lazy --resume results/runs/audio_lazy/last.pt
```

Semi-informed 阶段应重置 optimizer，并只用 lazy 权重初始化（不是普通 resume）：

```powershell
mmsv train-audio --config configs/semi_local.yaml --manifest artifacts/metadata/fisher_anonymized_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/audio_semi --init-from results/runs/audio_lazy/last.pt
```

固定范围的匿名化计划与可恢复执行：

```powershell
# O-A/A-A evaluation：只规划 evaluation trials 引用的 utterance
mmsv plan-anonymization --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --reference-pool artifacts/metadata/librispeech_target_pool.csv --trials artifacts/trials/evaluation.jsonl --audio-output-root artifacts/anonymized/evaluation --output artifacts/anonymization/evaluation_plan.csv

# semi-informed：每个 train call-side 固定选一条 source，覆盖全部训练 speaker
mmsv plan-anonymization --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --reference-pool artifacts/metadata/librispeech_target_pool.csv --splits artifacts/metadata/speaker_splits.csv --split-name train --one-per-call-side --audio-output-root artifacts/anonymized/train --output artifacts/anonymization/train_one_per_call_side_plan.csv

mmsv anonymize-streamvoice --plan artifacts/anonymization/evaluation_plan.csv --output-manifest artifacts/metadata/fisher_anonymized_evaluation_manifest.csv --streamvoice-root third_party/StreamVoiceAnon --delay 2 --alpha 1.0
```

runner 会跳过已经存在的非空 FLAC、追加 progress JSONL，并将 StreamVoiceAnon 的 44.1 kHz 输出立即转为 16 kHz FLAC。批量匿名化与 O-O 训练不要同时占用 GPU。

训练每 10 个 optimizer step 以及每个 epoch 结束时原子写入 `last.pt`，checkpoint 记录 epoch 内 batch 位置；中断后不会重跑整个 epoch。每个 epoch 尾部不足 gradient accumulation 的 batch 会在校正梯度缩放后正常更新。嵌入与 EER：

checkpoint 只保存可训练的 ECAPA/classifier/optimizer；冻结的 WavLM-Large 从 Hugging Face snapshot 复用，避免每个 checkpoint 重复约 1.2 GB 权重。

Windows 的 `desphere` 需要先解整通 Shorten call。训练 loader 因而按 call-side 组织，每个 epoch 每侧确定性抽一条 turn，并让同一 call 的 A/B 相邻以复用缓存；这是论文未公开采样细节下的本工程选择，已在状态文档中标明。若安装 `sph2pipe`，可按片段直接解码。

```powershell
mmsv extract-embeddings --checkpoint results/runs/audio_lazy/last.pt --manifest artifacts/metadata/fisher_manifest.csv --trials artifacts/trials/evaluation.jsonl --output artifacts/embeddings/original.npz
mmsv score-mean --trials artifacts/trials/evaluation.jsonl --original-embeddings artifacts/embeddings/original.npz --condition O-O --n 5 --output results/o_o_mean_n5.csv
```

`--trials` 会只提取固定 trial 实际引用的 utterance，避免对 929,364 条 manifest 全量重复推理。

## 论文协议已编码的约束

- speaker-disjoint train/validation/evaluation；validation 和 evaluation speaker 均须能构造 call-disjoint `N=15` 双侧 trial。
- enrollment/target 按 call 分池，没有 call 泄漏。
- `N={5,10,15}` 使用一次 max-N 采样再切前缀，严格满足 `U5 subset U10 subset U15`。
- O-O / O-A / A-A 可复用同一 trial identity composition。
- WavLM-Large 冻结；ECAPA 输出 192 维；4 秒 crop；AAM margin=0.2、scale=30。
- Query Attention 为 4 heads、temperature=0.3；未擅自加入论文未说明的 FFN/dropout。
- ECAPA 暴露 ASP 前 frame map，可实现真正的 frame concatenation。

全部实验命令、完成时间、实测数据、产物绝对路径与 SHA-256 统一记录在 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。简要状态和缺失资源见 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

## Git 版本管理

主分支为 `main`，复现实验基线标签为 `v0.1.0-reproduction-baseline`。StreamVoiceAnon 使用 submodule 固定版本；clone 时应同时初始化 submodule：

```powershell
git clone --recurse-submodules git@github.com:wwwYYYcom/multimodal-SV.git
```

数据、checkpoint 和逐条实验产物不会提交到普通 Git 历史；每次实验应把配置、代码以及结果摘要一并提交，并将全部结果追加到 `EXPERIMENT_RESULTS.md`。详细分支、提交和远程仓库命令见总账第 13 节。
