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

Windows 上正式训练前，先把 30 个 epoch 确定会抽到的 Fisher utterance 缓存成短 FLAC，避免每个 batch 重复解码整通 SPHERE：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -u scripts/build_fisher_training_cache.py --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir artifacts/cache/fisher_train_selected_30e
```

脚本支持重复执行并跳过非空缓存文件；只有生成 `audit.json` 后才视为缓存完整。`configs/local_fisher_p1.yaml` 中的 `segment_cache_dir` 会强制训练使用完整缓存，缺少任一片段时立即报错。

论文口径修正版不再采用“每 call-side 一条”的低覆盖采样。它缓存 Part 1 train split 的全部 572,951 条 utterance，并以物理 batch 64 训练 ECAPA；冻结的 WavLM 按 32 条分块前向以适配 8 GiB GPU：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -u scripts/build_fisher_full_training_cache.py --config configs/local_fisher_p1_corrected.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir artifacts/cache/fisher_train_all_p1 --reuse-cache-dir artifacts/cache/fisher_train_selected_30e

mmsv train-audio --config configs/local_fisher_p1_corrected.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/audio_corrected_p1
```

`scripts/build_full_cache_then_train_corrected.ps1` 会串联全量缓存、缓存审计、corrected 训练和独立的 O-O 后处理。旧版 `audio_lazy_p1_v2` 及 `results/o_o` 作为失效基线保留，不得用于初始化 corrected run。

全量缓存构建期间，`fisher_train_selected_30e` 作为硬链接复用源暂时保留。`scripts/consolidate_cache_after_build.ps1` 会在完整审计通过后逐条验证 180,311 个硬链接，把旧审计复制到全量目录并移除旧目录；最终训练缓存统一位于 `artifacts/cache/fisher_train_all_p1`。

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

# RTX 5060 8 GiB 上按 source 总时长平衡的双进程 evaluation 续跑
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/anonymize_evaluation_dual.ps1
```

runner 会跳过已经存在的非空 FLAC、追加 progress JSONL，并将 StreamVoiceAnon 的 44.1 kHz 输出立即转为 16 kHz FLAC。批量匿名化与 O-O 训练不要同时占用 GPU。

当前正式训练由 `scripts/run_o_o_after_training.ps1` 监控。它只接受完整的 epoch 29 checkpoint，随后自动执行 trial-filtered embedding 提取以及 Mean N=1/5/10/15 O-O 评分；训练提前退出时会拒绝评分。N=1 是本地新增的单语句对照，论文表格仍只报告 N=5/10/15。

训练每 10 个 optimizer step 以及每个 epoch 结束时原子写入 `last.pt`，checkpoint 记录 epoch 内 batch 位置；中断后不会重跑整个 epoch。每个 epoch 尾部不足 gradient accumulation 的 batch 会在校正梯度缩放后正常更新。嵌入与 EER：

checkpoint 只保存可训练的 ECAPA/classifier/optimizer；冻结的 WavLM-Large 从 Hugging Face snapshot 复用，避免每个 checkpoint 重复约 1.2 GB 权重。

Windows 的 `desphere` 需要先解整通 Shorten call。训练 loader 因而按 call-side 组织，每个 epoch 每侧确定性抽一条 turn，并让同一 call 的 A/B 相邻；正式运行进一步读取上述短 FLAC 缓存。这是论文未公开采样细节下的本工程选择，已在状态文档中标明。若安装 `sph2pipe`，也可按片段直接解码。

```powershell
mmsv extract-embeddings --checkpoint results/runs/audio_lazy/last.pt --manifest artifacts/metadata/fisher_manifest.csv --trials artifacts/trials/evaluation.jsonl --output artifacts/embeddings/original.npz
mmsv score-mean --trials artifacts/trials/evaluation.jsonl --original-embeddings artifacts/embeddings/original.npz --condition O-O --n 5 --output results/o_o_mean_n5.csv
```

`--trials` 会只提取固定 trial 实际引用的 utterance，避免对 929,364 条 manifest 全量重复推理。

## 论文协议已编码的约束

- speaker-disjoint train/validation/evaluation；validation 和 evaluation speaker 均须能构造 call-disjoint `N=15` 双侧 trial。
- enrollment/target 按 call 分池，没有 call 泄漏。
- `N={1,5,10,15}` 使用一次 max-N 采样再切前缀，严格满足 `U1 subset U5 subset U10 subset U15`；N=1 是本地扩展，不冒充论文报告项。
- O-O / O-A / A-A 可复用同一 trial identity composition。
- WavLM-Large 冻结；ECAPA 输出 192 维；4 秒 crop；AAM margin=0.2、scale=30。
- Query Attention 为 4 heads、temperature=0.3；未擅自加入论文未说明的 FFN/dropout。
- ECAPA 暴露 ASP 前 frame map，可实现真正的 frame concatenation。

全部实验命令、完成时间、实测数据、产物绝对路径与 SHA-256 统一记录在 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)。简要状态和缺失资源见 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

截至 2026-08-27，corrected Mean O-O 已完成。Fisher Part 1 范围的 N=1/5/10/15 EER 分别为 `15.5044/4.1719/3.1133/2.9265%`；论文报告的 Mean N=5/10/15 为 `3.87/3.29/3.09%`。N=1 是本地扩展，完整产物和差异分析见实验总账第 22 节。

evaluation StreamVoiceAnon 的 86,222 条匿名化已完成并通过最终校验，实际双进程 RTF 为 0.6956、输出约 5.585 GB。匿名 embeddings 与 Mean O-A/A-A N=1/5/10/15 已于 2026-08-30 11:54:11 +08:00 全部完成：O-A EER 为 `43.5243/38.4184/36.7995/36.6750%`，lazy-informed A-A EER 为 `47.3848/39.3524/31.5068/25.7783%`。输出位于 `results/o_a_corrected` 与 `results/a_a_corrected`，完整审计记录见 `EXPERIMENT_RESULTS.md` 第 26 节。semi-informed 阶段采用 Fisher Part 1 train split 的全部 572,951 utterances，而不是 7,272 条 one-per-call-side 近似；全量计划和中止记录见总账第 27 节。

双 RTX 4090 D Linux 服务器迁移使用 `scripts/remap_artifacts_for_linux.sh` 和 `scripts/anonymize_train_multigpu_then_train_semi.sh`；持久化目录、环境安装、数据传输、路径重写、8-worker dry run、正式启动和切换检查表见 [SERVER_MIGRATION.md](SERVER_MIGRATION.md)。本机任务在服务器 smoke 和最终增量同步完成前保持运行，不允许两个 supervisor 同时写同一共享输出目录。

SAAR 扩展实验采用 session-aware 固定伪说话人、固定 original enrollment、匿名 target
按 `N={1,2,5,10,15}` 增长且 5-seed nested sampling。Phase 1/2 的本机/服务器职责、
输入包、断点续跑和 Gate 1 命令见 [SAAR_RUNBOOK.md](SAAR_RUNBOOK.md)；实验数据仍只在
`EXPERIMENT_RESULTS.md` 追加，避免出现多个相互矛盾的结果总账。

## Git 版本管理

主分支为 `main`，复现实验基线标签为 `v0.1.0-reproduction-baseline`。StreamVoiceAnon 使用 submodule 固定版本；clone 时应同时初始化 submodule：

```powershell
git clone --recurse-submodules git@github.com:wwwYYYcom/multimodal-SV.git
```

数据、checkpoint 和逐条实验产物不会提交到普通 Git 历史；每次实验应把配置、代码以及结果摘要一并提交，并将全部结果追加到 `EXPERIMENT_RESULTS.md`。详细分支、提交和远程仓库命令见总账第 13 节。
