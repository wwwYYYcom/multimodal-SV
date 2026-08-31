# 复现状态（2026-08-27）

> 本文件只保留进度摘要。唯一完整实验总账是 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)，所有后续实验结果均追加到该文件。

> 用户已确认数据范围固定为 Fisher Part 1 + LibriSpeech `train-clean-360`。不再将 Fisher Part 2 或 `train-other-500` 作为待补资源。

## 已完成

- 逐页核对 8 页论文、Table I-IV、Figure 1 与 20 页复现说明。
- 核对作者官方仓库；代码和预训练模型尚未发布。
- 盘点本机 GPU/Conda/数据：RTX 5060 Laptop 8 GB；`pytorch` 环境可用 CUDA。
- 实现 Fisher transcript + calldata manifest、speaker-disjoint split、call-disjoint nested trials。
- 实现 LibriSpeech `>4s` reference pool 审计。
- 实现 WavLM-ECAPA、ASP 前 frame extraction、mean/query/frame aggregation、AAM-Softmax。
- 实现可恢复训练、embedding 提取、O-O/O-A/A-A mean-pooling cosine/EER 评分。
- 已生成本机 Fisher Part 1 manifest：5,850/5,850 calls 对齐、929,364 条 >=1 s utterances。
- 已生成 trial-capable P1 split：5,231/229/1,606；validation 生成 45,800 个固定 trials，evaluation 生成 3,212 个固定 trials，两个 split 均无 speaker 被剔除。
- 已筛出 train-clean-360 中 99,278 条 >4 s reference utterances（921 speakers）。
- WavLM-Large 官方权重已缓存；RTX 5060 上完成 grouped-loader 的 1 个真实优化步（loss=17.3031）并成功回读 2 条 192-d embeddings。
- StreamVoiceAnon 主模型及 4 个配套权重均已下载并完成文件指纹审计；离线模型初始化无 missing/unexpected keys。
- StreamVoiceAnon CPU 端到端 smoke 已成功生成 1 条 16 kHz FLAC；重复运行验证可跳过已有输出。
- 已生成 evaluation 匿名化计划（86,222 utterances / 94.14 小时）和 semi-informed train 计划（7,272 call-sides / 7.65 小时，覆盖 5,231 speakers）。
- corrected WavLM-ECAPA Mean O-O 已完成：30 epochs、268,560 steps；N=1/5/10/15 EER 分别为 `15.5044/4.1719/3.1133/2.9265%`。N=5/10/15 已达到论文 Mean O-O 的相近量级。

## 确定的数据范围与剩余限制

| 资源 | 当前采用范围 | 本机状态 | 说明 |
|---|---|---|---|
| Fisher | Part 1 | LDC2004S13/T19 Part 1 已就绪 | 这是最终实验范围，不再等待 Part 2 |
| LibriSpeech target pool | train-clean-360 | 99,278 条 >4 s utterances 已筛选 | 这是最终实验范围，不再等待 other-500 |
| StreamVoiceAnon checkpoint | 主模型 + ASR tokenizer + Firefly + CAMPPlus + Spark encoder | 5 个权重均已下载并校验 | CPU/GPU runner 已接通；O-O 已完成，GPU 可转入批量匿名化阶段 |
| 作者代码/split/trials | 当前自行固定并审计 | 官方论文仓库仍未发布精确协议 | 结果按本地范围报告，不能声称精确复现作者 trial |
| SPHERE decoder | shorten-capable decoder | 已安装 `desphere[fast]`；未检测到 sph2pipe | 可运行；sph2pipe 的按段解码会更快 |

## 最新完成状态

- evaluation StreamVoiceAnon 已于 2026-08-30 09:47:04 +08:00 完成：86,222 条、5.585 GB，最终校验 valid，missing/unreadable/wrong-format/nonfinite 均为 0，实际双进程 RTF 0.6956。
- 匿名 embedding 与 Mean O-A/A-A N=1/5/10/15 已于 2026-08-30 11:54:11 +08:00 全部完成。
- O-A N=1/5/10/15 EER：`43.5243/38.4184/36.7995/36.6750%`。
- lazy-informed A-A N=1/5/10/15 EER：`47.3848/39.3524/31.5068/25.7783%`。
- 结果目录为 `results/o_a_corrected`、`results/a_a_corrected`；完整指标、文件路径、日志与 SHA-256 见 `EXPERIMENT_RESULTS.md` 第 26 节。
- 全 utterance train 匿名化当前运行正常。初始任务在第 13,931 项的 43.24 秒超长语音上触发上游 2048 KV-cache CUDA 越界；现已对仅 13 条超过 30 秒的输入采用分块匿名化后拼接，并加入 worker 自动重启。2026-08-31 13:17:15 +08:00 以提交 `e975a5e` 恢复：supervisor PID `94440`，worker PID `102176/67860`。截至 13:23:22 已生成 30,922 / 572,951 条（5.396971%）、1,779,006,684 字节；两个 worker 均已越过原故障点，GPU 约 76%、4.95 GiB，D 盘剩余 42,053,750,784 字节。完成后自动进入 15 epoch semi-informed 训练和 O-A/A-A 评分。

## 未完成

- 按全 utterance 计划匿名化 Fisher Part 1 train split 的 572,951 条语音，随后从 corrected lazy checkpoint 初始化并重置 optimizer，完成 15 epoch semi-informed 训练和 O-A/A-A 表格；7,272 条 one-per-call-side 近似已中止且不用于结果。
- Whisper/LUAR/prosody/RJCA 等 Level 2-3；说明文档明确建议 audio 闭环稳定后再做。

## 论文目标值

O-O Table II（Mean/Query/Frame, N=5/10/15）分别为 `3.87/3.29/3.09`、`3.39/2.54/2.27`、`3.26/2.33/2.10` EER%。A-A semi-informed Table I 最强 Frame Concat 为 `15.10/8.83/6.96` EER%。这些数字只作为论文参考值；当前验收目标是在 Part 1 + clean-360 固定范围内得到完整、可重复、可审计的对应表格，并额外报告论文未提供的 N=1 单语句对照。
