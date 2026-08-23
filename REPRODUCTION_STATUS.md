# 复现状态（2026-08-22）

> 本文件只保留进度摘要。唯一完整实验总账是 [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md)，所有后续实验结果均追加到该文件。

## 已完成

- 逐页核对 8 页论文、Table I-IV、Figure 1 与 20 页复现说明。
- 核对作者官方仓库；代码和预训练模型尚未发布。
- 盘点本机 GPU/Conda/数据：RTX 5060 Laptop 8 GB；`pytorch` 环境可用 CUDA。
- 实现 Fisher transcript + calldata manifest、speaker-disjoint split、call-disjoint nested trials。
- 实现 LibriSpeech `>4s` reference pool 审计。
- 实现 WavLM-ECAPA、ASP 前 frame extraction、mean/query/frame aggregation、AAM-Softmax。
- 实现可恢复训练、embedding 提取、O-O/O-A/A-A mean-pooling cosine/EER 评分。
- 已生成本机 Fisher Part 1 manifest：5,850/5,850 calls 对齐、929,364 条 >=1 s utterances。
- 已生成 P1 兼容 split：5,231/229/1,606；evaluation 生成 3,210 个固定 trials，1 位因 pool 不足被审计剔除。
- 已筛出 train-clean-360 中 99,278 条 >4 s reference utterances（921 speakers）。
- WavLM-Large 官方权重已缓存；RTX 5060 上完成 grouped-loader 的 1 个真实优化步（loss=17.3031）并成功回读 2 条 192-d embeddings。

## 当前外部阻塞

| 资源 | 论文需要 | 本机状态 | 影响 |
|---|---|---|---|
| Fisher | Part 1 + Part 2 对应论文规模 | 仅 LDC2004S13/T19 Part 1 | 不能精确复现 speaker cardinality 与 EER |
| LibriSpeech target pool | train-clean-360 + train-other-500 | 仅 train-clean-360 | 不能按论文生成匿名化 reference pool |
| StreamVoiceAnon checkpoint | 发布权重 | 本机目录无权重 | 暂不能生成 O-A/A-A Fisher |
| 作者代码/split/trials | 论文脚注承诺公开 | 官方仓库仅 README | query/ECAPA 细节与精确 trial 不可完全对齐 |
| SPHERE decoder | shorten-capable decoder | 已安装 `desphere[fast]`；未检测到 sph2pipe | 可运行；sph2pipe 的按段解码会更快 |

## 未完成

- 生成匿名化音频后，提取对应 embeddings 并填充 O-A/A-A 表格。
- StreamVoiceAnon 匿名化、semi-informed 训练和 O-A/A-A 表格。
- Whisper/LUAR/prosody/RJCA 等 Level 2-3；说明文档明确建议 audio 闭环稳定后再做。

## 论文目标值

O-O Table II（Mean/Query/Frame, N=5/10/15）分别为 `3.87/3.29/3.09`、`3.39/2.54/2.27`、`3.26/2.33/2.10` EER%。A-A semi-informed Table I 最强 Frame Concat 为 `15.10/8.83/6.96` EER%。只有获得完整数据、匿名化权重及作者协议文件后，才将“小数点后二位一致”作为验收目标。
