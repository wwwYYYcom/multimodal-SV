# 复现状态（2026-08-23）

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
- 已生成 P1 兼容 split：5,231/229/1,606；evaluation 生成 3,210 个固定 trials，1 位因 pool 不足被审计剔除。
- 已筛出 train-clean-360 中 99,278 条 >4 s reference utterances（921 speakers）。
- WavLM-Large 官方权重已缓存；RTX 5060 上完成 grouped-loader 的 1 个真实优化步（loss=17.3031）并成功回读 2 条 192-d embeddings。
- StreamVoiceAnon checkpoint `dual_ar_delay_0_8.pth` 已下载并完成文件指纹审计。

## 确定的数据范围与剩余限制

| 资源 | 当前采用范围 | 本机状态 | 说明 |
|---|---|---|---|
| Fisher | Part 1 | LDC2004S13/T19 Part 1 已就绪 | 这是最终实验范围，不再等待 Part 2 |
| LibriSpeech target pool | train-clean-360 | 99,278 条 >4 s utterances 已筛选 | 这是最终实验范围，不再等待 other-500 |
| StreamVoiceAnon checkpoint | `dual_ar_delay_0_8.pth` | 已下载，598,136,107 字节 | 尚需完成匿名化流水线运行与验证 |
| 作者代码/split/trials | 当前自行固定并审计 | 官方论文仓库仍未发布精确协议 | 结果按本地范围报告，不能声称精确复现作者 trial |
| SPHERE decoder | shorten-capable decoder | 已安装 `desphere[fast]`；未检测到 sph2pipe | 可运行；sph2pipe 的按段解码会更快 |

## 未完成

- 在固定数据范围内完成 30 epoch O-O 训练、embedding 提取和 EER 表格。
- 使用现有 StreamVoiceAnon checkpoint 生成匿名化 Fisher Part 1，完成 semi-informed 训练和 O-A/A-A 表格。
- Whisper/LUAR/prosody/RJCA 等 Level 2-3；说明文档明确建议 audio 闭环稳定后再做。

## 论文目标值

O-O Table II（Mean/Query/Frame, N=5/10/15）分别为 `3.87/3.29/3.09`、`3.39/2.54/2.27`、`3.26/2.33/2.10` EER%。A-A semi-informed Table I 最强 Frame Concat 为 `15.10/8.83/6.96` EER%。这些数字只作为论文参考值；当前验收目标是在 Part 1 + clean-360 固定范围内得到完整、可重复、可审计的对应表格。
