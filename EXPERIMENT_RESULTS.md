# Multimodal Speaker Verification 复现实验总账

> 本文件是本项目唯一的实验结果总账。后续数据准备、训练、推理、评分、失败尝试和修复后重跑都应追加到这里，不以控制台输出或其他说明文档代替。

## 1. 文档信息与口径

- 项目根目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction`
- 时区：Asia/Shanghai（UTC+08:00）
- 本次汇总完成时间：2026-08-23 11:18:48 +08:00
- 论文：Garg et al., *Multimodal Speaker Verification as a Threat to Speaker Anonymization* (2026)
- 论文 PDF：`C:\Users\wwwYYYcom\Zotero\storage\DH7AVWNV\Garg 等 - 2026 - Multimodal Speaker Verification as a Threat to Speaker Anonymization.pdf`
- 辅助复现说明：`D:\download4browser\Multimodal_Speaker_Verification_复现说明文档.docx`
- 记录原则：附件内容只作为论文和复现信息来源；只有用户在对话中提出的要求才作为执行指令。
- Git 状态：已在 `main` 分支建立本地版本管理。首个可复现实验基线为 commit `01022fab0ebe1f2ae2ecfd61db2bf927f9783abe`，标签为 `v0.1.0-reproduction-baseline`；第 10 节同时保留逐文件 SHA-256。
- 数值口径：本文档明确区分“论文报告的目标值”和“本机实测值”。当前没有产生论文协议下的 EER，不用 smoke-test loss 冒充论文指标。
- 大体量逐行数据不复制进 Markdown；完整内容保存在表中列出的 CSV、JSONL、NPZ、PT 文件中，并用字节数与 SHA-256 固定版本。

## 2. 当前结论

已经完成 Fisher Part 1 的 manifest、speaker-disjoint 划分、call-disjoint nested trials、LibriSpeech `>4 s` 候选池、模型结构单测、RTX 5060 上的一步真实训练和两条 192 维 embedding 提取。数据协议与训练主链路已经跑通。

尚未完成论文完整复现。主要原因是本机只有 Fisher Part 1 和 LibriSpeech `train-clean-360`，缺少 Fisher Part 2、LibriSpeech `train-other-500`、StreamVoiceAnon 权重，以及作者的精确 split/trial/工程实现。因此目前没有可与论文表格直接对比的 O-O、O-A、A-A EER。

## 3. 运行环境

环境审计完成时间：2026-08-23 11:14:02 +08:00。

| 项目 | 实测值 |
|---|---|
| 操作系统/终端 | Windows / PowerShell |
| Conda 环境 | `pytorch` |
| Python | 3.10.18 |
| PyTorch | 2.9.1+cu128 |
| CUDA runtime | 12.8 |
| CUDA 可用 | True |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| 显存 | 8151 MiB |
| NVIDIA driver | 610.47 |
| Transformers | 4.56.2 |
| NumPy | 1.26.4 |
| pandas | 2.3.3 |
| SciPy | 1.13.1 |
| scikit-learn | 1.7.2 |
| SPHERE 解码 | `desphere[fast]`；未检测到 `sph2pipe` |
| WavLM | `microsoft/wavlm-large`，官方 Hugging Face snapshot 已缓存 |

安装命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m pip install -e '.[audio,test]'
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m pip install 'desphere[fast]'
```

## 4. 输入数据与资源状态

| 资源 | 本机路径/状态 | 论文所需情况 | 影响 |
|---|---|---|---|
| Fisher 音频 | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fisher_eng_tr_sp_LDC2004S13\fisher_eng_tr_sp_LDC2004S13` | 论文规模对应 Part 1 + Part 2 | 当前仅 Part 1 |
| Fisher transcript | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fe_03_p1_tran_LDC2004T19\fe_03_p1_tran\data\trans` | 需要与完整音频匹配 | 当前仅 Part 1 |
| Fisher calldata | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fe_03_p1_tran_LDC2004T19\fe_03_p1_tran\doc\fe_03_p1_calldata.tbl` | 用于说话人/通话映射 | 已用于本轮 |
| LibriSpeech | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\LibriSpeech\train-clean-360` | `train-clean-360` + `train-other-500` | 缺 `train-other-500` |
| StreamVoiceAnon | 本机未找到 checkpoint | 生成匿名 Fisher 和 semi-informed 条件 | O-A/A-A 暂不能运行 |
| 作者代码/权重 | 截至检查时官方仓库仍仅说明“后续发布” | 精确实现、split、trials、权重 | 无法逐项完全对齐 |

## 5. 实验索引

| ID | 实验 | 状态 | 完成时间（UTC+08:00） | 核心实测结果 | 主要输出 |
|---|---|---|---|---|---|
| E01 | Fisher Part 1 manifest | 完成 | 2026-08-22 23:17:05 | 5,850 calls；929,364 utterances；7,066 speakers | `artifacts\metadata\fisher_manifest.csv` |
| E02 | Speaker-disjoint split | 本地兼容规模完成 | 2026-08-22 23:17:19 | train/val/eval = 5,231/229/1,606；三者不相交 | `artifacts\metadata\speaker_splits.csv` |
| E03 | Evaluation nested trials | 完成 | 2026-08-22 23:40:07 | 1,605 target + 1,605 nontarget；N=5/10/15 | `artifacts\trials\evaluation.jsonl` |
| E04 | LibriSpeech target pool | 部分资源下完成 | 2026-08-22 23:19:40 | 99,278 utterances；921 speakers；仅 clean-360 | `artifacts\metadata\librispeech_target_pool.csv` |
| E05 | 单元测试 | 完成 | 2026-08-23 11:14:02 | 5 passed | 控制台；测试代码见第 10 节 |
| E06 | 无权重模型结构 smoke | 完成 | 2026-08-23 11:14:56 | ECAPA/query/mean 均输出 `[2,24]`；已归一化 | 控制台记录见 6.5 |
| E07-a | GPU 一步训练，初始调试 | 完成、非最终 | 2026-08-22 23:30:27 | loss 18.033920；4.604990 s | `results\runs\gpu_smoke` |
| E07-b | GPU 一步训练，精简 checkpoint | 完成、非最终 | 2026-08-22 23:36:36 | loss 18.474255；3.126842 s | `results\runs\gpu_smoke_v2` |
| E07-c | GPU 一步训练，grouped loader | 完成、当前有效 smoke | 2026-08-22 23:41:29 | loss 17.303108；2.033191 s | `results\runs\gpu_smoke_grouped` |
| E08 | 两条 embedding 提取 | 完成 | 2026-08-22 23:42:05 | shape `[2,192]`；L2 norm 约为 1 | `artifacts\embeddings\gpu_smoke_2.npz` |
| E09 | 论文 EER 评测 | 未运行 | — | 没有 EER 数据 | 等待完整训练/匿名数据 |

## 6. 实验详情

### 6.1 E01：Fisher Part 1 manifest

- 目的：对齐 Fisher SPHERE 音频、transcript turn 和 calldata speaker，过滤小于 1.0 秒的片段。
- 配置：`configs\local_fisher_p1.yaml`
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\config.py`、`src\mmsv\data\fisher.py`
- 完成时间：2026-08-22 23:17:05 +08:00

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli prepare-fisher `
  --config configs/local_fisher_p1.yaml `
  --output artifacts/metadata/fisher_manifest.csv
```

全部审计数据：

| 字段 | 实测值 |
|---|---:|
| audio_files | 5,850 |
| transcripts | 5,850 |
| matched_calls | 5,850 |
| missing_audio_for_transcript | 0 |
| missing_transcript_for_audio | 0 |
| utterances | 929,364 |
| speakers | 7,066 |
| speakers_with_at_least_2_calls | 3,302 |
| min_duration | 1.0 s |

输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\fisher_manifest.csv` | 300,904,760 | `e9172730119921a6358d7e47125c8d6e0a77950ef6edb4f90eac0e6ca2574d12` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\fisher_manifest.audit.json` | 383 | `a82053456a68ea90b0e6155294833e01383745dabd9c398d9211e7c907157c4b` |

### 6.2 E02：Speaker-disjoint split

- 目的：构造互不重叠的 train/validation/evaluation speaker 集；evaluation speaker 至少有两通 call。
- 随机种子：1234。
- 种子来源：复现工程自行固定；论文未公开。
- 配置：`configs\local_fisher_p1.yaml`，`require_exact_counts=false`。
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\config.py`、`src\mmsv\data\splits.py`
- 完成时间：2026-08-22 23:17:19 +08:00

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli split-speakers `
  --config configs/local_fisher_p1.yaml `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --output artifacts/metadata/speaker_splits.csv
```

全部审计数据：

| 字段 | 请求值 | 实测值 |
|---|---:|---:|
| train speakers | 5,712 | 5,231 |
| validation speakers | 250 | 229 |
| evaluation speakers | 1,753 | 1,606 |
| evaluation_min_calls | 2 | 2 |
| evaluation_candidates | — | 3,302 |
| split disjoint | True | True |

请求总数 7,715，大于本机 Part 1 的 7,066 个 speaker，因此不能生成论文精确规模。这里按论文比例缩放并明确标为本地兼容 split。

输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.csv` | 137,401 | `0fd8f12f5e05cd0d55c28485769da842b21ac0ccbd51824c21d568c0d9b386ff` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.audit.json` | 523 | `de22367a65331458305ace658f9797e165f977f29c764f6e645ec2192145a810` |

### 6.3 E03：Evaluation nested trials

- 目的：为每个合格 evaluation speaker 建立 target/nontarget trial；enrollment 与 target 按 call 分池，避免 call 泄漏；从一次 max-N 抽样中截取 N=5/10/15，满足嵌套关系。
- 配置：`configs\local_fisher_p1.yaml`
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\data\trials.py`
- 完成时间：2026-08-22 23:40:07 +08:00

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli build-trials `
  --config configs/local_fisher_p1.yaml `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --splits artifacts/metadata/speaker_splits.csv `
  --split evaluation `
  --output artifacts/trials/evaluation.jsonl
```

全部审计数据：

| 字段 | 实测值 |
|---|---:|
| seed | 1234 |
| n_values | 5, 10, 15 |
| nested_sampling | True |
| call_disjoint_pools | True |
| eligible_speakers | 1,605 |
| ineligible_speakers | 1 |
| target trials | 1,605 |
| nontarget trials | 1,605 |
| trials 合计 | 3,210 |

输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.jsonl` | 2,590,963 | `69529d37299b19f875a42ddf5ef07b1e8d5e1779c8a34a44a2c5822e43437276` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.audit.json` | 368 | `816d1e317fa6c71aa80e0dd2256836db47eb1f723b0d7bca578645a2e36b89dd` |

### 6.4 E04：LibriSpeech 匿名化 target pool

- 目的：筛选严格大于 4.0 秒的 LibriSpeech utterance，作为匿名化 reference 候选池。
- 实际输入：只有 `train-clean-360`。
- 配置：`configs\local_fisher_p1.yaml`
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\data\librispeech.py`
- 完成时间：2026-08-22 23:19:40 +08:00

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli build-libri-pool `
  --config configs/local_fisher_p1.yaml `
  --output artifacts/metadata/librispeech_target_pool.csv `
  --min-duration 4.0
```

| 字段 | 实测值 |
|---|---:|
| subset | train-clean-360 |
| utterances，duration > 4.0 s | 99,278 |
| speakers | 921 |
| 是否同时含 clean-360 与 other-500 | False |
| 缺失 subset | train-other-500 |

输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\librispeech_target_pool.csv`，20,264,920 字节，SHA-256 `ff04bd9e77d7702560e75147a44fd2e313f1957cf2e65b78927463062f19bea2`。

### 6.5 E05/E06：自动测试与模型结构 smoke

自动测试命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m pytest -q
```

完成时间：2026-08-23 11:14:02 +08:00。结果：`5 passed`。

覆盖文件：`tests\test_fisher.py`、`tests\test_metrics.py`、`tests\test_models.py`、`tests\test_protocol.py`。

模型结构 smoke 命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli model-smoke
```

完成时间：2026-08-23 11:14:56 +08:00。实测输出：

```json
{
  "ecapa_shape": [2, 24],
  "query_shape": [2, 24],
  "mean_shape": [2, 24],
  "normalized": true
}
```

运行代码：`src\mmsv\models.py`、`src\mmsv\aggregation.py`、`src\mmsv\cli.py`。这里故意使用小维度随机张量，只验证 shape、聚合路径和归一化，不是训练结果。

### 6.6 E07：RTX 5060 一步真实训练

共同设置：WavLM-Large 冻结、ECAPA embedding 192 维、4 秒 crop、batch size 2、gradient accumulation 1、AAM margin 0.2、scale 30、learning rate `5e-4`、weight decay `1e-4`、gradient clip 1.0、seed 1234。配置文件为 `configs\gpu_smoke.yaml`。

运行代码：`src\mmsv\cli.py`、`src\mmsv\config.py`、`src\mmsv\audio.py`、`src\mmsv\models.py`、`src\mmsv\train.py`。

三次运行都是链路验证，不足以报告收敛性能；它们使用了不同阶段的实现，因此 loss 不能横向比较。

| 运行 | global_step | epoch | loss | LR | 训练日志中的 seconds | checkpoint 字节数 | 定位 |
|---|---:|---:|---:|---:|---:|---:|---|
| E07-a `gpu_smoke` | 1 | 0 | 18.033920288085938 | 0.0005 | 4.604990482330322 | 1,377,753,009 | 初始端到端调试；checkpoint 过大，已弃用 |
| E07-b `gpu_smoke_v2` | 1 | 0 | 18.474254608154297 | 0.0005 | 3.126842498779297 | 115,736,515 | 验证精简 checkpoint；非最终 loader |
| E07-c `gpu_smoke_grouped` | 1 | 0 | 17.30310821533203 | 0.0005 | 2.033191442489624 | 115,736,515 | 当前有效 smoke；call-side grouped loader |

当前有效命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli train-audio `
  --config configs/gpu_smoke.yaml `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --splits artifacts/metadata/speaker_splits.csv `
  --output-dir results/runs/gpu_smoke_grouped `
  --max-steps 1
```

当前有效 checkpoint 元数据：epoch 0、global_step 1、5,231 个训练 speaker class、226 个模型张量、1 个 classifier 张量。冻结的 WavLM 权重不重复写入 checkpoint，加载时使用已缓存的 Hugging Face snapshot。

三个运行的输出与校验值：

| 文件 | 完成时间 | 字节数 | SHA-256 |
|---|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\train.jsonl` | 2026-08-22 23:30:25 | 224 | `28980c6a2be099dc1e7a7cc15185e82d8264836e1b45558ef5ae78d8e70335c9` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\last.pt` | 2026-08-22 23:30:27 | 1,377,753,009 | `a9d22886ea35d5707c6c3b1cf5806b65e453b5723c6ba1bf513b11911b703cec` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\train.jsonl` | 2026-08-22 23:36:35 | 227 | `92c108bbead7720fce3a1e4d79481979d37207116af837b624183ca93d50db56` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\last.pt` | 2026-08-22 23:36:36 | 115,736,515 | `3d0a9acc3d0ecd64400d297e014d1167fd349f4b2295d7368556213110770e5d` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\train.jsonl` | 2026-08-22 23:41:29 | 231 | `b7111242e1e632d1af3c2da7ffe4f6ce6fda934575a81fc1dd1fe7eec424685a` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\last.pt` | 2026-08-22 23:41:29 | 115,736,515 | `d2d5e38ab9dcbe987ef1f8e8d099532589656dbe614362743f6fbb13b65f6269` |

### 6.7 E08：Embedding 提取

- 输入 checkpoint：`results\runs\gpu_smoke_grouped\last.pt`
- 输入 manifest：`artifacts\metadata\fisher_manifest.csv`
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\audio.py`、`src\mmsv\models.py`、`src\mmsv\train.py`
- 完成时间：2026-08-22 23:42:05 +08:00

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli extract-embeddings `
  --checkpoint results/runs/gpu_smoke_grouped/last.pt `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --output artifacts/embeddings/gpu_smoke_2.npz `
  --limit 2
```

| 字段 | 实测值 |
|---|---|
| utterance IDs | `fe_03_00001_A_0001`, `fe_03_00001_B_0001` |
| embedding shape | `[2, 192]` |
| dtype | float32 |
| L2 norms | `[1.0, 0.9999999403953552]` |

完整 384 个浮点数保存在 `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\embeddings\gpu_smoke_2.npz`。文件 1,907 字节，SHA-256 `7fa7f74b1bf5a0cc2e9fb4699ae4708983bb0848520aaf0c88f6e6ea0ae83ba2`。

### 6.8 E09：EER 评测状态

尚未运行可报告的论文 EER。`src\mmsv\metrics.py` 和 `src\mmsv\multimodal.py` 已实现 cosine/EER 与 mean/query/frame 聚合路径，CLI 已提供 `score-mean`，但一步 smoke checkpoint 不能代表训练完成的 ASV 模型；匿名化语音也尚未生成。因此结果栏必须保持空缺，不能填入论文数字或 smoke loss。

完整训练后的 mean-pooling 评分命令模板：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli score-mean `
  --trials artifacts/trials/evaluation.jsonl `
  --original-embeddings artifacts/embeddings/original.npz `
  --condition O-O `
  --n 5 `
  --output results/o_o_mean_n5.csv
```

## 7. 失败尝试、工程修复与保留证据

| 事件 | 现象 | 处理 | 当前状态 |
|---|---|---|---|
| 论文精确 split 尝试 | `configs\paper.yaml` 请求 7,715 speakers，而 Part 1 只有 7,066 | 保留严格配置用于资源完整后的验收；本轮改用 `require_exact_counts=false` 的本地配置 | 仍受 Fisher Part 2 缺失阻塞 |
| Hugging Face 镜像 | 原环境的 `hf-mirror.com` 不可达 | 配置固定为 `https://huggingface.co` | 已解决，WavLM-Large 已缓存 |
| Transformers 后台 safetensors 转换 | 首次常规加载出现等待/挂起 | 先用 `snapshot_download` 完成快照，再从本地缓存加载 | 已解决 |
| 初始 checkpoint 过大 | E07-a 产物约 1.38 GB | 冻结 WavLM 不再重复写入 checkpoint | E07-b/c 降至约 115.7 MB |
| 随机逐 turn loader | 约 296,300 batches/epoch，Windows SPHERE 解码效率不可接受 | 改为 call-side grouped loader；同 call A/B 相邻并复用解码缓存 | 当前约 3,751 batches/epoch；E07-c 已验证 |
| 匿名化与 semi-informed | 缺 StreamVoiceAnon checkpoint、完整 Libri pool 和匿名 Fisher | 仅保留配置/代码接口，不伪造结果 | 未完成 |

说明：`results\runs\gpu_smoke` 和 `results\runs\gpu_smoke_v2` 是为保留调试证据而存在的非最终目录；后续正式实验不得从它们继续训练。当前 smoke 基线是 `results\runs\gpu_smoke_grouped`。

## 8. 论文目标值（仅供验收，不是本机结果）

论文 O-O Table II 的 EER% 目标：

| 聚合 | N=5 | N=10 | N=15 |
|---|---:|---:|---:|
| Mean | 3.87 | 3.29 | 3.09 |
| Query | 3.39 | 2.54 | 2.27 |
| Frame Concat | 3.26 | 2.33 | 2.10 |

论文 A-A semi-informed Table I 中 Frame Concat 的 EER% 目标为 15.10 / 8.83 / 6.96（N=5/10/15）。只有拿到完整数据、匿名化权重和作者协议文件后，才以小数点后二位对齐作为验收目标。

## 9. 全部输出路径清单

权威数据产物：

```text
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\fisher_manifest.csv
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\fisher_manifest.audit.json
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.csv
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.audit.json
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\librispeech_target_pool.csv
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.audit.json
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\last.pt
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\embeddings\gpu_smoke_2.npz
```

保留的非最终调试产物：

```text
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\last.pt
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\last.pt
```

## 10. 使用的运行代码、配置与版本指纹

CLI 入口与职责：

| 文件 | 用途 |
|---|---|
| `src\mmsv\cli.py` | 全部命令行入口 |
| `src\mmsv\config.py` | YAML 配置加载 |
| `src\mmsv\data\fisher.py` | Fisher manifest |
| `src\mmsv\data\splits.py` | speaker split |
| `src\mmsv\data\trials.py` | nested、call-disjoint trials |
| `src\mmsv\data\librispeech.py` | LibriSpeech target pool |
| `src\mmsv\audio.py` | SPHERE/普通音频解码、裁剪 |
| `src\mmsv\models.py` | WavLM + ECAPA、AAM classifier |
| `src\mmsv\aggregation.py` | mean/query/frame 聚合 |
| `src\mmsv\metrics.py` | cosine score、EER |
| `src\mmsv\train.py` | 训练、checkpoint、embedding 提取 |
| `src\mmsv\multimodal.py` | O-O/O-A/A-A 多 utterance 评分 |

本轮源文件 SHA-256：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `pyproject.toml` | 808 | `ad200b4473eded7e4c93c9ed9c7849e3131284f1e1d6e7d9e4ac7fb52166aa6c` |
| `configs\paper.yaml` | 1,311 | `51b2e2271b170ff12835fbd26d68a0938dbefd7c7c05c5d21b760a3af065c82f` |
| `configs\local_fisher_p1.yaml` | 1,514 | `36739f8b1c949c16190c9b360a9f27ddf70da64dde2a2099e814f95c79031afa` |
| `configs\gpu_smoke.yaml` | 482 | `b35222164eb9185f8f1a25931e2f300707b8016adf385b774a829dba6d53893a` |
| `configs\semi_local.yaml` | 469 | `edaa97bf7d9f85652899bff41ec86dddb8aafb3fc48da2e716adc9602bc49ffd` |
| `src\mmsv\__init__.py` | 89 | `d1f4a61fa0b2491a3fc25e5975d11b7e92a71edb35a682ef46c84bdb770e1d82` |
| `src\mmsv\config.py` | 803 | `d175a20a350ffe3c3bb3c19b7c01a48973c4fd622bee62cdb8b1f6d83019bf4b` |
| `src\mmsv\data\fisher.py` | 6,122 | `9e07adc7be42d10ed8a41405f9d71d1d8ccd4b3f5197ccfe239fdf998ab10ccb` |
| `src\mmsv\data\splits.py` | 4,285 | `6c8cddc031eb21e4491e459fb3b61406b9bf37e523663fbe227d5025c742a315` |
| `src\mmsv\data\trials.py` | 7,310 | `39b93bf9e2bade20bfb16d2fe8bc5017e3ec91d716c11642c1957eb300876976` |
| `src\mmsv\data\librispeech.py` | 1,909 | `a97ed1181e346251514db84dda746afd6c3c9b82a09096cac850114b32dc0397` |
| `src\mmsv\audio.py` | 4,482 | `b6976bd512e5c190ac89b5d34b7c98ac7a3e4b8e38fd4ebe208544edd40c4493` |
| `src\mmsv\aggregation.py` | 2,128 | `c17fe1ac00b0813b368444e32d63451c4112fcd4db2fc1d453e6c58a715c9b64` |
| `src\mmsv\models.py` | 10,475 | `a7c9a589c9edc28d861ce8494eed6e0a41559cbf57ec71c5106bd25278d2ea43` |
| `src\mmsv\metrics.py` | 4,593 | `0c6da0d5ef0f2a2a22e407c6d05e8dfbe715f20240f81fda0db69698468680e0` |
| `src\mmsv\train.py` | 11,656 | `9d99996dd3f2bc7193b17387589b614a6e2732196a98c35315d0ccf97b68d166` |
| `src\mmsv\multimodal.py` | 4,247 | `9d169e24eb85c8a2f8d05c5603cda48de5a68857850d4e52849324a1dc09549f` |
| `src\mmsv\cli.py` | 7,329 | `ec0454968a1e765daa2cd871b597b81f481f5e2ea22e31828c7cbb1b916f9ac8` |
| `tests\test_fisher.py` | 923 | `16e0573a4937dca2ab4c5e307130aad1ba1571a63979c2177258c62fe5c66d12` |
| `tests\test_metrics.py` | 321 | `fd8ee0ef0db2dd4fbd43bbdbed73b77d660e24dfd41ae518a3641149ab01e131` |
| `tests\test_models.py` | 1,360 | `aae35a4b447286e8e2893c5515c861d47df1e35f034a42d28cbc01efce892540` |
| `tests\test_protocol.py` | 742 | `86321640621ae72005c3e454638c3cda607a1d4dda94f09b38083ae8a95f6c52` |

## 11. 下一阶段与验收条件

1. 补齐 Fisher Part 2，使用 `configs\paper.yaml` 重建精确 5,712/250/1,753 split 和固定 trials。
2. 补齐 LibriSpeech `train-other-500`，重建论文完整匿名化 target pool。
3. 获取 StreamVoiceAnon checkpoint，生成匿名 Fisher manifest 与 O-A/A-A embeddings。
4. 执行 30 epoch lazy-informed 训练与 15 epoch semi-informed 训练；semi-informed 必须用 `--init-from` 重置 optimizer，而不是普通 `--resume`。
5. 对 Mean、Query、Frame Concat 在 N=5/10/15、O-O/O-A/A-A 条件下输出每个 trial 的 score CSV 和汇总 EER。
6. 实现并评测 Whisper、LUAR、prosody、RJCA 等 Level 2–3 多模态组合。
7. 作者发布代码、split、trials 或权重后，固定版本/commit 并重新审计所有未公开选择。

## 12. 后续追加模板

每次实验结束后复制以下小节到本文件末尾；失败实验也要记录，不覆盖旧结果。

```markdown
### EXXX：实验名称

- 状态：完成 / 失败 / 中止 / 部分完成
- 开始时间：YYYY-MM-DD HH:mm:ss +08:00
- 完成时间：YYYY-MM-DD HH:mm:ss +08:00
- 目的：
- 输入数据及 SHA-256：
- 配置文件及 SHA-256：
- 运行代码文件/commit：
- 完整命令：
- 硬件与软件环境：
- 全部超参数：
- 全部实验数据：
- 关键指标：
- 输出文件绝对路径、字节数、SHA-256：
- 日志路径：
- 异常、修复与解释：
- 与论文目标的差异：
```

## 13. Git 版本管理记录

- 建立时间：2026-08-23 11:57:53 +08:00
- 默认分支：`main`
- 初始基线 commit：`01022fab0ebe1f2ae2ecfd61db2bf927f9783abe`
- 基线标签：`v0.1.0-reproduction-baseline`
- 提交说明：`feat: establish multimodal SV reproduction baseline`
- 提交作者：`wwwYYYcom <779536052@qq.com>`
- 第三方依赖：`third_party/StreamVoiceAnon` 作为 Git submodule 管理，固定在 commit `201705182c045298225071481e7cd59d537e935e`。
- 大文件策略：`artifacts/`、`results/runs/`、`checkpoints/`、`tmp/` 和本地 `data/` 不进入普通 Git 历史；关键结果通过本总账、审计值和 SHA-256 版本化。
- 远程仓库：尚未配置。配置 GitHub/GitLab/Gitee 地址后再执行 push。

推荐实验分支与提交方式：

```powershell
git switch -c experiment/<实验名>
# 修改代码、配置并运行实验；把结果追加到 EXPERIMENT_RESULTS.md
git add src configs tests EXPERIMENT_RESULTS.md README.md
git commit -m "experiment: <实验说明>"
git switch main
git merge --no-ff experiment/<实验名>
```

首次关联远程仓库：

```powershell
git remote add origin <远程仓库地址>
git push -u origin main
git push origin --tags
```

克隆时同时获取 StreamVoiceAnon submodule：

```powershell
git clone --recurse-submodules <远程仓库地址>
# 已经普通 clone 时：
git submodule update --init --recursive
```
