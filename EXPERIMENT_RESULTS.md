# Multimodal Speaker Verification 复现实验总账

> 本文件是本项目唯一的实验结果总账。后续数据准备、训练、推理、评分、失败尝试和修复后重跑都应追加到这里，不以控制台输出或其他说明文档代替。

## 1. 文档信息与口径

- 项目根目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction`
- 时区：Asia/Shanghai（UTC+08:00）
- 本次汇总完成时间：2026-08-24 15:26:52 +08:00
- 论文：Garg et al., *Multimodal Speaker Verification as a Threat to Speaker Anonymization* (2026)
- 论文 PDF：`C:\Users\wwwYYYcom\Zotero\storage\DH7AVWNV\Garg 等 - 2026 - Multimodal Speaker Verification as a Threat to Speaker Anonymization.pdf`
- 辅助复现说明：`D:\download4browser\Multimodal_Speaker_Verification_复现说明文档.docx`
- 记录原则：附件内容只作为论文和复现信息来源；只有用户在对话中提出的要求才作为执行指令。
- 数据范围决策：用户于 2026-08-23 15:09:33 +08:00 明确指定只使用 Fisher Part 1 与 LibriSpeech `train-clean-360`。后续不等待、不引入 Fisher Part 2 或 `train-other-500`。
- Git 状态：已在 `main` 分支建立本地版本管理。首个可复现实验基线为 commit `01022fab0ebe1f2ae2ecfd61db2bf927f9783abe`，标签为 `v0.1.0-reproduction-baseline`；第 10 节同时保留逐文件 SHA-256。
- 数值口径：本文档明确区分“论文报告的目标值”和“本机实测值”。当前没有产生论文协议下的 EER，不用 smoke-test loss 冒充论文指标。
- 大体量逐行数据不复制进 Markdown；完整内容保存在表中列出的 CSV、JSONL、NPZ、PT 文件中，并用字节数与 SHA-256 固定版本。

## 2. 当前结论

已经完成 Fisher Part 1 的 manifest、speaker-disjoint 划分、call-disjoint nested trials、LibriSpeech `>4 s` 候选池、模型结构单测、RTX 5060 上的一步真实训练和两条 192 维 embedding 提取。数据协议与训练主链路已经跑通。

项目目标现已明确为“Fisher Part 1 + LibriSpeech `train-clean-360` 范围复现”。现有数据不再视为临时替代或待补状态。尚未产生可报告的 O-O、O-A、A-A EER；作者精确 split/trial/工程细节仍未公开，因此未来结果必须标注本地范围，不能声称是论文完整数据规模的精确数字。

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
| Fisher 音频 | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fisher_eng_tr_sp_LDC2004S13\fisher_eng_tr_sp_LDC2004S13` | 当前固定为 Part 1 | 已就绪；不使用 Part 2 |
| Fisher transcript | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fe_03_p1_tran_LDC2004T19\fe_03_p1_tran\data\trans` | 当前固定为 Part 1 transcript | 已就绪；与 5,850 个 calls 全部匹配 |
| Fisher calldata | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\fisher\fe_03_p1_tran_LDC2004T19\fe_03_p1_tran\doc\fe_03_p1_calldata.tbl` | 用于说话人/通话映射 | 已用于本轮 |
| LibriSpeech | `D:\deeplearning\realtimeVoiceAnon\dataset\prefor_vpc2024\Voice-Privacy-Challenge-2024-main\corpora\LibriSpeech\train-clean-360` | 当前固定为 `train-clean-360` | 已筛选 99,278 条 >4 s utterances；不使用 other-500 |
| StreamVoiceAnon | `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\third_party\StreamVoiceAnon\pretrained_checkpoints\dual_ar_delay_0_8.pth` | 生成匿名 Fisher 和 semi-informed 条件 | 已下载；598,136,107 字节；SHA-256 `df703a1a710c807ad0651dd1bbe45556bf5f3a47f1a79929ec3e6e8fecc56583` |
| 作者代码/权重 | 截至检查时官方仓库仍仅说明“后续发布” | 精确实现、split、trials、权重 | 无法逐项完全对齐 |

## 5. 实验索引

| ID | 实验 | 状态 | 完成时间（UTC+08:00） | 核心实测结果 | 主要输出 |
|---|---|---|---|---|---|
| E01 | Fisher Part 1 manifest | 完成 | 2026-08-22 23:17:05 | 5,850 calls；929,364 utterances；7,066 speakers | `artifacts\metadata\fisher_manifest.csv` |
| E02 | Trial-capable speaker-disjoint split | 完成、替代旧 split | 2026-08-23 15:34:07 | train/val/eval = 5,231/229/1,606；val/eval 均满足 call-disjoint N=15 | `artifacts\metadata\speaker_splits.csv` |
| E03 | Evaluation nested trials | 完成、已重建 | 2026-08-23 15:34:21 | 1,606 target + 1,606 nontarget；0 speaker 剔除 | `artifacts\trials\evaluation.jsonl` |
| E11 | Validation nested trials | 完成 | 2026-08-23 15:34:12 | 22,900 target + 22,900 nontarget；229/229 speakers | `artifacts\trials\validation.jsonl` |
| E04 | LibriSpeech target pool | 部分资源下完成 | 2026-08-22 23:19:40 | 99,278 utterances；921 speakers；仅 clean-360 | `artifacts\metadata\librispeech_target_pool.csv` |
| E05 | 单元测试 | 完成 | 2026-08-23 15:49:54 | 10 passed | 控制台；测试代码见第 10 节 |
| E06 | 无权重模型结构 smoke | 完成 | 2026-08-23 11:14:56 | ECAPA/query/mean 均输出 `[2,24]`；已归一化 | 控制台记录见 6.5 |
| E07-a | GPU 一步训练，初始调试 | 完成、非最终 | 2026-08-22 23:30:27 | loss 18.033920；4.604990 s | `results\runs\gpu_smoke` |
| E07-b | GPU 一步训练，精简 checkpoint | 完成、非最终 | 2026-08-22 23:36:36 | loss 18.474255；3.126842 s | `results\runs\gpu_smoke_v2` |
| E07-c | GPU 一步训练，grouped loader | 完成、当前有效 smoke | 2026-08-22 23:41:29 | loss 17.303108；2.033191 s | `results\runs\gpu_smoke_grouped` |
| E08 | 两条 embedding 提取 | 完成 | 2026-08-22 23:42:05 | shape `[2,192]`；L2 norm 约为 1 | `artifacts\embeddings\gpu_smoke_2.npz` |
| E10 | epoch 内 checkpoint/resume GPU smoke | 完成 | 2026-08-23 15:21:58 | 同一 epoch 从 step 1/batch 1 恢复到 step 2/batch 2 | `results\runs\checkpoint_resume_smoke` |
| E12 | 首次 O-O 正式训练启动 | 主动中止、结果弃用 | 2026-08-23 15:32:29 | 到 batch 267、step 8；发现 validation split 协议问题，无 checkpoint | `results\runs\audio_lazy_p1\process.stderr.log` |
| E13 | StreamVoiceAnon 配套权重准备 | 完成 | 2026-08-23 15:28:42 | 5/5 权重大小与 SHA-256 已审计 | `third_party\StreamVoiceAnon\pretrained_checkpoints` |
| E14 | StreamVoiceAnon CPU smoke | 完成 | 2026-08-23 15:42:12 | 1 条 16 kHz FLAC；重复运行 generated=0/skipped=1 | `results\runs\streamvoice_cpu_smoke` |
| E15 | O-O 正式训练 v2 | 运行中 | 2026-08-23 15:37:16 启动 | trial-capable split；30 epochs；每 10 steps checkpoint | `results\runs\audio_lazy_p1_v2` |
| E16 | Evaluation 匿名化计划 | 完成 | 2026-08-23 15:40:53 | 86,222 utterances；94.1448 h；57,569 refs | `artifacts\anonymization\evaluation_plan.csv` |
| E17 | Semi-informed train 匿名化计划 | 完成 | 2026-08-23 15:44:23 | 7,272 call-sides；7.6451 h；覆盖 5,231 speakers | `artifacts\anonymization\train_one_per_call_side_plan.csv` |
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
- 完成时间：2026-08-23 15:34:07 +08:00。2026-08-22 版本因 validation 未强制 trial capacity 而被替代，旧哈希仅保留在 Git 历史和本总账早期版本中。

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
| validation_min_calls | 2 | 2 |
| trial_max_n | 15 | 15 |
| evaluation_candidates | — | 3,301 |
| validation_candidates | — | 3,301 |
| split disjoint | True | True |

请求总数 7,715，大于本机 Part 1 的 7,066 个 speaker，因此不能生成论文精确规模。这里按论文比例缩放并明确标为本地兼容 split。

输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.csv` | 137,401 | `9dd0c24d47aeb94ca453b35a76d755976d504f86fde0abdcaf284919fb34f0fc` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\speaker_splits.audit.json` | 609 | `fd250974bfd21f27f664c69d2a5dd7994337b25ba241df02eea49985b9653c35` |

### 6.3 E03：Evaluation nested trials

- 目的：为每个合格 evaluation speaker 建立 target/nontarget trial；enrollment 与 target 按 call 分池，避免 call 泄漏；从一次 max-N 抽样中截取 N=5/10/15，满足嵌套关系。
- 配置：`configs\local_fisher_p1.yaml`
- 运行代码：`src\mmsv\cli.py`、`src\mmsv\data\trials.py`
- 完成时间：2026-08-23 15:34:21 +08:00。旧 evaluation trials（1,605 + 1,605）已被 trial-capable split 生成的新版本替代。

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
| eligible_speakers | 1,606 |
| ineligible_speakers | 0 |
| target trials | 1,606 |
| nontarget trials | 1,606 |
| trials 合计 | 3,212 |

输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.jsonl` | 2,592,387 | `e41c457d31be96a1a2f0fa66af67a553e1007a925ea077153fd4994f5e9646d2` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\evaluation.audit.json` | 368 | `3bb6d3babb84558cd3494f6828dcdd08d0ee37802a72c66569bf66b851351e3b` |

### 6.3.1 E11：Validation nested trials

- 第一次诊断运行：旧 split 仅 68/229 speakers 合格，生成 6,800 target + 6,800 nontarget；该结果于发现协议问题后立即废弃。
- 修复：split 阶段同时为 validation/evaluation 预留能构造 call-disjoint `max_n=15` 双侧池的 speakers。
- 最终完成时间：2026-08-23 15:34:12 +08:00。
- 最终结果：229 eligible、0 ineligible、22,900 target、22,900 nontarget，共 45,800 trials。
- 输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\validation.jsonl`，36,966,714 字节，SHA-256 `c706f5b85bf277b8bd316b441145f6e2b67d346e1257bdb5b032020a298a4a3e`。
- 审计：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\validation.audit.json`，369 字节，SHA-256 `8568be7006139bb0d92cdd54f05aca135685bfcdca2ad87bf454c7668b121afb`。

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
| 未选用 subset | train-other-500（不在当前范围内） |

输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\metadata\librispeech_target_pool.csv`，20,264,920 字节，SHA-256 `ff04bd9e77d7702560e75147a44fd2e313f1957cf2e65b78927463062f19bea2`。

### 6.5 E05/E06：自动测试与模型结构 smoke

自动测试命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m pytest -q
```

最近完成时间：2026-08-23 15:49:54 +08:00。结果：`10 passed`。

覆盖文件：`tests\test_fisher.py`、`tests\test_metrics.py`、`tests\test_models.py`、`tests\test_protocol.py`、`tests\test_train.py`、`tests\test_anonymization.py`。

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

### 6.8 E10：epoch 内 checkpoint/resume GPU smoke

- 目的：验证正式长训练能够在 epoch 内从精确 batch 位置恢复，并验证 partial gradient accumulation 的校正逻辑。
- 配置：`configs\gpu_smoke.yaml`，`checkpoint_interval_steps=1`。
- 运行代码：`src\mmsv\train.py`、`src\mmsv\cli.py`。
- 完成时间：2026-08-23 15:21:58 +08:00。

第一段：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli train-audio `
  --config configs/gpu_smoke.yaml `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --splits artifacts/metadata/speaker_splits.csv `
  --output-dir results/runs/checkpoint_resume_smoke `
  --max-steps 1
```

第二段：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli train-audio `
  --config configs/gpu_smoke.yaml `
  --manifest artifacts/metadata/fisher_manifest.csv `
  --splits artifacts/metadata/speaker_splits.csv `
  --output-dir results/runs/checkpoint_resume_smoke `
  --resume results/runs/checkpoint_resume_smoke/last.pt `
  --max-steps 2
```

| 阶段 | epoch | epoch_complete | batch_in_epoch | global_step | 累计平均 loss |
|---|---:|---|---:|---:|---:|
| 第一段结束 | 0 | False | 1 | 1 | 17.30310821533203 |
| 恢复后结束 | 0 | False | 2 | 2 | 16.064202308654785 |

最终 checkpoint 中 `running_loss=32.12840461730957`，证明第二段保留并继续累计第一段的 loss，而不是从新 epoch 重启。自动测试还验证了旧 checkpoint 的兼容恢复和不足 accumulation 时的梯度重缩放。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\checkpoint_resume_smoke\train.jsonl` | 619 | `9990abbca20aeec4dfa433320fa992ac18e0c1bf2c02016e8bb14a90b4592532` |
| `D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\checkpoint_resume_smoke\last.pt` | 115,736,643 | `e7411a1c686a8baf852e1a7111d4d478ea90f859352393d88383dae4aae2dc3a` |

### 6.9 E09：EER 评测状态

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

### 6.10 E12：首次正式训练启动后主动中止

- 启动时间：2026-08-23 15:24:06 +08:00。
- 中止时间：2026-08-23 15:32:29 +08:00。
- 固定代码：commit `9afd8dbac22fbeced36ab86c171271635eadb65e`。
- 进展：epoch 0，batch 267/3751，global step 8，最近累计平均 loss 16.2868。
- 输出日志：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_lazy_p1\process.stderr.log`。
- checkpoint/train.jsonl：均未生成，因为尚未达到 step 10 的首次中途保存点。
- 中止原因：validation trial 审计发现旧 speaker split 只有 68/229 speakers 可构造 call-disjoint N=15 trials。继续训练会使用协议不完整的 split，因此主动终止并重建 split/trials。
- 结果口径：此运行全部结果弃用，不用于后续恢复或 EER。

### 6.11 E13/E14：StreamVoiceAnon 权重与 CPU smoke

5 个官方权重全部就绪：

| 文件 | 完成时间 | 字节数 | SHA-256 |
|---|---|---:|---|
| `dual_ar_delay_0_8.pth` | 2026-08-23 11:46:11 | 598,136,107 | `df703a1a710c807ad0651dd1bbe45556bf5f3a47f1a79929ec3e6e8fecc56583` |
| `campplus_cn_common.bin` | 2026-08-23 15:25:58 | 28,036,335 | `3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8` |
| `asr_s2s_bsq_8192_causal_down_whisper.pth` | 2026-08-23 15:27:08 | 618,561,748 | `dd02fc319d66216159693f6523ebbc4262afd43f41630de70599cf77f99b159e` |
| `firefly-gan-vq-fsq-8x1024-21hz-generator.pth` | 2026-08-23 15:28:31 | 188,518,579 | `01b81dbf753224a156c3fe139b88bf0b9a0f54b11bee864f95e66511c3ccd754` |
| `spark_speaker_encoder.pth` | 2026-08-23 15:28:42 | 56,378,895 | `84adb871ada3c41ac54b8c4897b88c2ed80962937e283437f9a392980ffd3483` |

Hugging Face SDK 因当前代理返回异常 HEAD 元数据而失败；改用官方 `resolve/main` URL、curl 重试和逐文件大小校验后完成。离线 `InferenceWrapper` 在 CPU 上初始化成功，speech tokenizer 与主模型均报告空 missing/unexpected keys。第一次生成因上游 KV cache destination Half、CPU source Float 失败；主工程在不修改 submodule 的情况下为 CPU 重建 FP32 cache。第二次已成功生成 `fe_03_00001_A_0001.flac`：16 kHz、28,236 frames、1.76475 s、mono、FLAC PCM_16，31,688 字节，SHA-256 `be3138de31befed06163d459c0645e783fdec99ea329bfeaa66f88451a2dccd2`。再次运行得到 `generated=0, skipped_existing=1`，断点跳过有效。

### 6.12 E15–E17：正式训练 v2 与匿名化计划

O-O 正式训练 v2 于 2026-08-23 15:37:16 +08:00 启动，PID `78368`，代码 commit `413fa00467daab851cd41c16b2e039e5c0176ea2`，使用新 trial-capable split。运行目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_lazy_p1_v2`。状态为运行中；最终 loss/checkpoint/EER 在产生后继续追加，当前不能填入结果。

首次中途 checkpoint 于 2026-08-23 15:48:40 +08:00 成功原子落盘：epoch 0、`epoch_complete=false`、batch 320/3636、global step 10、running loss 5,177.040755271912、累计平均 loss 16.17825236。`last.pt` 当时为 115,737,923 字节，SHA-256 `89a3ec058c98ea72e6fb2b0142ba545a7c50dc872029295e0a0a9c2ac6aac4a3`。说明：`last.pt` 会被后续 step checkpoint 原子替换，该哈希只标识 step-10 快照时刻。

训练结束后的自动衔接由 `scripts\run_o_o_after_training.ps1` 执行。watcher 必须验证 checkpoint 为 epoch 29 且 `epoch_complete=true`，否则拒绝评测；验证通过后只提取 evaluation trials 引用的 utterances，并依次生成 `artifacts\embeddings\original_evaluation.npz`、`results\o_o\mean_n5.csv`、`mean_n10.csv`、`mean_n15.csv` 及对应 metrics JSON。watcher 自身只生成机器结果，完成后仍需把指标和 SHA-256 追加到本总账。watcher 于 2026-08-23 15:53:50 +08:00 启动，PID `74604`，脚本 commit `5e555dc4de1519eb63cc22ac92d554a796de612e`；日志为 `results\runs\audio_lazy_p1_v2\post_pipeline.stdout.log` 和 `post_pipeline.stderr.log`。

Evaluation 匿名化计划：

| 字段 | 值 |
|---|---:|
| source utterances | 86,222 |
| source hours | 94.1448275 |
| clean-360 reference pool | 99,278 |
| unique references selected | 57,569 |
| mapping | per-utterance deterministic random，seed 1234 |
| plan 文件 | 54,190,866 字节；SHA-256 `8f45abc4c8757767209f123e299bb9819e6a8a0bfa9da1da181c797ce5da1151` |
| audit 文件 | 821 字节；SHA-256 `8bf527db451468b81317b9e95c08e9b32dce22ce99d60044719dd0340c6979fe` |

计划路径：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\evaluation_plan.csv`。

Semi-informed train 匿名化计划：每个 train call-side 用 seed 1234 固定选择一条 source。这是磁盘受限且论文未公开采样细节下的本工程选择；它覆盖全部 5,231 个训练 speakers。

| 字段 | 值 |
|---|---:|
| source utterances / call-sides | 7,272 |
| covered train speakers | 5,231 |
| source hours | 7.64506944 |
| unique references selected | 7,034 |
| plan 文件 | 4,519,696 字节；SHA-256 `5a164a2b65da93203a32024712b02395fa7f56d521f32b6da5fa3cefc216f218` |
| audit 文件 | 912 字节；SHA-256 `6fa4c43a92e7822effb550b3a6c56e502a1bdd4742fbc17ec31984a8c31edde0` |

计划路径：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\train_one_per_call_side_plan.csv`。

## 7. 失败尝试、工程修复与保留证据

| 事件 | 现象 | 处理 | 当前状态 |
|---|---|---|---|
| 论文精确 split 尝试 | `configs\paper.yaml` 请求 7,715 speakers，而 Part 1 只有 7,066 | `configs\paper.yaml` 仅保留作参考；全部正式实验固定用 `configs\local_fisher_p1.yaml` | 已按用户指定范围关闭，不再等待 Part 2 |
| Hugging Face 镜像 | 原环境的 `hf-mirror.com` 不可达 | 配置固定为 `https://huggingface.co` | 已解决，WavLM-Large 已缓存 |
| Transformers 后台 safetensors 转换 | 首次常规加载出现等待/挂起 | 先用 `snapshot_download` 完成快照，再从本地缓存加载 | 已解决 |
| 初始 checkpoint 过大 | E07-a 产物约 1.38 GB | 冻结 WavLM 不再重复写入 checkpoint | E07-b/c 降至约 115.7 MB |
| 随机逐 turn loader | 约 296,300 batches/epoch，Windows SPHERE 解码效率不可接受 | 改为 call-side grouped loader；同 call A/B 相邻并复用解码缓存 | 当前约 3,751 batches/epoch；E07-c 已验证 |
| epoch 内恢复与尾部梯度 | 旧实现只在 epoch 末保存，且尾部不足 32 batches 的梯度会跨 epoch 残留 | 每 10 optimizer steps 保存 batch/running-loss 状态；尾部梯度按实际累计数校正后更新 | E10 两段式 GPU 恢复与自动测试已通过 |
| validation speaker 容量 | 旧 split 随机抽 validation，只有 68/229 可构造双侧 N=15 trials | split 阶段联合预留 trial-capable validation/evaluation speakers | val 229/229、eval 1606/1606 合格；旧正式训练已主动中止 |
| StreamVoiceAnon CPU 生成 | 官方 KV cache 固定 FP16，CPU source tensor 为 FP32 | 主工程为 CPU 重建 FP32 cache；不修改 submodule；改用 SoundFile 写 FLAC，避免 TorchCodec 依赖 | 1 条端到端输出成功；重复运行跳过成功 |
| 匿名化与 semi-informed | checkpoint 已就绪，但尚未生成匿名 Fisher Part 1 | 固定使用 clean-360 target pool，下一阶段运行并审计匿名化流水线 | 未完成 |

说明：`results\runs\gpu_smoke` 和 `results\runs\gpu_smoke_v2` 是为保留调试证据而存在的非最终目录；后续正式实验不得从它们继续训练。当前 smoke 基线是 `results\runs\gpu_smoke_grouped`。

## 8. 论文目标值（仅供验收，不是本机结果）

论文 O-O Table II 的 EER% 目标：

| 聚合 | N=5 | N=10 | N=15 |
|---|---:|---:|---:|
| Mean | 3.87 | 3.29 | 3.09 |
| Query | 3.39 | 2.54 | 2.27 |
| Frame Concat | 3.26 | 2.33 | 2.10 |

论文 A-A semi-informed Table I 中 Frame Concat 的 EER% 目标为 15.10 / 8.83 / 6.96（N=5/10/15）。这些数值只用于观察趋势和差距；当前项目的正式验收是在 Part 1 + clean-360 范围内生成完整、可重复、可审计的同结构结果表。

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
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\validation.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\trials\validation.audit.json
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_grouped\last.pt
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\embeddings\gpu_smoke_2.npz
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\evaluation_plan.csv
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\evaluation_plan.audit.json
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\train_one_per_call_side_plan.csv
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymization\train_one_per_call_side_plan.audit.json
```

保留的非最终调试产物：

```text
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke\last.pt
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\gpu_smoke_v2\last.pt
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\checkpoint_resume_smoke\train.jsonl
D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\checkpoint_resume_smoke\last.pt
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
| `src\mmsv\anonymization.py` | 确定性匿名化计划、StreamVoiceAnon runner、FLAC/manifest/progress |
| `scripts\run_o_o_after_training.ps1` | 等待完整训练、校验最终 checkpoint、自动提取 embeddings 与 Mean O-O EER |

本轮源文件 SHA-256：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `pyproject.toml` | 808 | `ad200b4473eded7e4c93c9ed9c7849e3131284f1e1d6e7d9e4ac7fb52166aa6c` |
| `configs\paper.yaml` | 1,369 | `af1a73cc7c648f8c8083d316cd3f0f83cd1c534910594c79caf4fc2413641aa4` |
| `configs\local_fisher_p1.yaml` | 1,572 | `8a914cacf6f31231e68a5f3c7724c66ed4125c02b642ea65da6deaacd1b3fae9` |
| `configs\gpu_smoke.yaml` | 513 | `2047b85139bba5f4482bdc57421621bb7981a3e5b3269b9bd80f03fc1152155b` |
| `configs\semi_local.yaml` | 500 | `cdecb1389ecb1f0ee8d07cf069f10dfca07a3677efaa84508ba25706a8f48ae7` |
| `src\mmsv\__init__.py` | 89 | `d1f4a61fa0b2491a3fc25e5975d11b7e92a71edb35a682ef46c84bdb770e1d82` |
| `src\mmsv\config.py` | 803 | `d175a20a350ffe3c3bb3c19b7c01a48973c4fd622bee62cdb8b1f6d83019bf4b` |
| `src\mmsv\data\fisher.py` | 6,122 | `9e07adc7be42d10ed8a41405f9d71d1d8ccd4b3f5197ccfe239fdf998ab10ccb` |
| `src\mmsv\data\splits.py` | 5,753 | `3055682d5bb9892937a223f9757fe3e1b7b2824201e039a46fc32f0be3d69958` |
| `src\mmsv\data\trials.py` | 7,310 | `39b93bf9e2bade20bfb16d2fe8bc5017e3ec91d716c11642c1957eb300876976` |
| `src\mmsv\data\librispeech.py` | 1,909 | `a97ed1181e346251514db84dda746afd6c3c9b82a09096cac850114b32dc0397` |
| `src\mmsv\audio.py` | 4,482 | `b6976bd512e5c190ac89b5d34b7c98ac7a3e4b8e38fd4ebe208544edd40c4493` |
| `src\mmsv\aggregation.py` | 2,128 | `c17fe1ac00b0813b368444e32d63451c4112fcd4db2fc1d453e6c58a715c9b64` |
| `src\mmsv\models.py` | 10,475 | `a7c9a589c9edc28d861ce8494eed6e0a41559cbf57ec71c5106bd25278d2ea43` |
| `src\mmsv\metrics.py` | 4,593 | `0c6da0d5ef0f2a2a22e407c6d05e8dfbe715f20240f81fda0db69698468680e0` |
| `src\mmsv\train.py` | 16,644 | `fe12190116e37bbb7a0e726451869c777667b4ea4018c8ea9a76c49e18014da2` |
| `src\mmsv\multimodal.py` | 4,247 | `9d169e24eb85c8a2f8d05c5603cda48de5a68857850d4e52849324a1dc09549f` |
| `src\mmsv\cli.py` | 10,214 | `ef88311048750eab01d87f2faf020e349685c0dafecf77931dbe17c99d928f8d` |
| `src\mmsv\anonymization.py` | 11,426 | `466f2460c2823e218a1ff02cd4ebf10778b95a541f50ebe189762393cd5e73ca` |
| `tests\test_fisher.py` | 923 | `16e0573a4937dca2ab4c5e307130aad1ba1571a63979c2177258c62fe5c66d12` |
| `tests\test_metrics.py` | 321 | `fd8ee0ef0db2dd4fbd43bbdbed73b77d660e24dfd41ae518a3641149ab01e131` |
| `tests\test_models.py` | 1,360 | `aae35a4b447286e8e2893c5515c861d47df1e35f034a42d28cbc01efce892540` |
| `tests\test_protocol.py` | 2,536 | `13d5a3de3bbe58a4e3c4490b3fbf598443e463d8a8a213194e3582d608020184` |
| `tests\test_train.py` | 782 | `136e9599bea2ff87cffc36f582759c341dd4dc5181dd225bfbb6d6e7491a1c8d` |
| `tests\test_anonymization.py` | 4,054 | `b19d097076fb542449c3db09130d28d56d2433795daf283a0e21b64bfd9481d1` |
| `scripts\run_o_o_after_training.ps1` | 2,352 | `209f9efdcfe9a0688e31bf6779ccce58d0e3db96dcf52134184f2e3a54e067c1` |

## 11. 下一阶段与验收条件

1. 始终使用 `configs\local_fisher_p1.yaml`、现有 Part 1 manifest/split/trials 和 clean-360 target pool，不运行 `configs\paper.yaml`，不等待额外数据。
2. 在固定范围内执行 30 epoch lazy-informed 训练，输出 original embeddings 和 O-O EER。
3. 使用现有 `dual_ar_delay_0_8.pth` 与 clean-360 target pool 生成匿名 Fisher Part 1，记录匿名音频、manifest 和审计信息。
4. 执行 15 epoch semi-informed 训练；必须用 `--init-from` 重置 optimizer，而不是普通 `--resume`。
5. 对 Mean、Query、Frame Concat 在 N=5/10/15、O-O/O-A/A-A 条件下输出每个 trial 的 score CSV 和汇总 EER。
6. 实现并评测 Whisper、LUAR、prosody、RJCA 等 Level 2–3 多模态组合。
7. 作者发布代码、split、trials 或新权重后只做差异审计，不改变当前数据范围，除非用户另行明确授权。

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
- 远程仓库：`origin = git@github.com:wwwYYYcom/multimodal-SV.git`；当前 `main` 已跟踪 `origin/main`。

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

## 14. 数据范围变更记录

- 决策时间：2026-08-23 15:09:33 +08:00
- 决策来源：用户在对话中明确指定。
- Fisher：只使用 Part 1；采用已经审计的 5,850 calls、929,364 utterances、7,066 speakers，以及 5,231/229/1,606 本地 speaker split。
- LibriSpeech：只使用 `train-clean-360`；采用已经审计的 99,278 条严格大于 4 秒 utterances、921 speakers。
- 排除数据：Fisher Part 2、LibriSpeech `train-other-500`。后续实验不得静默加入这些数据。
- 默认配置：`configs\local_fisher_p1.yaml`。`configs\paper.yaml` 仅用于保存论文规模参考，不作为运行配置。
- 报告名称：所有后续指标统一标注为“Fisher Part 1 + LibriSpeech train-clean-360 范围复现结果”。
- 验收标准：优先保证固定 seed、固定 split/trials、完整日志、输出路径和可重复运行；论文表格仅作参考对照，不要求数值完全相同。
- 范围验证完成时间：2026-08-23 15:12:54 +08:00。配置断言确认只有 1 个 LibriSpeech root 且以 `train-clean-360` 结尾，Fisher 路径指向 LDC2004S13/T19 Part 1；manifest、split、target pool 与 StreamVoiceAnon checkpoint 均存在；当时自动测试结果为 `5 passed`，最近完整测试已增至 `10 passed`。

## 15. GPU 利用率诊断与训练供数优化

### E18：并行 DataLoader 诊断与断点基准

- 状态：完成；正式训练已从验证后的原子 checkpoint 继续推进。
- 诊断时间：2026-08-23 22:22:33 至 22:23:17 +08:00。
- 旧训练进程：PID `78368`，开始于 2026-08-23 15:37:16 +08:00；安全停止于 step 390 检查点之后。
- 连续 GPU 采样：30 秒内峰值 `95%`、最低 `0%`、平均约 `10.3%`；显存稳定为约 `3,081 MiB`。同一时段磁盘占用约 `1%–5%`，CPU 总占用约 `14%–30%`。
- 根因：`num_workers=0` 使 GPU 等待 CPU；无 `sph2pipe` 时 `desphere` 必须先解码整通 Fisher Shorten/SPHERE，再截取一个短 turn。batch size 2 又使 GPU 计算脉冲较短。
- step 390 checkpoint：`epoch=3`、`epoch_complete=false`、`batch_in_epoch=1536/3636`、`global_step=390`、`running_loss=22904.411051750183`。
- 并行设置：`num_workers=2`、`prefetch_factor=2`、`pin_memory=true`；未使用 4 workers，因为当时系统可用物理内存仅约 4 GB。
- 基准命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -u -m mmsv.cli train-audio --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/loader_prefetch_smoke --resume results/runs/audio_lazy_p1_v2/last.pt --max-steps 391
```

- 基准数据：worker 初次启动约 10 秒；随后 31 个 batch 约 23 秒，稳态约 `0.74 s/batch`，旧日志约 `1.8 s/batch`，局部吞吐约 `2.4×`。正式恢复后的含波动均值约 `1.06 s/batch`，相对旧训练约 `1.7×`。
- 优化后 30 秒 GPU 采样：平均 `15.17%`、峰值 `96%`；GPU 仍受整通 SPHERE 解码限制，因此继续执行 E19。
- 基准输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\loader_prefetch_smoke\last.pt` 与 `train.jsonl`。
- 代码提交：`f3f0782d85c5d47651b4ff1e7098791205ca73bd`（`perf: prefetch Fisher training batches`）。

### E19：Fisher 确定性训练片段缓存

- 状态：完成；缓存完成后已自动恢复并完成旧版正式训练。
- step 400 切换时间：2026-08-23 22:40:06 +08:00。
- 正式断点：`epoch=3`、`epoch_complete=false`、`batch_in_epoch=1856/3636`、`global_step=400`、`running_loss=27661.14385318756`。
- 缓存开始时间：2026-08-23 22:40:26 +08:00。
- 缓存监督进程：PID `38272`。
- 目的：只缓存固定 seed 1234 下 30 epoch 会抽到的 Fisher Part 1 utterance，保存为 16 kHz FLAC/PCM16；不复制整通电话，不引入 Fisher Part 2 或其他数据集。
- 缓存输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\cache\fisher_train_selected_30e`；完整后生成 `audit.json`，其中记录完成时间、耗时、unique utterance 数、生成/跳过数、manifest/split SHA-256。
- 运行命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -u scripts/build_fisher_training_cache.py --config configs/local_fisher_p1.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir artifacts/cache/fisher_train_selected_30e
```

- 自动恢复脚本：`scripts\build_cache_then_resume.ps1`。它仅在 `audit.json` 完整性校验通过后启动训练，并重新启动 `scripts\run_o_o_after_training.ps1`。
- 当前日志：`results\runs\audio_lazy_p1_v2\cache_supervisor.stdout.log`、`cache_supervisor.stderr.log`；恢复后日志为 `process_cached.stdout.log`、`process_cached.stderr.log` 和 `post_pipeline_cached.*.log`。
- 缓存进行中快照（2026-08-23 22:41:08 +08:00）：701 个 FLAC、31,627,661 字节；这是进度值，不是最终审计值。
- 自动测试：`11 passed`。
- 代码提交：`39ed207b80371e4b28de8b11a9ee9f19d537e1cb`（`perf: cache selected Fisher training segments`）。
- 关键代码指纹：`src\mmsv\train.py` 18,598 字节，SHA-256 `fd4dd476dfc14511eb400326bc7d4ca1fd95839789523b1fc21dfdef8b02b2db`；`scripts\build_fisher_training_cache.py` 3,580 字节，SHA-256 `6f056136cc040bb7f75ebe38dd2f3fe3d5039baae34714504218f46b5e785384`；`scripts\build_cache_then_resume.ps1` 2,631 字节，SHA-256 `c50a29e51818af2a34ac8ee470fa04f46f8dacfb008fe512c39b4aee942c8b27`；配置 SHA-256 `0c09778b24037b18e0d6f84d8f1abcd797b1e8e653350251e8851b6d17f965d3`。

最终缓存审计：2026-08-24 01:19:30 +08:00 完成，耗时 9,541.38 秒；180,311 个 FLAC、8,069,502,411 字节（7.515 GiB）。`audit.json` 798 字节，SHA-256 `b8fbfee1865c212aa32017b6d3f76c21ce9f90b52dc6400abf6cd97c65ee4918`。

## 16. E20：旧版 O-O Mean 结果与失效诊断

- 状态：完成，但判定为失效训练基线；保留全部产物，不作为论文数值复现结论。
- 训练完成时间：2026-08-24 04:18:41 +08:00。
- 后处理完成时间：2026-08-24 12:30:53 +08:00。
- 最终 checkpoint：epoch 29 完整、batch 3636/3636、global step 3420、loss `13.491323512510629`、learning rate `7.8125e-06`。
- checkpoint：`results\runs\audio_lazy_p1_v2\last.pt`，115,738,115 字节，SHA-256 `f8056f101eeed10aca73e57d087994be6834be0a3103d1195bfce57d62da9c61`。
- 训练日志：`results\runs\audio_lazy_p1_v2\train.jsonl`，9,324 字节，SHA-256 `3ba6730037705765535fff74d7db286d7d6353d0f073356901af2379a3f74a9d`。
- evaluation embeddings：86,222 条、192 维；`artifacts\embeddings\original_evaluation.npz`，62,468,340 字节，SHA-256 `71cc51e1dc6ef956a3455243d8f97c1df31271077242641176eccbad657d96d7`。

| N | target trials | non-target trials | EER | threshold | 论文 Mean O-O |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,606 | 1,606 | 46.513076% | 0.495667 | 未报告（本地扩展） |
| 5 | 1,606 | 1,606 | 38.480697% | 0.727445 | 3.87% |
| 10 | 1,606 | 1,606 | 34.869240% | 0.780447 | 3.29% |
| 15 | 1,606 | 1,606 | 30.946451% | 0.813190 | 3.09% |

结果文件：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `results\o_o\mean_n1.csv` | 151,467 | `dba46626b97965cd6757758f06818ce3a156ffc865f6614e9bdc903d7450094b` |
| `results\o_o\mean_n1.metrics.json` | 289 | `c9e60f12c26ba1cc9d601b70736e3785cd327f85fc985c52c2b54eb756a3ce16` |
| `results\o_o\mean_n5.csv` | 150,790 | `73cc8ff555f14f72122cce04fc0018e6ceba0510553f15fba47638371e45c68e` |
| `results\o_o\mean_n5.metrics.json` | 289 | `c888f164b87d387321b0f8757b25cc5395dc95d52175452c1a4dfa9056716557` |
| `results\o_o\mean_n10.csv` | 150,457 | `e42855164ba6fbce0869b2b122b0af320d0fde1dd9da06667b2995e590c16405` |
| `results\o_o\mean_n10.metrics.json` | 289 | `924d36755d41ca1e415def4a224bd0e8cb942c25ed9f04a60f54eaf1c6d0c340` |
| `results\o_o\mean_n15.csv` | 150,284 | `a934d2bffe48f6dfa6b451a4a246cbf4bb4a9af2c218eef67bf62195ef17efd6` |
| `results\o_o\mean_n15.metrics.json` | 291 | `df575d2b2aaf0b4f052862fd94c8b749f85e8f535d116ffd862d331568e82cdd` |

失效诊断（2026-08-24 14:55 至 15:14 +08:00）：

- EER 标签、trial speaker/call 一致性和 target/non-target 方向均正确；N 增加时 EER 单调下降，排除标签反转。
- 192 维 evaluation embedding 的第一主成分占 73.36% 方差，前 10 主成分占 99.59%，参与率有效秩仅 1.81，存在严重表示塌缩。
- epoch 29 的 256 条训练样本闭集诊断：5,231 类 cosine top-1 仅 0.390625%，AAM loss 14.2644，模型未充分学成。
- 本地 train split 有 572,951 utterances，但 call-side 采样每 epoch 只呈现 7,272 条；30 epochs 仅 3,420 optimizer steps。按物理 batch 64 遍历全部本地 turns 的同口径估算为 268,560 steps，旧版仅为 1.27%。
- 梯度累积 `2×32` 不等于论文物理 batch 64：ECAPA 的 BatchNorm（尤其 ASP 后 3,072 维 BN）每次仍只观察 2 条样本；固定 A/B call-side 配对且不 shuffle 会进一步污染批统计。
- 66.66% 的缓存训练 utterance 小于 4 秒，旧版使用零填充且未向 WavLM/ASP 传 attention mask。
- `train-clean-360` 只参与匿名化 target pool，不参与 O-O，因此不是本次 O-O 巨大差距的原因。
- 作者官方仓库 commit `9384c1b610a1261bdf5d7346c63d227095ab411f` 当前仍只有“Code and pretrained models will be released soon”，精确 WavLM 层融合与 ECAPA 细节仍不可核对。

修正决策：保留 E20 全部产物；新建 corrected 配置和运行目录，使用全量随机 utterance、ECAPA/AAM 物理 batch 64、冻结 WavLM 小块前向、短语音循环补齐、每 epoch 独立 checkpoint。不得从 E20 的塌缩 checkpoint 初始化 corrected 模型。

## 17. Corrected 全量训练实现与启动记录

### E21：物理 batch 64 GPU smoke

- 状态：完成。
- 完成时间：2026-08-24 15:23:00 +08:00。
- 目的：确认冻结 WavLM 可分块前向，同时 ECAPA、ASP BatchNorm 和 AAM 一次接收真实 64 utterance batch；确认 AMP、反向传播、optimizer、scaler 和 checkpoint 均可用。
- 数据：epoch 0 固定 shuffle 顺序的首个 64-utterance batch；42 条临时解码、22 条从旧缓存硬链接复用。临时缓存审计路径：`results\runs\corrected_physical_batch_smoke\cache\audit.json`。
- 正式采用参数：physical batch 64、gradient accumulation 1、WavLM feature micro-batch 32、AMP FP16、短语音循环填充、全量 utterance 模式。
- 结果：global step 1、loss `16.812938690185547`、核心 step 时间 `1.7371` 秒；`backend.asp_bn.num_batches_tracked=1`，AMP scale `65536`。
- GPU 监控：峰值利用率 99%、峰值显存 5,930 MiB、峰值功耗 95.56 W；RTX 5060 8 GiB 无 OOM，并保留约 2.2 GiB 余量。
- 输出目录：`results\runs\corrected_physical_batch_smoke_v4`。
- checkpoint：115,737,155 字节，SHA-256 `7c1962178ebd3b9fd0f1842590a3fdd014507a0d690dbdb80749fb0b48cc5ed7`。
- train log：321 字节，SHA-256 `e9517747d3a14d2184430073384f94480fa254747330b3ab566b751760ae61d3`。
- 运行命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -u -m mmsv.cli train-audio --config configs/gpu_smoke_corrected.yaml --manifest artifacts/metadata/fisher_manifest.csv --splits artifacts/metadata/speaker_splits.csv --output-dir results/runs/corrected_physical_batch_smoke_v4 --max-steps 1
```

### E22：Corrected 全量缓存与正式训练监督流程

- 状态：进行中；当前阶段为全量 Fisher Part 1 train utterance 缓存，审计通过后自动开始随机初始化训练。
- 启动时间：2026-08-24 15:26:18 +08:00。
- supervisor PID：`79252`；cache builder PID：`59628`。
- 训练代码 commit：`e0ce0bff6df8a597faa993386e65518e9f0d1d70`（`fix: train ECAPA with full physical batches`）。
- 数据范围：仅 Fisher Part 1 train split 的 572,951 utterances；LibriSpeech 仍只保留 `train-clean-360`，且不参与 O-O 训练。
- 全量缓存目录：`artifacts\cache\fisher_train_all_p1`。将硬链接复用旧缓存中已有的 180,311 条，缺失条目从 Part 1 SPHERE 解码；不删除旧缓存。
- 正式训练目录：`results\runs\audio_corrected_p1`；从随机初始化开始，不加载 E20 checkpoint。
- 训练规模：8,952 full batches/epoch、30 epochs；理论 total optimizer steps 为 268,560，末 batch 余数因 `drop_last=true` 每 epoch 丢弃 23 条。
- 每 100 steps 原子保存 `last.pt`，并保留 `epoch_00.pt` 至 `epoch_29.pt`，供后续 validation 选择最佳 checkpoint。
- corrected embedding 输出：`artifacts\embeddings\original_evaluation_corrected.npz`；corrected Mean O-O 输出：`results\o_o_corrected`，不会覆盖 E20 失效基线。
- supervisor 日志：`results\runs\audio_corrected_p1\supervisor.stdout.log`、`supervisor.stderr.log`；训练与后处理日志使用同目录下 `process.*.log` 与 `post_pipeline.*.log`。
- 完整启动脚本：`scripts\build_full_cache_then_train_corrected.ps1`。
- 自动测试：`15 passed`。

关键运行代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `configs\local_fisher_p1_corrected.yaml` | 2,037 | `0babedd4ba4124310c2f58e6034615e258710ee7f27fc13106dc3f49882c173f` |
| `configs\gpu_smoke_corrected.yaml` | 706 | `e5763c2c6348b472cca82afc0482f74487b27fb2dc70ed74b5f2b4f027165ffc` |
| `src\mmsv\train.py` | 23,371 | `525b3c531da371bbe061ce88f85e5c3e26ff3357be07d858c7c3a14275d9df8b` |
| `src\mmsv\audio.py` | 5,124 | `1ef9f9e17eee6bf4d53c75885cf6f1851585bc79379daab32f46417a8146b2c8` |
| `scripts\build_fisher_full_training_cache.py` | 4,361 | `9fb77b4389060c2634b849db3ae471f08f9f639087c935ffa03efbd75976b49d` |
| `scripts\build_full_cache_then_train_corrected.ps1` | 2,937 | `d3c91f5c5db66b37ed88d5bbd5dda7b1f4cc8e377b90d1cdf824b7d63d0c75e1` |
| `scripts\run_o_o_after_training.ps1` | 1,970 | `04907e281463eebc01af7b9f1eace4d2ee5d3792a8b96df5de63cc8e7b67d4e1` |

## 18. Fisher 训练缓存目录整合

- 状态：完成。
- 安排时间：2026-08-24 17:39:05 +08:00。
- 完成时间：2026-08-24 19:04:31 +08:00。
- 目的：最终只保留 `artifacts\cache\fisher_train_all_p1`，消除旧目录 `artifacts\cache\fisher_train_selected_30e`，同时保留旧缓存审计信息。
- 重合核验：抽查时全量目录与旧目录已有 106,881 个同名 FLAC；106,881 个全部具有相同的设备号和文件索引号，即为同一物理文件的 NTFS 硬链接，不是复制件。对应逻辑大小为 4,587,644,389 字节，因此当前没有重复占用这部分物理磁盘空间。
- 旧缓存总量：180,311 个 FLAC。不能在全量构建过程中提前删除，因为构建器仍以旧目录作为硬链接复用源。
- 安全条件：全量目录必须存在 `audit.json`，且其中 `train_utterances=572951`、`target_utterances=572951`；随后必须逐条确认旧目录全部 180,311 个 FLAC 在全量目录中存在并指向同一物理文件。任一条件不满足即报错并保留旧目录。
- 完成动作：把旧目录的 `audit.json` 保存为 `artifacts\cache\fisher_train_all_p1\selected_30e.audit.json`，然后删除旧目录名。音频仍完整保存在全量目录，删除的只是多余硬链接名称。
- 缓存进度快照：2026-08-24 17:39:30 +08:00，全量目录已有 354,905 / 572,951 个 FLAC；cache builder PID `59628`、训练 supervisor PID `79252` 均正常。
- 整合 watcher PID：`84704`；启动时已输出 `waiting_for_builder=true`。
- 标准输出：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\cache_consolidation.stdout.log`。
- 标准错误：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\cache_consolidation.stderr.log`。
- 运行代码 commit：`57c0566`（`feat: consolidate Fisher caches safely`）。
- 自动测试：固定 `pytorch` 环境下 `17 passed`；PowerShell AST 解析、Python 编译与 `git diff --check` 均通过。

运行命令：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\consolidate_cache_after_build.ps1 -BuilderPid 59628
```

本次新增代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `scripts\verify_cache_hardlinks.py` | 2,193 | `a7cf4eb43f45129714fe662129f257b4a57ca9b889cad0f072bde7b3f5c32383` |
| `scripts\consolidate_cache_after_build.ps1` | 3,233 | `338ea9c8ebe82e1186fda75934ffab74932fa8fa75f1a34c09dbb12010b9ce22` |
| `tests\test_cache_verification.py` | 1,119 | `db21638a0b18ce456045111b9e47b9172d089839823f28ff6885c5789ea36544` |

最终结果：全量缓存于 2026-08-24 19:03:02 +08:00 完成，共 572,951 个 FLAC、25,246,096,582 逻辑字节；其中新解码 392,640 条、从旧缓存硬链接 180,311 条。整合验证结果为 `valid=true`、`verified_hardlinks=180311`、缺失 0、不同物理文件 0。旧目录已删除，只保留 `artifacts\cache\fisher_train_all_p1`。全量 `audit.json` 为 961 字节，SHA-256 `58bb63fdb4fe9cb22eb5a694785805e03de7eeb685ea0be190ba1e06b7cd4a24`；保留的 `selected_30e.audit.json` 为 798 字节，SHA-256 `b8fbfee1865c212aa32017b6d3f76c21ce9f90b52dc6400abf6cd97c65ee4918`。

## 19. E22 corrected 正式训练进度快照

- 状态：训练正常进行中；尚未开始 corrected O-O 后处理。
- 检查时间：2026-08-25 17:52:38 +08:00。
- 训练进程：PID `71796`，启动于 2026-08-24 19:03:05 +08:00；O-O watcher PID `83540` 正常等待训练结束。
- 当前阶段：epoch 索引 11，即第 12/30 个 epoch；实时日志为 701/8,952 batch、global step 99,173/268,560，总 optimizer-step 进度约 36.93%。
- 最近原子 checkpoint：`last.pt` 记录 epoch 11、batch 628、global step 99,100、`epoch_complete=false`；实时进度比 checkpoint 超前 73 steps，若异常中断最多重跑该区间。
- 已完整完成 epoch 0 至 10；对应 checkpoint `epoch_00.pt` 至 `epoch_10.pt` 均存在，每个约 115.74 MB。
- 完整 epoch loss：7.289619、3.668416、2.918002、2.489858、2.180034、1.630999、1.429098、1.287293、1.162275、1.051252、0.775268。当前第 12 个 epoch 的实时累计平均 loss 约 0.6098，下降趋势正常。
- 当前吞吐：约 1.03 batch/s；完整 epoch 实测约 2.03 至 2.14 小时。按当前速度估算剩余约 39 小时，预计 2026-08-27 09:00 +08:00 前后完成训练，之后自动进行 embedding 提取和 Mean O-O N=5/10/15 评估；该时间是动态估计，不是完成记录。
- GPU 快照：利用率 97%，显存 6,185/8,151 MiB，功耗 81.03 W；GPU 已处于高负载状态。
- 磁盘快照：D 盘剩余 25.07 GiB；当前 checkpoint 增长规模可控。
- 训练日志：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\process.stderr.log`。
- epoch 汇总：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\train.jsonl`。
- checkpoint：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\last.pt` 及同目录 `epoch_00.pt` 至最终 `epoch_29.pt`。
- 后处理日志：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\post_pipeline.stdout.log`、`post_pipeline.stderr.log`；当前内容为 `waiting_for_training=true`。

### 2026-08-25 20:29 训练进度更新

- 状态：PID `71796` 正常运行；PID `83540` 正常等待训练完成，后处理尚未启动。
- 已完整完成 epoch 0 至 11；`epoch_11.pt` 于 2026-08-25 19:47:42 +08:00 写入。epoch 11 loss 为 `0.6868679094410454`，耗时 `7537.209326267242` 秒，global step `107424`。
- 当前为 epoch 索引 12，即第 13/30 个 epoch。实时日志约为 2,961/8,952 batch、global step 110,385；总训练进度约 41.10%，实时累计平均 loss 约 `0.5734`。
- 最近原子 checkpoint 于 2026-08-25 20:29:47 +08:00 写入：epoch 12、batch 2,976、global step 110,400、`epoch_complete=false`、`running_loss=1706.709677129984`，文件大小 115,738,499 字节。
- GPU 快照：利用率 99%，显存 6,316/8,151 MiB，功耗 83.52 W。
- D 盘剩余 24.93 GiB。
- 按最近完整 epoch 的实际速度估计尚需约 36.5 小时，训练预计仍在 2026-08-27 09:00 +08:00 前后完成；完成后 watcher 自动开始 corrected embedding 和 Mean O-O N=5/10/15 评估。

### 2026-08-26 09:06 训练进度更新

- 状态：训练 PID `71796` 正常运行；后处理 watcher PID `83540` 正常等待，尚未生成 corrected embeddings 或 O-O 结果。
- 已完整完成 epoch 0 至 17，即 18/30 个 epoch；最新完整 checkpoint `epoch_17.pt` 于 2026-08-26 08:10:31 +08:00 写入。
- epoch 12 至 17 的完整 loss 依次为 `0.6263739847`、`0.5731074184`、`0.5280308967`、`0.4085137895`、`0.3763672326`、`0.3541229078`，下降趋势正常。
- 当前为 epoch 索引 18，即第 19/30 个 epoch；实时日志约为 4,071/8,952 batch、global step 165,207/268,560，总 optimizer-step 进度约 61.52%，实时累计平均 loss 约 `0.3183`。
- 最近原子 checkpoint：epoch 18、batch 4,064、global step 165,200、`epoch_complete=false`、`running_loss=1293.007419064641`；于 2026-08-26 09:06:00 +08:00 写入。
- GPU 快照：利用率 96%，显存 6,432/8,151 MiB，功耗 86.00 W，SM 时钟 2,542 MHz。
- D 盘剩余 52.02 GiB，足够保存剩余 epoch checkpoints 和后处理结果。
- 按最近 epoch 实测速度估计尚需约 23.6 小时，预计训练于 2026-08-27 08:45 +08:00 前后完成，随后自动开始 corrected embedding 提取及 Mean O-O N=5/10/15 评估；该时间仍为动态估计。

### 2026-08-26 14:50 训练进度更新

- 状态：训练 PID `71796` 与后处理 watcher PID `83540` 均正常；corrected embeddings 和 O-O 结果尚未生成。
- 已完整完成 epoch 0 至 20，即 21/30 个 epoch；最新完整 checkpoint `epoch_20.pt` 于 2026-08-26 14:27:06 +08:00 写入。
- epoch 18 至 20 的完整 loss 依次为 `0.3363849453`、`0.3182027325`、`0.2719619149`；epoch 20 完成后的 learning rate 为 `3.125e-05`。
- 当前为 epoch 索引 21，即第 22/30 个 epoch；实时日志约为 1,660/8,952 batch、global step 189,652/268,560，总 optimizer-step 进度约 70.62%，实时累计平均 loss 约 `0.2471`。
- 最近原子 checkpoint：epoch 21、batch 1,608、global step 189,600、`epoch_complete=false`、`running_loss=397.15735380351543`；于 2026-08-26 14:49:37 +08:00 写入。
- GPU 快照：利用率 99%，显存 6,705/8,151 MiB，功耗 79.85 W，SM 时钟 2,400 MHz。
- D 盘剩余 51.69 GiB。
- 按最近 epoch 实测速度估计尚需约 18.3 小时，预计训练于 2026-08-27 09:10 +08:00 前后完成；随后 watcher 自动执行 corrected embedding 提取和 Mean O-O N=5/10/15 评估。

### 2026-08-26 22:42 训练进度更新

- 状态：训练 PID `71796` 与后处理 watcher PID `83540` 均正常；corrected embeddings 和 O-O 结果尚未生成。
- 已完整完成 epoch 0 至 23，即 24/30 个 epoch；最新完整 checkpoint `epoch_23.pt` 于 2026-08-26 20:44:23 +08:00 写入。
- epoch 21 至 23 的完整 loss 依次为 `0.2597133081`、`0.2524261782`、`0.2455864026`，下降趋势稳定。
- 当前为 epoch 索引 24，即第 25/30 个 epoch；实时日志约为 8,268/8,952 batch、global step 223,116/268,560，总 optimizer-step 进度约 83.08%，实时累计平均 loss 约 `0.2372`。
- 最近原子 checkpoint：epoch 24、batch 8,252、global step 223,100、`epoch_complete=false`、`running_loss=1957.3072680011392`；于 2026-08-26 22:41:44 +08:00 写入。
- GPU 快照：利用率 100%，显存 6,623/8,151 MiB，功耗 82.81 W。
- D 盘剩余 51.27 GiB。
- 当前 epoch 预计约十分钟后完成；按最近 epoch 实测速度，完整训练仍预计于 2026-08-27 09:00 +08:00 前后结束，随后自动进行 corrected embedding 提取及 Mean O-O N=5/10/15 评估。

## 20. E23：增加 N=1 单语句评估

- 状态：实现与自动后处理接入完成；corrected N=1 指标等待 E22 训练完成后自动生成。
- 用户决策时间：2026-08-26 22:42 +08:00；实现完成时间：2026-08-26 23:11:27 +08:00。
- 定义：N=1 使用现有 max-N=15 trial 中 enrollment 和 target 各自的第一个 utterance；与 N=5/10/15 使用完全相同的 trial identity composition，满足 `U1 ⊂ U5 ⊂ U10 ⊂ U15`。
- 论文边界：论文只报告 N=5/10/15；N=1 是用户要求的本地扩展，`configs\paper.yaml` 保持不变，不把 N=1 冒充论文结果。
- 本地配置：`configs\local_fisher_p1.yaml` 和 `configs\local_fisher_p1_corrected.yaml` 的 `n_values` 更新为 `[1,5,10,15]`；CLI `score-mean --n` 同步接受 1。
- trial 一致性：重新生成审计前后 `artifacts\trials\evaluation.jsonl` 的 SHA-256 均为 `e41c457d31be96a1a2f0fa66af67a553e1007a925ea077153fd4994f5e9646d2`，证明没有改变 3,212 个现有 trials 的身份或顺序。
- 更新后的 trial audit：`n_values=[1,5,10,15]`、eligible speakers 1,606、ineligible 0、target trials 1,606、non-target trials 1,606；`evaluation.audit.json` 376 字节，SHA-256 `02287006cb97207cbc9cb754af9e7edc625df89a23f8697a5de6631ca2ef942d`。
- 自动测试：固定 `pytorch` 环境下 `18 passed`；新增 N=1 prefix nesting 与单 utterance mean-scoring 测试；Python compileall、PowerShell AST 解析和 `git diff --check` 通过。
- 代码 commit：`d886eec`（`feat: add single-utterance evaluation`）。
- 旧 watcher PID `83540` 已在确认其命令行为 `run_o_o_after_training.ps1 -TrainingPid 71796` 后停止；训练 PID `71796` 始终正常运行，未被中断。
- 新 watcher PID：`77800`，启动时间 2026-08-26 23:11:27 +08:00，状态 `waiting_for_training=true`。
- 新 watcher 日志：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\runs\audio_corrected_p1\post_pipeline_n1.stdout.log`、`post_pipeline_n1.stderr.log`。
- corrected 计划输出：`results\o_o_corrected\mean_n1.csv`、`mean_n1.metrics.json`，以及原定 N=5/10/15 文件；所有 N 共享 `artifacts\embeddings\original_evaluation_corrected.npz`。
- 旧 E20 失效训练基线的端到端 N=1 验证已完成：3,212 trials，EER `46.51307596513076%`，threshold `0.4956672191619873`。该数值仅用于验证新增路径和观察旧模型趋势，不作为 corrected 复现结论。

旧 E20 N=1 运行命令：

```powershell
& 'D:\codeAPP\anaconda3\envs\pytorch\python.exe' -m mmsv.cli score-mean --trials artifacts/trials/evaluation.jsonl --original-embeddings artifacts/embeddings/original_evaluation.npz --condition O-O --n 1 --output results/o_o/mean_n1.csv
```

关键代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `configs\local_fisher_p1.yaml` | 1,801 | `353069d5800deef5dbdb511d1ea89ba2d18fe3e452fcab3b2519eadf24996228` |
| `configs\local_fisher_p1_corrected.yaml` | 2,040 | `4e15165f4a35774347a8319353caf1931cc614c960007b8e31a9016299693476` |
| `src\mmsv\cli.py` | 10,217 | `2e42fa9e96f475bc2813fef9b17ea9049c6ceac0e2cf7ee989948d09ed44df78` |
| `src\mmsv\data\trials.py` | 7,317 | `09a9fa0d47a7a1572b3d01a65a4ff46ca4889501ec0874012feecee2db0384e9` |
| `scripts\run_o_o_after_training.ps1` | 1,973 | `f685740c72e50370f76dbc17cc3a38bc1470cd72bb8faba80b24e01b62d2758f` |
| `tests\test_protocol.py` | 2,580 | `2e2594bdc6f80f32ee71fcec7de3878eae3019046d01e1b45c5bb4519e98f0e0` |
| `tests\test_metrics.py` | 1,229 | `bff28cac79cd4194d74b00a083a7ba1b55ee4b998ff37570e0d8f72d239a879f` |

## 21. E22 corrected 30-epoch 训练完成与后处理启动

- 训练状态：完成；checkpoint 完整性验证通过。
- 启动时间：2026-08-24 19:03:05 +08:00。
- 完成时间：2026-08-27 09:18:59 +08:00；墙钟耗时约 62 小时 15 分 54 秒。
- epoch 内计时总和：224,079.41655874252 秒，即约 62 小时 14 分 39 秒。
- 完整规模：30/30 epochs、每 epoch 8,952 physical batches、global step 268,560；epoch 0 至 29 的独立 checkpoint 全部存在。
- 最终训练数据：epoch 29、`epoch_complete=true`、batch 8,952/8,952、loss `0.20560660441443523`、learning rate `7.8125e-06`、该 epoch 耗时 `7533.646664857864` 秒。
- 最终 `last.pt`：115,738,499 字节，SHA-256 `0c69749dbb51929054e3e57990b04d2e737cefd96902f1d0100e80b402313508`。
- `epoch_29.pt`：115,741,587 字节，SHA-256 `92479d2a6551a32bdae42c2123d1db1146fd7128facfef4284cfe7a0c2b391c7`。
- `train.jsonl`：9,433 字节，SHA-256 `026c180ff9498da7598cead377f05f7fdd8731f27b325ebab9989dbeabbc7115`。
- 训练代码 commit：`e0ce0bff6df8a597faa993386e65518e9f0d1d70`；N=1 后处理扩展 commit：`d886eec`。
- 后处理状态：watcher PID `77800` 已验证最终 checkpoint，并启动 Python embedding 提取进程 PID `17004`。
- embedding 范围：evaluation trials 引用的 86,222 个 unique utterances；启动后 97 秒快照为约 817/86,222，端到端平均约 8.4 utterances/s。考虑逐条音频读取与推理波动，保守预计约 2.8 至 3 小时完成；tqdm 的短时瞬时速度不作为 ETA 依据。
- 当前阶段 GPU 快照：embedding 逐 utterance 推理期间利用率瞬时约 4%，显存约 3,525/8,151 MiB；与 30-epoch batch 训练的高利用率不可直接比较。
- D 盘剩余 49.74 GiB。
- 待生成文件：`artifacts\embeddings\original_evaluation_corrected.npz`，以及 `results\o_o_corrected\mean_n1/n5/n10/n15.csv` 和对应 `.metrics.json`。本节不提前填写尚未生成的 EER。
- 后处理日志：`results\runs\audio_corrected_p1\post_pipeline_n1.stdout.log` 和 `post_pipeline_n1.stderr.log`。

## 22. E22 corrected Mean O-O 最终结果

- 状态：完成；30-epoch 训练、checkpoint 校验、trial-filtered embedding 提取及 N=1/5/10/15 Mean O-O 评分全部成功。
- embedding 提取开始：2026-08-27 09:19:06 +08:00。
- embedding 写入完成：2026-08-27 12:02:33 +08:00；86,222 utterances，耗时约 2 小时 43 分 27 秒，端到端平均约 8.8 utterances/s。
- pipeline 完成时间：2026-08-27 12:02:50 +08:00。
- 数据范围：仅 Fisher Part 1；LibriSpeech `train-clean-360` 不参与 O-O 训练或评分。
- trial：1,606 target + 1,606 non-target，共 3,212；speaker-disjoint、target trial call-disjoint；N=1/5/10/15 使用嵌套前缀。
- checkpoint：`results\runs\audio_corrected_p1\last.pt`，epoch 29 完整，global step 268,560，SHA-256 `0c69749dbb51929054e3e57990b04d2e737cefd96902f1d0100e80b402313508`。

最终 EER：

| N | target | non-target | EER | threshold | 论文 Mean O-O | 本地 - 论文 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,606 | 1,606 | 15.504359% | 0.148990 | 未报告 | — |
| 5 | 1,606 | 1,606 | 4.171856% | 0.260938 | 3.87% | +0.301856 pp |
| 10 | 1,606 | 1,606 | 3.113325% | 0.286501 | 3.29% | -0.176675 pp |
| 15 | 1,606 | 1,606 | 2.926526% | 0.289310 | 3.09% | -0.163474 pp |

与 E20 失效基线相比，N=1/5/10/15 EER 分别相对下降 `66.67%/89.16%/91.07%/90.54%`。N=5 比论文高 0.302 个百分点；N=10 和 N=15 分别比论文低 0.177 和 0.163 个百分点。三项均已进入论文同一数量级，且 EER 随 N 增加单调下降。由于本工程只使用 Fisher Part 1、自建 speaker split/trials，数值接近或略优不代表对作者完整数据协议的精确复现。

embedding 完整性与非塌缩诊断：

| 检查项 | 结果 |
|---|---:|
| shape | 86,222 × 192 |
| unique IDs | 86,222 |
| finite | true |
| norm mean / min / max | 1.000000 / 1.000000 / 1.000000 |
| PC1 variance fraction | 4.3922% |
| top-10 variance fraction | 26.5968% |
| participation ratio | 59.4209 |
| 10,000 random-pair cosine mean / std | 0.00899 / 0.13235 |

旧 E20 的 PC1 方差占比为 73.36%、participation ratio 仅 1.81；corrected embedding 的上述结果证明严重表示塌缩已消除。

全部最终输出：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `artifacts\embeddings\original_evaluation_corrected.npz` | 61,680,027 | `9c8c944a758f92dac45b4225f73317a83113d6a3ad744a5eac2580f2fb314ff3` |
| `results\o_o_corrected\mean_n1.csv` | 153,149 | `bf0dbfa2db82e7ab9f373707182ea5eddc533ff3bc06dfe9d19c12a0d78bf512` |
| `results\o_o_corrected\mean_n1.metrics.json` | 301 | `1ec398b172e572546f735314b2f702863fc3c933250bc3c05dac4d642050cd68` |
| `results\o_o_corrected\mean_n5.csv` | 152,351 | `2d27469d128bc388ffb37037d8be65f4bfc4f8c5ffe26bf28c78f938b53ad4bd` |
| `results\o_o_corrected\mean_n5.metrics.json` | 298 | `516ff17bc51619daf3b8730c475b437714e383e54af921c7dfa3b0f49014b0c7` |
| `results\o_o_corrected\mean_n10.csv` | 152,287 | `972991855ca3035f021332d05aedb83ebad786ec34723082626b353eac7246f5` |
| `results\o_o_corrected\mean_n10.metrics.json` | 305 | `5cd68230750dc1a2f902688501c1753b0f63a48a407f27b64991533becaefec3` |
| `results\o_o_corrected\mean_n15.csv` | 152,268 | `7920646487a3c966e64ca3f46b157abbeda713541c91302ab57fc9c7f093f46a` |
| `results\o_o_corrected\mean_n15.metrics.json` | 303 | `e2941dfc8809ea61eee01121a8c66e8176727dfeb8d4e31ec3f303e500025726` |
| `results\runs\audio_corrected_p1\post_pipeline_n1.stdout.log` | 1,892 | `3571feac8f960b33c2b9004b6d2b379d0351fd2c3aefa6e8d292d216611b4826` |
| `results\runs\audio_corrected_p1\post_pipeline_n1.stderr.log` | 2,706,613 | `114970b6b0caa231db5da11e62dcb99b426a33ef962402aad51b6fb8f5323fa3` |

版本化策略：embedding、checkpoint 和逐步日志继续由 `.gitignore` 排除，但其绝对路径、大小和 SHA-256 已记录；8 个 corrected score/metrics 文件与本节摘要进入 Git。下一阶段为 evaluation 匿名化、O-A/A-A 和 corrected semi-informed 训练；Mean O-O 完成不等于整篇论文全部复现完成。

## 23. E24：evaluation StreamVoiceAnon GPU 基准与完整任务启动

- 状态：100 条基准完成并通过；完整 86,222 条生成进行中。
- 数据范围：evaluation trials 引用的 86,222 条 Fisher Part 1 utterances，共 94.14482750000104 小时；匿名目标仅来自 LibriSpeech `train-clean-360` 的 99,278 条大于 4 秒 reference pool。
- 模型：StreamVoiceAnon `dual_ar_delay_0_8.pth`，CUDA，delay 2 frames，speaker embedding mixing `alpha=1.0`，输出 16 kHz FLAC。
- 基准开始：2026-08-27 15:23:17 +08:00。
- 基准完成：2026-08-27 15:29:06 +08:00；墙钟 347.6405532 秒。
- 基准结果：processed 100、generated 100、skipped 0、missing 0、unreadable 0、wrong format 0、nonfinite 0；manifest 行数与 plan 顺序完全一致。
- 基准 source/output 时长：256.24 / 253.7974375 秒；输出 4,424,036 字节。
- duration relative error：P50 `0.9874%`、P95 `3.4072%`、最大 `3.9170%`。
- 实时系数：`1.3566990056`；按基准投影完整生成约 `127.7262` 小时，即约 5.32 天，动态预计 2026-09-01 23:13 +08:00 前后完成。
- 完整输出空间投影：5,851,546,916 字节，约 5.45 GiB；启动时 D 盘空闲 53,225,635,840 字节，投影低于 80% 安全阈值。
- 完整任务启动：2026-08-27 15:29:06 +08:00；监督进程 PID `87012`、当前完整生成 Python PID `74380`。
- 进度快照：2026-08-27 15:32:08 +08:00，156/86,222 个 FLAC、6,645,735 字节；GPU 利用率 27%、显存 3,784/8,151 MiB、功耗 32.35 W。低于训练阶段利用率是自回归逐 utterance 生成和音频 I/O 的预期表现。
- 正式输出目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymized\evaluation`。
- 正式 manifest/audit/progress：`artifacts\metadata\fisher_anonymized_evaluation_manifest.csv`、`.audit.json`、`.progress.jsonl`。
- 监督日志：`results\runs\anonymization_evaluation\supervisor.stdout.log`、`supervisor.stderr.log`。
- 自动后处理 watcher PID `59036`：等待生成 PID `87012`；仅在 audit 与 `final.validation.json` 均确认 86,222 条完整后，提取 `artifacts\embeddings\anonymized_evaluation_corrected.npz`，然后生成 `results\o_a_corrected` 和 `results\a_a_corrected` 下 N=1/5/10/15 Mean 结果。
- 后处理日志：`results\runs\anonymization_evaluation\post_scoring.stdout.log`、`post_scoring.stderr.log`。
- 安全行为：基准或完整校验失败时不启动下一阶段；已生成的非空 FLAC 会在重启时跳过，可断点续跑。
- 自动测试：`19 passed`；Python compileall、PowerShell AST 解析与 `git diff --check` 通过。
- 监督/验证代码 commit：`a080974`；自动 O-A/A-A 后处理 commit：`1c036c5`。

基准产物与代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `results\runs\anonymization_evaluation\benchmark.manifest.csv` | 22,718 | `ebe62b4b158274b0f818586797c91483695e5da4d8fdfca1373570be8e97f53a` |
| `results\runs\anonymization_evaluation\benchmark.audit.json` | 556 | `e81e0eb15bbe32ff3dde797affa8d90207f1abb21d82f4ce290049ac11ca6377` |
| `results\runs\anonymization_evaluation\benchmark.progress.jsonl` | 20,391 | `4f4f8fb93f5168bae0b65e4787c0d5060b11636460b102eed0e03c669656e91e` |
| `results\runs\anonymization_evaluation\benchmark.validation.json` | 996 | `3425828171e524f320dd7519ad26e0139c106a49a7b92d9085eae85b31d50c5c` |
| `scripts\benchmark_then_anonymize_evaluation.ps1` | 3,538 | `6d80d6aa22db36b76389d87ca144be317bfc85aca0c6d6ac6fe871da1556b2e4` |
| `scripts\validate_anonymization_outputs.py` | 6,033 | `62e65a569ffef72728accf82acdf9397aa01b79ad876b7ba4185eb94d389bb3c` |
| `tests\test_anonymization_validation.py` | 1,178 | `7f5bbc8391950b9fe3966ffb5b3b176806b902962efd273dc779e433f644e141` |
| `scripts\run_anonymized_scoring_after_generation.ps1` | 2,847 | `35a2aef118f635ba5d09a64c27ca67db950bf9ce2f6dcc58637fd555f261126a` |

## 24. E25：evaluation 匿名化加速 A/B 与双进程续跑

- 状态：加速方案筛选完成；正式路径确定为双 FP32 进程，等待 Git 提交后从已有 FLAC 断点启动。
- 诊断与测速时间：2026-08-27 15:40:53 至 16:09:06 +08:00。
- 原任务暂停点：2026-08-27 15:40:53 +08:00；保留 303/86,222 条正式 FLAC，共 13,563,981 字节，没有删除或覆盖。
- 根因：上游 `InferenceWrapper` 将主模型 KV cache 固定为 `max_batch_size=1`，逐 utterance 自回归生成；原运行 GPU 利用率约 27%–36%。Fisher SPHERE 整通话解码已由本工程 4-entry LRU cache 缓解，不是当前第一瓶颈。
- 公平样本：evaluation plan 从零基下标 303 开始；单进程各 20 条，前 3 条排除为预热；双进程分别处理下标 303–322 与 323–342，共 40 条且输出到互不重叠的独立 benchmark 目录。
- 单进程 FP32 基线：20 条墙钟 93.0427 秒；排除 3 条后 measured audio 56.74994 秒、elapsed 67.89395 秒、稳态 RTF `1.1963705`、0.25039 items/s；20/20 可读、16 kHz、finite。
- `torch.compile` 结论：不采用。首次因 Windows 默认临时路径过长失败；短缓存路径解决后，AR、encoder 和 decoder 均在 PyTorch 2.9.1 + Triton 的 Windows static CUDA launcher 触发 `OverflowError: Python int too large to convert to C long`。所有失败均发生在第一条测试输出落盘前，没有污染正式结果。
- 单进程 FP16：20 条墙钟 94.1438 秒；稳态 RTF `1.2231524`，比 FP32 慢约 2.24%；20/20 输出有效。实测运行中总显存约 3,393 MiB，FP32 基线约 3,920 MiB。
- 双 FP16：40 条墙钟 138.5460 秒、峰值总显存 5,783 MiB、GPU 利用率快照 77%；两 worker 均 return code 0，40/40 输出有效；相对两个单进程 FP32 基线串行墙钟加速 `1.3431×`。
- 双 FP32：40 条墙钟 144.0900 秒、峰值总显存 6,754 MiB、GPU 利用率快照 76%；两 worker 均 return code 0，40/40 输出有效；相对串行基线加速 `1.2915×`。
- 选择：双 FP32。它只比双 FP16 慢约 4%，但与已生成的 303 条和原始 StreamVoiceAnon 路径保持相同权重精度；峰值仍低于 8,151 MiB，约留 1,397 MiB 余量。三进程未采用，因为按双进程峰值推算会接近或超过显存上限。
- 正式分片：按 source 总时长而不是条数平衡；split index `44,950`。worker 1 为 44,950 条 / 169,467.879 秒，worker 2 为 41,272 条 / 169,453.500 秒。
- 新投影：按双 FP32 加速比，完整墙钟约 `127.7262 / 1.2915 = 98.90` 小时，即约 4.12 天；已有 303 条会略微缩短该时间。该值是短基准投影，不作为完成时间保证。
- 正式运行方式：`scripts\anonymize_evaluation_dual.ps1` 启动两个互斥 plan slice；两 worker 完整后，`scripts\merge_anonymization_manifests.py` 按原 plan 顺序合并 manifest，运行 86,222 条完整性验证，再由原后处理 watcher 提取匿名 embedding 并评分 Mean O-A/A-A N=1/5/10/15。
- 可恢复性：两个 worker 都跳过已有非空正式 FLAC；任一 worker 未产生完整 audit 时不合并、不评分，重新执行监督器即可续跑。
- 自动测试：`21 passed`；Python compileall、两个 PowerShell AST 解析和 `git diff --check` 通过。

测速产物与代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `results\runs\anonymization_compile_benchmark\20260827_154743\benchmark.json` | 3,733 | `6f4d00fb4ffe4f1a10c77d257b03e5b986ecd2bdac8a9d0e2bd341effdc27a63` |
| `results\runs\anonymization_compile_benchmark\20260827_155919\benchmark.json` | 2,440 | `217c320df4951f228d9fdb90c7c89eb9776dac96a08d91f91e4dba623f9ea216` |
| `results\runs\anonymization_dual_fp16_benchmark\20260827_160244_971\benchmark.json` | 15,612 | `ce4a0308f486fc5e0114225cc7cb56b75660550ad0a2f50d0dd11d6a969251a9` |
| `results\runs\anonymization_dual_fp16_benchmark\20260827_160641_860\benchmark.json` | 14,876 | `622a9e7bbf92fddaa4e51e64af358885edd9292883c15f9a6a8843a5c76d4808` |
| `src\mmsv\anonymization.py` | 13,184 | `8db094319625460c9f8cb0ca9fe093003278d4ddac09e4688baf82144c0233b8` |
| `src\mmsv\cli.py` | 10,662 | `99d8e220b3aabe5e21d77cf880205a96128a0d1269433df3ae183d9031f89556` |
| `scripts\benchmark_streamvoice_compile.py` | 9,376 | `a3b2de4d4aed2b28d6c853ee1ea8e72134d10a55ebaac1973a52a6c0860f9bff` |
| `scripts\benchmark_dual_fp16.ps1` | 4,263 | `6495143541b889db0cf8ba5fee153cf1638385631a7e6731117fb2080b212914` |
| `scripts\anonymize_evaluation_dual.ps1` | 5,020 | `64adbfcf6e2756254675443d2c7339dea1704da0597872ad5dd370865d6abad4` |
| `scripts\merge_anonymization_manifests.py` | 2,949 | `9a51a10235d55506fec5483c6528d846638d394eb2d1b2cfff6da168a4754b70` |
| `tests\test_anonymization.py` | 4,727 | `0011b3cdf63eca7673f4599e115f3defce7ba7987752206e58139fa885535466` |
| `tests\test_anonymization_validation.py` | 2,994 | `5eb24d5996031fb0f14156a3f075ea215e8982484e7aa30c71023a51d895886a` |
