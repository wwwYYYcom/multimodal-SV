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

- 状态：加速方案筛选完成；正式双 FP32 进程已从已有 FLAC 断点启动并稳定运行。
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
- 加速实现 commit：`9beb126`（`perf: parallelize evaluation anonymization`）。
- 正式续跑启动：2026-08-27 16:16:05 +08:00；监督器 PID `81472`，worker 1 PID `57148`，worker 2 PID `45832`，自动评分 watcher PID `86460`。
- 正式运行目录：`results\runs\anonymization_evaluation\dual_20260827_161605_474`；监督日志为 `results\runs\anonymization_evaluation\dual_supervisor.stdout.log` 与 `.stderr.log`，自动评分日志为同目录下 `dual_post_scoring.stdout.log` 与 `.stderr.log`。
- 启动后快照：worker 1 progress 317/44,950（包含跳过的 303 条）、worker 2 progress 16/41,272；正式输出 333 条、15,233,027 字节；GPU 利用率 76%、显存 6,670/8,151 MiB、功耗 41.23 W、温度 64°C，无 OOM 或推理异常。

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

## 25. E26：完整 evaluation 匿名化完成与离线后处理恢复

- 状态：86,222 条匿名化已完成并通过最终校验；首次自动 embedding 提取因 Hugging Face SSL 中断而失败，本地离线 smoke 已通过，准备重启全量后处理。
- 双进程正式启动：2026-08-27 16:16:05 +08:00。
- 最后一条 FLAC 完成：2026-08-30 09:45:16 +08:00。
- manifest 合并与最终校验完成：2026-08-30 09:47:04 +08:00。
- 墙钟时间：235,757.3949926 秒，即 65.4882 小时；实际 RTF `0.6956108691`。
- GPU 峰值显存：7,731 MiB / 8,151 MiB；双进程运行期间无 OOM。
- 输出：86,222 条 16 kHz mono FLAC，共 5,585,060,420 字节；source/output 总时长 338,921.3790 / 336,916.3377 秒。
- 完整性：plan 86,222 行、manifest 86,222 行、unique IDs 86,222、顺序匹配；missing/unreadable/wrong-format/nonfinite 均为 0。
- 时长相对误差：P50 `0.6874%`、P95 `2.9571%`、最大 `4.3873%`；抽查 100 条 finite。
- 正式音频目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\artifacts\anonymized\evaluation`。
- 首次后处理失败：2026-08-30 09:47 +08:00，`snapshot_download` 访问 `huggingface.co` 时发生 `SSL: UNEXPECTED_EOF_WHILE_READING`；失败发生在 embedding 输出创建前，匿名音频、manifest 和最终校验均不受影响。
- 修复：`scripts\run_anonymized_scoring_after_generation.ps1` 在启动 Python 前固定 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，强制使用本机完整缓存 `models--microsoft--wavlm-large`。
- 离线 smoke：2026-08-30 10:49:03 +08:00 完成；从正式匿名 manifest 提取 1 条 embedding，模型加载成功，单条推理约 1.95 秒。
- 后续输出：`artifacts\embeddings\anonymized_evaluation_corrected.npz`，以及 `results\o_a_corrected`、`results\a_a_corrected` 下 N=1/5/10/15 Mean score 与 metrics。

完成产物与恢复代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `results\runs\anonymization_evaluation\final.validation.json` | 992 | `d760c00a6432337ae961dd5f2aaccf30d686e16a6f6c0be61adc25d339207946` |
| `artifacts\metadata\fisher_anonymized_evaluation_manifest.csv` | 20,623,558 | `30f2b8450b98f5ee571239fb599eaf0c1e0edafd8540a46aac58b79d44c45eea` |
| `artifacts\metadata\fisher_anonymized_evaluation_manifest.audit.json` | 2,412 | `d5ed9527fe41de233579f7a577c553fc11b528a39f5e86a6b7bcc60b1e2df9c6` |
| `results\runs\anonymization_evaluation\offline_embedding_smoke.npz` | 1,203 | `2f3d2a8cc756b37b4980333a3725917b171e55efe64f0ca30547c51e267cfb11` |
| `scripts\run_anonymized_scoring_after_generation.ps1` | 2,945 | `3736de23b7b3a2e0d7093298ca925aa542ff8439c36bcf943c39adc79d8ee8f3` |

## 26. E27：corrected Mean O-A / lazy-informed A-A 最终结果

- 状态：完成。匿名 embedding 提取以及 O-A/A-A 的 N=1/5/10/15 Mean 评分全部成功。
- 后处理启动：2026-08-30 10:50:44 +08:00。
- embedding 写入完成：2026-08-30 11:53:46 +08:00；86,222 utterances，tqdm 推理耗时约 1:02:48，平均约 22.88 utterances/s。
- pipeline 完成：2026-08-30 11:54:11 +08:00；从启动到全部评分完成约 1:03:27。
- 运行环境：`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，使用本地 WavLM-Large 缓存；正式 checkpoint 为 `results\runs\audio_corrected_p1\last.pt`。
- 运行入口：`scripts\run_anonymized_scoring_after_generation.ps1`。该脚本依次调用 `mmsv.cli extract-embeddings`，再对 O-A/A-A 和 N=1/5/10/15 调用 `mmsv.cli score-mean`。
- trials：`artifacts\trials\evaluation.jsonl`；每项 1,606 target + 1,606 non-target，共 3,212 trials；N 使用同一 nested enrollment 前缀。
- 原始 embedding：`artifacts\embeddings\original_evaluation_corrected.npz`。
- 匿名 embedding：`artifacts\embeddings\anonymized_evaluation_corrected.npz`。
- O-A 输出目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\o_a_corrected`。
- A-A 输出目录：`D:\deeplearning\ICASSP2027\multimodal_sv_reproduction\results\a_a_corrected`。
- 日志：`results\runs\anonymization_evaluation\dual_post_scoring_retry.stdout.log` 和 `dual_post_scoring_retry.stderr.log`。

最终 EER：

| 条件 | N | target | non-target | EER | threshold | 论文 Mean | 本地 - 论文 |
|---|---:|---:|---:|---:|---:|---:|---:|
| O-A | 1 | 1,606 | 1,606 | 43.524284% | 0.033054 | 未报告 | — |
| O-A | 5 | 1,606 | 1,606 | 38.418431% | 0.059750 | 40.61% | -2.191569 pp |
| O-A | 10 | 1,606 | 1,606 | 36.799502% | 0.069051 | 39.77% | -2.970498 pp |
| O-A | 15 | 1,606 | 1,606 | 36.674969% | 0.069988 | 39.30% | -2.625031 pp |
| A-A lazy-informed | 1 | 1,606 | 1,606 | 47.384807% | 0.330420 | 未报告 | — |
| A-A lazy-informed | 5 | 1,606 | 1,606 | 39.352428% | 0.701010 | 43.77% | -4.417572 pp |
| A-A lazy-informed | 10 | 1,606 | 1,606 | 31.506849% | 0.810067 | 40.26% | -8.753151 pp |
| A-A lazy-informed | 15 | 1,606 | 1,606 | 25.778331% | 0.856150 | 37.59% | -11.811669 pp |

解释边界：这里的 A-A 使用原始语音训练的 corrected checkpoint 同时比较匿名 enrollment 与匿名 test，属于 lazy-informed A-A；尚未使用匿名训练集重训攻击者，因此不得当作论文 Table I 的 semi-informed A-A。论文值只作量级对照；本地仅使用 Fisher Part 1 + LibriSpeech `train-clean-360`，并使用自建 split/trials，数值优于论文不代表严格超越作者协议。O-A 从 N=5 到 N=15 下降 1.743462 pp，A-A 从 N=5 到 N=15 下降 13.574097 pp，说明匿名侧 enrollment 聚合在当前 A-A 协议中收益更明显。

匿名 embedding 完整性与表示诊断：

| 检查项 | 结果 |
|---|---:|
| shape | 86,222 × 192 |
| unique IDs | 86,222 |
| 与原始 embedding ID 集及顺序 | 完全一致 |
| finite | true |
| norm mean / min / max | 1.000000 / 1.000000 / 1.000000 |
| PC1 variance fraction | 9.3640% |
| top-10 variance fraction | 41.4806% |
| participation ratio | 34.5691 |
| seed 1234、10,000 random-pair cosine mean / std | 0.32091 / 0.15287 |

输出文件指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `artifacts\embeddings\anonymized_evaluation_corrected.npz` | 61,679,125 | `4cccde8929e403b353d63a500e6b3063f203df0cbcc869919a34b0b23b9be7a0` |
| `results\o_a_corrected\mean_n1.csv` | 154,703 | `32534e7cac4da2255acc32fdfa26ae144c69852514157eb73acf047671d5191f` |
| `results\o_a_corrected\mean_n1.metrics.json` | 301 | `c92d6297ec254eee39adb077ff97961db7f2e72cf1996661970df4666083e5af` |
| `results\o_a_corrected\mean_n5.csv` | 154,299 | `3e5461f8cdde54acb482d40ea37f2baa52737a1a137bdb27e29d08df33815179` |
| `results\o_a_corrected\mean_n5.metrics.json` | 300 | `e052918fc8a7f86168e8aa8d7a07a28a6845dc36b64de1b4a4c87b5aaa7dc15c` |
| `results\o_a_corrected\mean_n10.csv` | 154,122 | `bcf72f04c41dc46fb4b49f10bf91381afc12e7a40e0777b19ac66e524d6ed2dd` |
| `results\o_a_corrected\mean_n10.metrics.json` | 302 | `543d8f6307fca29eca36342dd940248c6bae3a69ba2f9be208bd13ef3b8783da` |
| `results\o_a_corrected\mean_n15.csv` | 154,082 | `3862ba089fa283b7457551b6fc5bfe8bdc631b12cc1bbd378597407616621490` |
| `results\o_a_corrected\mean_n15.metrics.json` | 303 | `ea6454b12a22b821d68542d161a37d6f87ecd77a49c627adadbb1b17c57ba52b` |
| `results\a_a_corrected\mean_n1.csv` | 151,431 | `327d2772d78ea412d62515076faa0988262f222f33dad990447d2b2589d349bc` |
| `results\a_a_corrected\mean_n1.metrics.json` | 299 | `90f7d0a1df3d441079b7f54cbc25aa93ca55499438744489dee4cbd7163f5436` |
| `results\a_a_corrected\mean_n5.csv` | 149,627 | `2e090ed934396ae469be6ce6d564bff257f74ba348971db87913c5863c621e6e` |
| `results\a_a_corrected\mean_n5.metrics.json` | 301 | `f1ad00228075bb32641ef6cdf922796bf2f8ec58725846584cd142e1626ee7ac` |
| `results\a_a_corrected\mean_n10.csv` | 149,621 | `dba10163fdb4086bf20c651c3906e2188a8109e601f04d8ac80d66feb36d98fb` |
| `results\a_a_corrected\mean_n10.metrics.json` | 303 | `7fe0c33c1ddff25b14a3a212b1e40f048dced941105fd7e9c504283b8e2f4b41` |
| `results\a_a_corrected\mean_n15.csv` | 149,576 | `abd9a63554413ae63766abd6f81aa655069b82ca046c26262a3bf531e2d5c446` |
| `results\a_a_corrected\mean_n15.metrics.json` | 303 | `3294e68ab350cde8aa1a397395c39125aca9249da75645a73f230f715c879556` |
| `results\runs\anonymization_evaluation\dual_post_scoring_retry.stdout.log` | 3,014 | `9285cdfff62387cc7916977b9c908af3cbd021e8f1aed4e78859a297e57c0c76` |
| `results\runs\anonymization_evaluation\dual_post_scoring_retry.stderr.log` | 1,927,269 | `fcba6400b94e081ede15712dbb7f2946d92a6015aad1cda35f66c8a8e6783b0b` |

运行代码与输入指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `scripts\run_anonymized_scoring_after_generation.ps1` | 2,945 | `3736de23b7b3a2e0d7093298ca925aa542ff8439c36bcf943c39adc79d8ee8f3` |
| `src\mmsv\cli.py` | 10,662 | `99d8e220b3aabe5e21d77cf880205a96128a0d1269433df3ae183d9031f89556` |
| `src\mmsv\train.py` | 23,371 | `525b3c531da371bbe061ce88f85e5c3e26ff3357be07d858c7c3a14275d9df8b` |
| `src\mmsv\metrics.py` | 4,593 | `0c6da0d5ef0f2a2a22e407c6d05e8dfbe715f20240f81fda0db69698468680e0` |
| `configs\local_fisher_p1_corrected.yaml` | 2,040 | `4e15165f4a35774347a8319353caf1931cc614c960007b8e31a9016299693476` |
| `artifacts\trials\evaluation.jsonl` | 2,592,387 | `e41c457d31be96a1a2f0fa66af67a553e1007a925ea077153fd4994f5e9646d2` |
| `artifacts\embeddings\original_evaluation_corrected.npz` | 61,680,027 | `9c8c944a758f92dac45b4225f73317a83113d6a3ad744a5eac2580f2fb314ff3` |

Git 策略：16 个 O-A/A-A score/metrics 小文件与本节实验总账进入 Git；61.7 MB embedding、5.585 GB 匿名音频和逐步日志继续由 `.gitignore` 排除，但绝对路径、大小、完成时间和 SHA-256 已在本节留档。本节当时拟采用 7,272 条 train call-side 近似；该决定随后被判定不符合论文全 utterance 协议并在第 27 节中止、修正。

## 27. E28：中止 one-per-call-side 近似并改为全 utterance semi-informed 协议

### E28a：7,272 条缩减流水线（已中止，不用于结果）

- 状态：中止。用户指出该方案不等价于论文的 utterance-level semi-informed 协议后立即停止。
- 启动时间：2026-08-30 15:07:57 +08:00。
- 停止时间：2026-08-30 15:10 +08:00。
- supervisor PID：`26752`；worker PID：`99696`、`97468`；三者均已终止。
- 错误输入计划：`artifacts\anonymization\train_one_per_call_side_plan.csv`，7,272 条、7.6451 小时、5,231 speakers。
- 停止时保留 46 条 FLAC，共 2,262,285 字节；路径仍位于 `artifacts\anonymized\train`。这些 utterance 及 seed 1234 reference 映射均属于后续全量计划，因此正式任务可校验后跳过复用；它们未进入训练。
- 中止日志：`results\runs\anonymization_train\supervisor.stdout.log`、`supervisor.stderr.log`，以及 `results\runs\anonymization_train\dual_20260830_150757_810` 下两个 worker 的日志和 progress。
- 启动前代码提交：`5ead896`。该提交保留在 Git 历史中作为决策记录，后续提交将协议改为全 utterance，不改写历史。

### 论文原文核验与本地协议映射

- 核验来源：`C:\Users\wwwYYYcom\Zotero\storage\DH7AVWNV\Garg 等 - 2026 - Multimodal Speaker Verification as a Threat to Speaker Anonymization.pdf`，完整阅读相关第 4–6 页并渲染核对版面。
- 第 IV-A 节：作者按 speaker 划分为 5,712/250/1,753 train/validation/evaluation；每条待匿名 utterance 独立从 LibriSpeech `train-clean-360` 与 `train-other-500` 的大于 4 秒语音中随机选择 target，固定 delay=2、alpha=1。
- 第 IV-C(c) 节：明确写明 Fisher training split 在 utterance level 匿名化；semi-informed 模型从对应 lazy-informed 模型初始化，继续优化 ECAPA backend 与 aggregation module，学习率 `1e-4`，训练 15 epochs。
- 本地受用户数据范围约束：只使用 Fisher Part 1 和 LibriSpeech `train-clean-360`，因此本地 split 为 5,231/229/1,606 speakers，reference pool 为 clean-360 的 99,278 条大于 4 秒语音。正式名称为“Fisher Part 1 + LibriSpeech train-clean-360 semi-informed reproduction”，不是作者完整数据范围的精确复刻。
- 正式本地训练语音：Fisher Part 1 train split 的全部 572,951 utterances，而不是每个 call-side 只取一条；总时长 2,161,308.76 秒，即 600.3635 小时。

### E28b：全 utterance 计划与可恢复流水线准备

- 状态：正式全量双进程匿名化运行中；完成后自动进入 15 epoch semi-informed 训练与 O-A/A-A 评分。
- 全量计划完成时间：2026-08-30 15:14:00 +08:00。
- 计划：572,951 行、572,951 unique IDs、572,951 unique output paths、5,231 speakers；`one_per_call_side=false`。
- reference：只来自现有 `train-clean-360` 大于 4 秒池；seed 1234 下选中 98,958 条 unique references；每个 utterance 使用稳定的独立随机映射。
- 双进程按 source 总时长平衡：split index `301,378`；worker 1/2 分别为 1,080,657.03 / 1,080,651.73 秒。
- 46 条中止产物全部存在于全量计划，且在缩减/全量计划中的 reference ID 完全一致，可安全断点复用。
- 实测投影：按 evaluation 输出压缩率，预计约 35,828,004,345 字节（33.37 GiB）；按双进程 RTF 0.6956108691，匿名化墙钟约 417.62 小时（17.40 天）。
- 启动前 D 盘：46,590,763,008 字节（43.39 GiB）可用；流水线内置“剩余预计输出 + 6 GiB”空间保护，预留 checkpoint、embedding、合并临时库和日志空间。
- 训练：从 `results\runs\audio_corrected_p1\last.pt` 使用 `--init-from` 初始化并重置 optimizer；WavLM 冻结，ECAPA 继续训练；physical batch 64、feature micro-batch 32、AMP、短语音 repeat、学习率 `1e-4`、15 epochs；每 epoch 8,952 full batches，共 134,280 optimizer steps。
- 训练完成后自动提取 semi-informed 模型下的 original/anonymized evaluation embeddings，并计算 Mean O-A/A-A N=1/5/10/15；论文 Table I 的 Mean semi-informed A-A N=5/10/15 参考值为 `18.68/14.70/13.70%`。
- 全量计划、音频、manifest、embedding、checkpoint 和逐步日志由 `.gitignore` 排除；代码、配置、测试和本节总账进入 Git。

正式运行入口：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/anonymize_train_dual_then_train_semi.ps1
```

关键输入与代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `artifacts\anonymization\train_all_utterances_plan.csv` | 355,706,506 | `34d6d893efceab4508591965c73c31927d401afdaf503a4a9c04552b45d7cb28` |
| `artifacts\anonymization\train_all_utterances_plan.audit.json` | 913 | `60ab5ea0d5c521e900e7be6c9cfa302ea206b462c3c065f849a17beb78634641` |
| `artifacts\metadata\librispeech_target_pool.csv` | 20,264,920 | `ff04bd9e77d7702560e75147a44fd2e313f1957cf2e65b78927463062f19bea2` |
| `artifacts\metadata\fisher_manifest.csv` | 300,904,760 | `e9172730119921a6358d7e47125c8d6e0a77950ef6edb4f90eac0e6ca2574d12` |
| `artifacts\metadata\speaker_splits.csv` | 137,401 | `9dd0c24d47aeb94ca453b35a76d755976d504f86fde0abdcaf284919fb34f0fc` |
| `results\runs\audio_corrected_p1\last.pt` | 115,738,499 | `0c69749dbb51929054e3e57990b04d2e737cefd96902f1d0100e80b402313508` |
| `configs\semi_local_corrected.yaml` | 860 | `7af7c1c0a1678830e1dc5e04ddba2dd958f6f17ce864faad74ab1295b806f413` |
| `scripts\anonymize_train_dual_then_train_semi.ps1` | 10,424 | `1cd3d07e127997e3569eaaca5dfc1ea200ec60d604d1643bee30de4b61ffc451` |
| `src\mmsv\anonymization.py` | 13,632 | `5a95256954c50ff704390a0455011267b5df1cefac14a4f15ff9f919f381811a` |
| `scripts\merge_anonymization_manifests.py` | 4,212 | `0ff17a0ac1b2ce4d02c1e975b3c54e1cedee0f3ab80d0fc3a41b644da0ebe949` |
| `scripts\validate_anonymization_outputs.py` | 6,491 | `066840aacdb9c4fb71d07e89149c111f965b5898523f915a2cb5e7fa9553fe3d` |
| `tests\test_anonymization.py` | 6,813 | `9957f6e1f5e22e312e0a9358cb3eaac61a560af325d4517591c7342c0056d4f9` |

为支持 572,951 条规模，runner 已改为用 `itertools.islice` 流式读取计划、逐行写入临时 manifest 并原子替换，progress 文件保持行缓冲；manifest 合并使用磁盘 SQLite 索引恢复 plan 顺序；最终验证改为流式 plan/manifest 对齐。这样避免两个 worker 各自同时持有 355 MB 计划及数十万字典对象。PowerShell AST、Python compileall、流式 slice/原子 manifest 回归测试、乱序 worker manifest 合并测试及全部测试均通过（`22 passed`）。

正式启动记录：

- 修正代码提交：`939cc03bbceda0d281cf49ec1249ba50bc91f470`（`fix: use full utterances for semi-informed training`）。
- 启动时间：2026-08-30 15:34:03 +08:00。
- supervisor PID：`48012`；worker 1 PID：`86028`；worker 2 PID：`98496`。
- 正式运行目录：`results\runs\anonymization_train\dual_20260830_153403_742`。
- supervisor 日志：`results\runs\anonymization_train\full_supervisor.stdout.log`、`full_supervisor.stderr.log`。
- worker 日志：正式运行目录下的 `worker1.stdout.log`、`worker1.stderr.log`、`worker2.stdout.log`、`worker2.stderr.log`；progress 使用对应 `worker*.manifest.progress.jsonl`。
- 启动空间：46,590,685,184 字节；复用输出：2,262,285 字节；预计最终匿名音频 35,828,004,345 字节。
- 稳定性快照：2026-08-30 15:38:08 +08:00，worker 1/2 progress 为 79/301,378 与 42/271,573，GPU 显存 5,028 MiB；随后 15:39:04 输出目录共 197 条、9,463,283 字节。GPU 利用率约 75%、显存约 5,074/8,151 MiB、61°C、39.22 W；supervisor 与 worker stderr 均为空，无 OOM。
- 主存快照：两个 StreamVoiceAnon worker working set 分别约 3.50 GB 与 3.40 GB；计划/manifest 已流式化，该内存主要为两个模型实例，不会随 572,951 行线性累积。
- 匿名化预计约 17.4 天；按启动时间粗略推算约 2026-09-17 前后完成，实际完成时间以逐条源时长和机器连续运行情况为准。随后全量训练和两套 embedding/评分预计另需约 3 天。

## 28. E29：全量匿名化超长输入故障、修正与恢复

### 故障与根因

- 发现时间：2026-08-31 13:05 +08:00。原 supervisor 仍在，但 worker 1 已于 08:29:59 停在计划索引 13,931；worker 2 继续运行。发现时累计约 30,687 / 572,951 条（5.36%）。
- 精确故障输入：`fe_03_00170_B_0060`，source 43.24 秒，reference `3889-130125-0005` 为 14.045 秒。该输入在 StreamVoiceAnon 上游自回归推理约 703/930 帧处触发 vectorized gather 越界，随后表现为 CUDA device-side assert / CUBLAS internal error；并非显存不足。
- 原 evaluation 匿名化计划的最长输入为 28.37 秒，因而此前 86,222 条 evaluation 音频可全部完成；全量 train 计划最长为 96.59 秒，超过 30 秒共 13 条，其中超过 40 秒 7 条。故障只在扩大为 572,951 条全量训练语音后暴露。
- 原运行日志：`results\runs\anonymization_train\dual_20260830_153403_742\worker1.stderr.log`，27,312,946 字节，SHA-256 `2a48035ab9364b684e6c723a6a469f62b9e3139bcf96ccf0626c270bef477aa6`。原 supervisor stdout 为 142,522 字节，SHA-256 `db4fc21cd6550bebdd65a12e52fc7ace60fcc508616068a4e9b1ae4deab46d52`。

### 修正、验证与 Git 记录

- 监督脚本先加入 worker 非零退出自动恢复，单 worker 最多重试 20 次；后续训练最多重试 10 次。提交：`43263ca733d2108a46a012d17a731ab9dee80518`（`fix: restart failed long-running workers`）。这能处理偶发失败，但同一超长输入会确定性重现，不能单靠重试解决。
- 最终修正：仅当 source 超过 30 秒时，将波形等分为不超过 30 秒的连续块；各块使用同一 reference、`delay=2`、`alpha=1` 独立匿名化，再按原顺序拼接并写为单个 FLAC。未超过阈值的 572,938 条语音路径完全不变。audit 新增 `generated_chunked_utterances`、`generated_inference_chunks` 与 `max_source_chunk_seconds`。
- 精确故障样本 GPU smoke 于 2026-08-31 13:15:48 +08:00 完成：1 条 source 被分为 2 个推理块，输出 `artifacts\anonymized\train\3696\fe_03_00170_B_0060.flac`，359,805 字节、16 kHz、mono、43.189125 秒、finite=true；相对 source 时长误差 0.117657%。输出 SHA-256 `8e6c4e986d1de90cb248810db5648e93b3e7c80f993506227e4c00826f0fd654`。
- smoke 文件：`results\runs\anonymization_train\long_utterance_chunk_smoke.manifest.csv`（265 字节，SHA-256 `12841dee92f108eb62ab228f7793d3c07207e4372b45ba1c695a2160018f66c2`）；对应 audit（824 字节，SHA-256 `6621d28c1591f9587d6902888f2c2a0bac2ff973fb7b4e6a09bb96d6c1561dcd`）与 progress（255 字节，SHA-256 `2813c832888e81f95d60de8cc363f553c22890ffe91033c72721959c15860181`）。
- 验证：PowerShell AST 通过，全部 Python 测试 `22 passed`。最终修正提交：`e975a5e65733d0fc7d370073138c029ef7872fa5`（`fix: chunk overlong anonymization inputs`）。

修正后运行代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `scripts\anonymize_train_dual_then_train_semi.ps1` | 13,563 | `1d65e35c4cb5e018508a66c0f042dcea873b73cd46325c61aa1400315e277422` |
| `scripts\merge_anonymization_manifests.py` | 4,580 | `2c8d650b09b5a800b75d1639a3bef6e27f501c43d25c441b64cadada9694f7da` |
| `src\mmsv\anonymization.py` | 15,168 | `3d4781acbe150014f2adb4e4a591d5c7110a2412e7da6751168f77ce09e522d0` |
| `src\mmsv\cli.py` | 10,920 | `95bd799ffd9adb739f35410ec19ec541eba24b9e21e8e5681216a8d97fc1d323` |
| `tests\test_anonymization.py` | 7,095 | `760c4813c2e83c6515227bd6b7c2e7d697beb9a394928b6668644b3f8baa0378` |

### 正式恢复与当前快照

- 保留之前已成功生成的 30,752 条左右输出，停止旧 supervisor/worker 后从已有 FLAC 断点跳过恢复，没有重做或删除有效产物。
- 最终恢复时间：2026-08-31 13:17:15 +08:00；supervisor PID `94440`，worker 1/2 PID `102176/67860`；代码提交 `e975a5e65733d0fc7d370073138c029ef7872fa5`。
- 运行目录：`results\runs\anonymization_train\dual_20260831_131714_406`。supervisor 日志：`results\runs\anonymization_train\full_chunked_supervisor.stdout.log`、`full_chunked_supervisor.stderr.log`；worker 日志为运行目录下 `worker1.attempt1.*.log`、`worker2.attempt1.*.log`，progress 为 `worker*.manifest.progress.jsonl`。
- 13:18:36 快照已确认 worker 1 越过旧故障索引；13:23:22 累计输出 30,922 / 572,951 条（5.396971%），共 1,779,006,684 字节（1.6568 GiB）。两个 worker 均存活，GPU 利用率约 76%、显存约 4,954 / 8,151 MiB、63°C、39.65 W；D 盘剩余 42,053,750,784 字节（39.1656 GiB）。
- 匿名化仍为当前阶段，尚未开始 semi-informed 15 epoch 训练。按原实测投影及本次约 4.8 小时单 worker 停机影响，预计匿名化约 2026-09-17 至 09-18 完成；机器暂停、后续异常和超长分块实际耗时会改变该日期。匿名化完成后监督脚本会自动合并/校验 manifest、开始训练并最终输出 O-A/A-A N=1/5/10/15。

## 29. E30：双 RTX 4090 D Linux 服务器迁移准备

### 服务器核验

- 核验日期：2026-09-01 +08:00；主机 `worker-0`，用户目录 `/public/home/wwwyyycom123_`。
- Python 3.10.12；系统预装 PyTorch `2.1.0+cu121`、CUDA available=true。该 PyTorch 低于工程 `torch>=2.4` 约束，不用于正式运行；迁移方案在持久化用户目录创建独立 venv，并安装与本机成功运行一致的 `torch 2.9.1+cu128`、`torchaudio 2.9.1+cu128`、`transformers 4.56.2`、NumPy 1.26.4、SciPy 1.13.1 等环境。
- GPU：2 × NVIDIA GeForce RTX 4090 D，每张 PyTorch 可见显存 47.3731 GiB；`nvidia-smi` 报告驱动 595.58.03、CUDA 13.2、单卡 49,140 MiB，核验时均无运行进程。
- CPU/内存：32 logical CPUs、251 GiB RAM、243 GiB available、无 swap。
- 持久化 NFS：`/public/home/wwwyyycom123_`，总计约 20 TiB、可用约 13 TiB。容器根 overlay 虽有约 806 GiB 可用，但不作为正式持久化目录。
- 缺失工具：`curl`、`rsync`、`tmux`、`ffmpeg`、`conda` 未检测到；发行版尚待 `/etc/os-release` 确认（设备名 `/dev/mapper/rl-root` 提示可能是 Rocky/RHEL 系）。迁移文档先检测 `apt-get`/`dnf`/`yum` 再安装基础工具，并采用 Python venv，不依赖 conda。

### Linux 实现与验证

- 代码提交：`47bc1b3c12f3334de7b411beae131f53bebf5bb4`（`feat: add Linux multi-GPU migration pipeline`）。
- `scripts/remap_csv_paths.py` 流式读取大型 CSV，最长前缀优先、大小写不敏感地把 Windows 路径转换为 Linux 路径；临时文件完整写入后原子替换。任何未覆盖的绝对路径默认报错，并写 `*.remap.audit.json`。
- `scripts/remap_artifacts_for_linux.sh` 统一转换全量 train plan、Fisher manifest、LibriSpeech reference pool 和 anonymized evaluation manifest；默认映射本机 corpora/project 根目录，服务器目标由 `CORPORA_ROOT`/`PROJECT_ROOT` 显式指定。
- `scripts/anonymize_train_multigpu_then_train_semi.sh` 默认使用 GPU `0,1`，每卡 4 个 StreamVoiceAnon worker，共 8 个互斥 slice；每个 worker 独立 manifest/progress/stdout/stderr，失败最多自动重启 20 次，已有非空 FLAC 断点跳过，超过 30 秒 source 继续使用分块拼接。全部 worker 完成后按原 plan 顺序合并 572,951 行 manifest、完整校验、训练 15 epochs，并自动完成 O-A/A-A N=1/5/10/15。
- 8-worker dry run 完成时间：2026-09-01 10:18:43 +08:00。slice 1–7 各 71,619 行，起点依次为 0、71,619、143,238、214,857、286,476、358,095、429,714；slice 8 起点 501,333、71,618 行；终点精确为 572,951。
- 验证：两个 Bash 文件 `bash -n` 通过；Python compileall 通过；路径重写、匿名化、合并校验及全工程测试共 `25 passed`；`git diff --check` 通过。

迁移代码指纹：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `SERVER_MIGRATION.md` | 8,584 | `c493fb56716feba37d97ed64241d44113f3a692dd61ad2817f8a9d4c9624ee4d` |
| `scripts\anonymize_train_multigpu_then_train_semi.sh` | 10,792 | `1d8e6c74e7be0f1f92c4e0dc852f651647a2c9b87f0eddefd8afbf401bd0536b` |
| `scripts\remap_artifacts_for_linux.sh` | 1,193 | `8b3eb7dd47ac42ca83b6d663d806c670386cf578489c607c3bdcc3e43873c7cb` |
| `scripts\remap_csv_paths.py` | 4,337 | `5fe33088d03781e2a957b52e9e4dac6c80309e7d38c02ff0abd55460af3fdb1b` |
| `tests\test_remap_csv_paths.py` | 2,254 | `b73f0b23864266188044888fbf58f23b289cdb1d9739ac160d942cf31726b1a7` |

### 切换状态与本机快照

- 状态：服务器迁移工具准备完成，但尚未停止本机、尚未传输数据、尚未在服务器启动正式任务。下一步是服务器安装工具/venv、获得 Git 代码和约 60–65 GiB 必需数据、路径转换、普通与 43.24 秒超长样本 smoke，然后才执行最终增量同步和切换。
- 本机继续运行：2026-09-01 10:19:34 +08:00，supervisor PID `94440`、worker PID `102176/67860` 均存活；train 匿名输出 62,491 / 572,951 条（10.906866%），3,666,375,699 字节。GPU 快照 79%、7,066/8,151 MiB、71°C、46.79 W。
- 服务器正式数据根规划为 `/public/home/wwwyyycom123_/datasets/corpora`，项目根规划为 `/public/home/wwwyyycom123_/multimodal_sv_reproduction`，venv 为 `/public/home/wwwyyycom123_/venvs/mmsv`；完整命令和切换检查表见 `SERVER_MIGRATION.md`。

### E30a：服务器基础工具安装与首次 Git 获取

- 结果接收时间：2026-09-01 10:28 +08:00；发行版最终确认为 Ubuntu 22.04.3 LTS (Jammy Jellyfish)。
- `apt-get update` 从阿里云 Ubuntu mirror 成功更新索引；NVIDIA CUDA 官方源发生连接超时，但 apt 使用已有索引继续执行，没有阻塞本阶段工具安装，也没有改动当前可用的 GPU 驱动。
- 安装成功：`curl 7.81.0`、`rsync 3.2.7`、`tmux 3.2a`、`ffmpeg 4.4.2`、`libsndfile1 1.0.31`；Git 从 2.34.1-1ubuntu1.11 更新到 2.34.1-1ubuntu1.17。
- 首次 `git clone --recurse-submodules https://github.com/wwwYYYcom/multimodal-SV.git` 失败：`GnuTLS recv error (-110): The TLS connection was non-properly terminated`。因此 `/public/home/wwwyyycom123_/multimodal_sv_reproduction` 尚未形成有效 Git worktree；后续 `rev-parse` 的“目录不存在”是该失败的连带结果。
- 修正方案：不关闭 TLS 校验；在目标持久化目录执行 `git init`，使用 HTTP/1.1、`http.maxRequests=1` 和 `fetch --depth=1` 最多重试 5 次，成功后 checkout `FETCH_HEAD`，再以单 job 获取 StreamVoiceAnon submodule。命令已写入 `SERVER_MIGRATION.md`。
- 本机没有因服务器准备而停止：同一时刻 supervisor/worker PID `94440/102176/67860` 均存活；2026-09-01 10:28:42 +08:00 已有 62,702 / 572,951 条（10.943693%），3,679,413,298 字节。

### E30b：可重试 Git 获取成功与环境安装器

- 服务器按 E30a 修正方案首次尝试即成功：主仓库 shallow fetch/checkout 得到 `3b922afb33be44719000f2d14b8d9fa5c45ac592`；StreamVoiceAnon submodule 首次获取成功并固定为 `201705182c045298225071481e7cd59d537e935e`。Linux 多 GPU runner 的 executable bit 已在服务器核验为 `-rwxrwxr-x`。
- 新增持久化环境安装器 `scripts/setup_linux_server_env.sh`：目标 venv `/public/home/wwwyyycom123_/venvs/mmsv`；PyTorch/torchvision/torchaudio 从官方 cu128 index 以 10 次重试和 120 秒 timeout 安装，其余依赖固定为本机成功版本。脚本随后运行全工程 pytest、StreamVoice `InferenceWrapper` import，并写 `results/runs/server_setup/environment.json`。
- 安装器验收条件：Python 3.10–3.12、`torch==2.9.1+cu128`、`torchaudio==2.9.1+cu128`、CUDA available、恰好 2 张 GPU；audit 记录完成时间、hostname、platform、venv、关键版本、GPU 名称/显存与 Git HEAD。任何条件不满足均非零退出。
- 实现提交：`7bc526a3cba219405ce17e9212fd5b9b0f58b490`（`feat: add reproducible Linux environment setup`）；Bash 语法检查、全部 `25 passed` 和 `git diff --check` 通过。
- 文件指纹：`scripts\setup_linux_server_env.sh` 3,760 字节，SHA-256 `4a15198716fd3bedafca464f2343d2e4f4dfc6583c3e9aab2dc35b3c2314639e`；更新后的 `SERVER_MIGRATION.md` 7,674 字节，SHA-256 `ea197579e80f0882de5f7994c7418e876de5b2170bdd2d149d306dab46918fd0`。
- 本机任务仍未停止：2026-09-01 10:34:16 +08:00 已生成 62,842 / 572,951 条（10.968128%），3,687,267,801 字节。

### E30c：服务器 Python/CUDA 环境验收完成

- 完成时间：audit 为 `2026-09-01T03:16:24.272473+00:00`，即 2026-09-01 11:16:24.272473 +08:00。
- 输出：`/public/home/wwwyyycom123_/multimodal_sv_reproduction/results/runs/server_setup/environment.json`；安装日志：同目录 `setup.log`；Python executable：`/public/home/wwwyyycom123_/venvs/mmsv/bin/python`。
- Git HEAD：`efa563cc0925b9c1c542c73865c0940318dac916`；hostname `worker-0`；platform `Linux-5.14.0-284.11.1.el9_2.x86_64-x86_64-with-glibc2.35`；Python `3.10.12`，venv 位于持久化 NFS。
- 关键版本：PyTorch `2.9.1+cu128`、PyTorch CUDA `12.8`、torchaudio `2.9.1+cu128`、transformers `4.56.2`、NumPy `1.26.4`、SciPy `1.13.1`、soundfile `0.13.1`。
- GPU 验收：`cuda_available=true`、`cuda_device_count=2`；GPU 0/1 均为 NVIDIA GeForce RTX 4090 D，每张 `50,866,487,296` bytes（47.37 GiB）可见显存。
- 软件验证：全工程 `25 passed`；StreamVoiceAnon `InferenceWrapper` import 成功；安装器最终输出 `server_environment_ready=true`。环境层面已满足双 GPU smoke 和正式匿名化要求。
- 下一阶段：传输 StreamVoiceAnon 5 个正式 checkpoint（合计主要文件 1,489,631,664 字节）、WavLM-Large snapshot、Fisher Part 1、LibriSpeech train-clean-360、metadata/plan/trials、86,222 条 evaluation 匿名语音、corrected lazy checkpoint 和切换时最新 train 匿名断点；随后执行路径 remap 与两类 smoke。
- 本机继续运行快照：2026-09-01 11:18:00 +08:00，64,113 / 572,951 条（11.189962%），3,751,585,675 字节；未因服务器环境安装而暂停。

### E30d：服务器传输链路诊断

- 服务器侧：`/usr/sbin/sshd` 存在，listener 进程正常，配置为 `0.0.0.0:22` 且支持 public-key authentication；持久化目录 `/public/home/wwwyyycom123_` 权限归当前用户，NFS 约 13 TiB 可用。
- 容器 `hostname -I` 返回平台内部地址。本机于 2026-09-01 11:49 +08:00 对该地址执行只读 `Test-NetConnection -Port 22`，TCP 和 ping 均超时；因此该地址不能作为本机到 worker 的直接传输端点。
- 结论：必须从平台控制台取得 SSH gateway/forward hostname、映射端口和用户名，或使用平台持久化数据集/NFS 导入功能。浏览器逐文件上传不适用于约 60–65 GiB、二十余万个文件的迁移。
- 安全处置：诊断输出意外包含平台生成的明文登录凭证，已要求用户立即通过控制台或交互式 `passwd` 轮换；本工程代码、Markdown、日志摘要和 Git 历史均不记录该凭证或其哈希，后续只使用 SSH 公钥。
- 本机继续运行：2026-09-01 11:49:56 +08:00，64,961 / 572,951 条（11.337968%），3,800,875,709 字节。

### E30e：E-File 归档传输链路验收与第一批正式包

- 平台容器详情未提供外部 SSH 映射端口，但 E-File 已确认支持“本地上传”文件。为避免浏览器逐个处理二十余万个小文件，迁移改为稳定排序的 TAR 分包；每次只在本机 `C:\mmsv_transfer` 暂存一包，服务器完成 SHA-256、成员和解包验收后再生成下一包。
- 探针生成时间：2026-09-01 12:05 +08:00。本机文件 `C:\mmsv_transfer\mmsv_transfer_probe_20260901_1205.tar`，10,240 字节，SHA-256 `29ee1bdba02c258287e45f4b30770d34bb31d788fdabd0fa9786a4af1dd3609f`；唯一成员为 `SERVER_MIGRATION.md`。
- 服务器验收完成时间：2026-09-01 12:09 +08:00。服务器上传路径 `/public/home/wwwyyycom123_/mmsv_transfer_probe_20260901_1205.tar` 的 SHA-256 与本机完全一致；解包输出 `/public/home/wwwyyycom123_/incoming/probe_unpacked/SERVER_MIGRATION.md` 的 SHA-256 为 `c493fb56716feba37d97ed64241d44113f3a692dd61ad2817f8a9d4c9624ee4`，也与本机源文件一致。结论：E-File 文件上传、TAR 列表、服务器解包和端到端内容校验均通过。
- 新增运行代码 `scripts\create_server_transfer_pack.ps1`：输入 base directory、一个或多个输入路径、服务器 destination root、目标输入字节数和指定 part number；按相对路径稳定排序并计算全部分片，但只生成指定的一片，同时输出 `.tar`、`.files.txt`、`.audit.json` 和 `.sha256`。已有输出一律拒绝覆盖，以便断点迁移和审计。
- 分包器 smoke 于 2026-09-01 12:09:01 +08:00 通过：以 `SERVER_MIGRATION.md` 为输入生成 1 个 10,240 字节 TAR，archive SHA-256 与独立探针相同，成员列表只有 `SERVER_MIGRATION.md`；PowerShell AST 和 `git diff --check` 通过。代码文件 5,018 字节，SHA-256 `c6b8f422b469c1a48ff405152446607161c53d5110f0e3065f7a9bd2a3835ed4`。
- 第一批正式包生成完成时间：2026-09-01 12:07:00 +08:00。本机输出 `C:\mmsv_transfer\mmsv_01_streamvoice_checkpoints_20260901.tar`，1,489,640,960 字节，SHA-256 `62f63dec99e3f0c8f5aba6333b7e54aea3444e58fce34f0ffd027afd959fb434`。包内 `third_party/StreamVoiceAnon/pretrained_checkpoints/` 共 7 个文件，其中 5 个正式权重合计 1,489,631,664 字节，另有 2 个 Hugging Face 下载元数据文件；服务器目标为 `/public/home/wwwyyycom123_/multimodal_sv_reproduction/third_party/StreamVoiceAnon/pretrained_checkpoints/`。
- 第一批服务器验收结果于 2026-09-01 12:45 +08:00 收到：`asr_s2s_bsq_8192_causal_down_whisper.pth`、`campplus_cn_common.bin`、`dual_ar_delay_0_8.pth`、`firefly-gan-vq-fsq-8x1024-21hz-generator.pth`、`spark_speaker_encoder.pth` 五项逐文件 `sha256sum -c` 全部为 `OK`。当前状态更新为“服务器已解包并验收完成”。
- 第二批正式包生成完成时间：2026-09-01 12:45:25 +08:00。本机输出 `C:\mmsv_transfer\mmsv_02_project_inputs_20260901.part001-of-001.tar`，911,752,192 字节，SHA-256 `129ce50c5280e72497d27967b792a2abbb9daa8f56b28be82192ee5c759b518f`；包含 `artifacts\metadata`、`artifacts\anonymization`、`artifacts\trials` 和 `results\runs\audio_corrected_p1\last.pt` 共 21 个文件，源文件合计 911,736,013 字节，服务器目标为项目根 `/public/home/wwwyyycom123_/multimodal_sv_reproduction`。当前状态为“本机已生成并校验，等待 E-File 上传与服务器验收”。对应分包审计和成员清单位于同一 `C:\mmsv_transfer` 目录的 `.audit.json`、`.files.txt` 与 `.sha256`。
- 生成第一批包时本机任务未停止：2026-09-01 12:07:11 +08:00，65,374 / 572,951 条（11.410051%），3,827,849,755 字节；supervisor PID `94440`、worker PID `102176/67860` 仍存活。

### E30f：第二批验收与 WavLM-Large 有效缓存迁移

- 第二批服务器验收结果于 2026-09-01 13:03 +08:00 收到：整包 SHA-256 为 `129ce50c5280e72497d27967b792a2abbb9daa8f56b28be82192ee5c759b518f`，与本机一致；`train_all_utterances_plan.csv`、`librispeech_target_pool.csv`、`fisher_manifest.csv`、`speaker_splits.csv` 和 corrected `last.pt` 五个关键文件逐项为 `OK`。服务器 Git fast-forward 至 `cde402a6a9519b67e0735a3c36e45a5339f6696e`，第二批状态更新为“服务器已解包并验收完成”。
- WavLM 原缓存核查发现 7 个文件、2,132,310,639 字节，其中 `blobs\92063b9e...incomplete` 为未完成下载残留，870,318,080 字节；另有两个 `.no_exist` 零字节标记。首次计算出的 `mmsv_03_wavlm_large_20260901.part001-of-002.tar` 因包含该 `.incomplete` 文件而被拒绝使用；该 TAR 为 870,325,760 字节、SHA-256 `b690d43b8e822c59e16bd772f741a1064411975013831ef5c1a44799a5ed84e4`，只保留在本机暂存区且明确不得上传或解包。安全策略阻止了当场删除，因此总账保留其“废弃临时包”状态，后续统一清理。
- 修正原则：WavLM 只迁移 `refs/` 和 commit `c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c` 下的有效 snapshot，不迁移 `.incomplete` 或 `.no_exist`。有效文件共 4 个、1,261,992,559 字节。
- 第三批干净包完成时间：2026-09-01 13:04:27 +08:00。本机输出 `C:\mmsv_transfer\mmsv_03_wavlm_large_clean_20260901.part001-of-001.tar`，1,261,997,056 字节，SHA-256 `ef953e4e3a506e13eeacf71bc19ca5e9f0f6fbd61a785ca013ec4812f26a7f16`；服务器目标为 `/public/home/wwwyyycom123_/.cache/huggingface/hub`。当前状态为“本机已生成并校验，等待 E-File 上传与服务器验收”。
- 有效 snapshot 指纹：`config.json` 2,222 字节，SHA-256 `a3d8fe831aaf63d725b54a8ac36f3549cd4365c5086774b2c89cabbc6f9e129d`；`pytorch_model.bin` 1,261,990,257 字节，SHA-256 `fdee460e529396ddb2f8c8e8ce0ad74cfb747b726bc6f612e666c7c1e1963c9d`；`refs/main` 与 `refs/refs/pr/7` 各 40 字节，SHA-256 分别为 `a8658333c95584b73d326b1cb3a3fe0313a17d4daeab848813feca41829d09ee`、`08023d930d0cb7de60b2439fcf3ddd7b29987f3b509fb355b1cd3a7b10dbc9d7`。
- 本机任务继续运行：2026-09-01 13:03:35 +08:00，66,978 / 572,951 条（11.690005%），3,913,800,408 字节。

### E30g：WavLM 离线解析与 Fisher Part 1 transcript 包

- WavLM 服务器离线解析结果于本轮收到：在 `HF_HOME=/public/home/wwwyyycom123_/.cache/huggingface`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 下，`AutoConfig.from_pretrained("microsoft/wavlm-large", local_files_only=True)` 成功返回 `model_type=wavlm`、`hidden_size=1024`、`num_hidden_layers=24` 和 `wavlm_cache_ready=true`。这证明 refs、snapshot commit 与 Transformers 离线解析路径有效；第三批整包和 `pytorch_model.bin` 的服务器 SHA 输出仍需单独归档，不能仅以 config 解析替代权重完整性证据。
- 首次 transcript 打包因唯一非 ASCII 附加说明文件 `Fisher_English_数据集详细说明.txt` 触发 Windows bsdtar `Can't convert a path to a wchar_t string` 而非零退出。未完成 TAR 为 `C:\mmsv_transfer\mmsv_04_fisher_p1_transcripts_20260901.part001-of-001.tar`，296,184,832 字节，SHA-256 `87a945fdfa2586f13478d583fbf9db53442ef5c9e2948a37abd503fb859ead6a`；它只保留在本机暂存区，明确不得上传或解包。源 corpus 与本机正式任务未被修改。
- `scripts\create_server_transfer_pack.ps1` 新增 `-ExcludeRelativePath`，执行大小写不敏感的精确相对路径排除，并在 audit 写入 `excluded_file_count` 和完整排除列表；修正后文件 5,611 字节，SHA-256 `4faa15a84b895c56c103a9ce656ee7257f006445f2c906b042dc1f83d8fae0bc`。排除仅限上述附加中文说明，不涉及 transcript、calldata、LDC 文档或任何训练/评测输入。
- 第四批可上传包完成时间：2026-09-01 13:28:30 +08:00。本机输出 `C:\mmsv_transfer\mmsv_04_fisher_p1_transcripts_runtime_20260901.part001-of-001.tar`，296,188,416 字节，SHA-256 `68377eace95962cff93c6cebeaf2ab1350bf1d99da6ee13fffe24744ae197fa0`；共 31,236 个成员、源文件 272,191,172 字节，`tar -tf` 返回 31,236 行且退出码为 0。服务器目标为 `/public/home/wwwyyycom123_/datasets/corpora`，当前状态为“本机已生成并校验，等待上传与服务器验收”。
- 第四批关键文件指纹：`doc/fe_03_p1_calldata.tbl` 471,434 字节，SHA-256 `a612620cb4e3d47818524ebe989bd4c0b14294cb2422dfa64b8a2f1fbbba3fda`；`data/trans/000/fe_03_00001.txt` 13,705 字节，SHA-256 `19e18082c4123f9e60d0f35d5cbc03ef8fa4f79288c0b03c52029b3da19edaac`；corpus `index.html` 2,825 字节，SHA-256 `a5904be0f0ea5985d2e1c98f1df41a8483442d9758b04cf97093fec571ec1015`。

### E30h：Fisher transcript 验收与原始音频分片启动

- 第四批服务器验收结果于 2026-09-01 13:39 +08:00 收到：整包 SHA-256 `68377eace95962cff93c6cebeaf2ab1350bf1d99da6ee13fffe24744ae197fa0` 与本机一致；上传 TAR 成员数和服务器解包文件数均为 31,236；calldata、代表 transcript 和 `index.html` 三项 SHA-256 全部为 `OK`。第四批状态更新为“服务器已解包并验收完成”。
- 第五批 Fisher Part 1 原始音频按 1,900,000,000 目标输入字节计算为 16 片；全目录 5,879 个文件、30,040,328,697 字节，其中包括 5,850 个 SPHERE calls 及 LDC 目录附属文件。archive 成员保持相对于 `/public/home/wwwyyycom123_/datasets/corpora` 的完整 corpus 结构。
- 第 1/16 片完成时间：2026-09-01 13:39:11 +08:00。本机输出 `C:\mmsv_transfer\mmsv_05_fisher_p1_audio_20260901.part001-of-016.tar`，1,895,350,272 字节，SHA-256 `6191a3593aa68acb888c8600a600fe559ef6e1ba6be1265ff84e271bd7ad9594`；包含 329 个文件、源字节 1,895,093,518，首成员为 corpus `.DS_Store`，末成员为 `audio/003/fe_03_00327.sph`；`tar -tf` 精确返回 329 行。当前状态为“本机已生成并校验，等待上传与服务器验收”。
- 生成第 1 片时本机正式任务仍运行：2026-09-01 13:39:15 +08:00，67,902 / 572,951 条（11.851275%），3,968,577,285 字节。

### E30i：Fisher 原始音频第 1 片验收与第 2 片

- 第 1/16 片服务器验收结果于 2026-09-01 15:04 +08:00 收到：整包 SHA-256 `6191a3593aa68acb888c8600a600fe559ef6e1ba6be1265ff84e271bd7ad9594` 与本机一致；TAR 成员 329、服务器累计文件 329、累计 SPHERE 327；`fe_03_00001.sph` 和 `fe_03_00327.sph` 抽样 SHA-256 均为 `OK`。第 1 片状态更新为“服务器已解包并验收完成”。
- 第 2/16 片生成完成时间：2026-09-01 15:04:35 +08:00。本机输出 `C:\mmsv_transfer\mmsv_05_fisher_p1_audio_20260901.part002-of-016.tar`，1,900,129,792 字节，SHA-256 `e629d71a5b725143db2d5f902983fa0486ec53d38484efae002c3130b1cedba2`；包含 380 个文件且全部为 SPHERE，源字节 1,899,834,340，范围为 `audio/003/fe_03_00328.sph` 至 `audio/007/fe_03_00707.sph`；`tar -tf` 返回 380 行。当前状态为“本机已生成并校验，等待上传与服务器验收”。
- 第 2 片首末抽样：`fe_03_00328.sph` 6,555,525 字节，SHA-256 `fa3427e415665f257dcfa8b47a62d069fcb23369c08cb79f91a3ef3baee8336b`；`fe_03_00707.sph` 5,068,749 字节，SHA-256 `e4b643192a44c1b57445465a038cd399402aecdb715322dd8589b14ce4a050ab`。
- 生成时本机正式任务继续运行：2026-09-01 15:04:39 +08:00，70,300 / 572,951 条（12.269810%），4,100,646,715 字节。
