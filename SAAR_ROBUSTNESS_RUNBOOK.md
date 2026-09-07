# SAAR cross-session A-A and robustness evaluation

Run after the original reproduction and session-fixed baseline have completed.
No SAAR training is started. Preserve the original trial files and mapping.

`scripts/saar_robustness.py --mode prepare` derives 3,752 enrollment utterances
missing from the 66,712 target outputs, giving a 70,464 utterance union. There
are 12 additional sessions. The original deterministic mapping (seed 2027,
sorted train-clean-360 reference pool) is checked against all 2,796 existing
sessions before extending it. Existing FLAC files are reused. Server preparation
uses the already remapped baseline plan as its path template; no local plan
upload is needed.

Server execution (after updating code):

```bash
cd /public/home/wwwyyycom123_/multimodal_sv_reproduction
mkdir -p results/runs/saar_robustness
nohup setsid bash scripts/run_saar_robustness.sh \
  </dev/null >results/runs/saar_robustness/launcher.log 2>&1 &
```

The shell runner uses a nonblocking file lock and refuses active MMSV GPU jobs.
It creates a new run directory each time; repeated invocations reuse FLACs,
but embedding and statistical computation are recomputed in the new directory.
Generation uses one worker on each requested GPU (default 0,1,2,3).

Pipeline: prepare and audit mapping, generate missing enrollment audio, merge
and validate the 70,464-row manifest, extract extra lazy embeddings, evaluate
lazy O-A/A-A, extract all anonymous embeddings using the completed semi-informed
checkpoint, evaluate semi-informed-transfer O-A/A-A, record results in the
server EXPERIMENT_RESULTS.md and create a TAR plus SHA256 under the parent home.
Original semi-informed enrollment embeddings must already exist at
`artifacts/embeddings/original_evaluation_semi_corrected.npz`.

The transfer attacker was trained on utterance-random anonymous speech. Do not
describe it as a fully adapted session-fixed attacker. Original and anonymous
embeddings for each evaluation must use the same checkpoint.

Both conditions use identical five-seed, call-disjoint trials, fixed enrollment
15 and nested target N=1/2/5/10/15. A-A labels refer to real source identities,
not reference identities. Different sessions use their independent deterministic
reference selection; no reference is forced equal for a genuine pair.

Statistical outputs include 1,000 replicate samples for each of two methods:

- Enrollment-speaker cluster bootstrap, jointly across all seeds and N.
- Shared-speaker dyadic sensitivity bootstrap: a speaker is resampled once in
  both roles; genuine trial weight is its multiplicity, impostor trial weight
  is the product of both speakers' multiplicities. This accounts for both-role
  reuse but is conditional on the fixed observed trial graph.

These CIs do not cover new reference mappings or new trained checkpoints.
Weighted ROC groups tied scores at a common threshold. Original baseline point
metrics remain available for comparison. No automatic Gate-to-training decision.

Monitor (Ctrl+C only exits watch):

```bash
watch -n 60 'cd /public/home/wwwyyycom123_/multimodal_sv_reproduction; tail -n 15 "$(cat results/runs/saar_robustness/log.latest)"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader'
```

Detailed per-worker generation logs and `postprocess.log` are in the directory
named by `results/runs/saar_robustness/output.latest`. Download the final TAR and
SHA256 to the local results directory; preserve the server ledger as evidence
when importing instead of overwriting newer local ledger changes.
