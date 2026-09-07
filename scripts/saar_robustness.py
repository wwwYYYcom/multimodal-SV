"""Supplement the frozen SAAR baseline with cross-call A-A and clustered CIs.

Run from the project root. No training is performed and baseline outputs are
never replaced. GPU generation/extraction is only enabled by --mode run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mmsv.anonymization import _stable_reference_index
from mmsv.metrics import load_embeddings, score_session_trials, summarize_privacy_curve

BASE = Path('artifacts/saar/session_baseline')
NS = [1, 2, 5, 10, 15]
SEEDS = [1, 2, 3, 4, 5]


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def trials():
    result = {}
    for seed in SEEDS:
        with (BASE / f'manifests/sampling_seed_{seed}.jsonl').open(encoding='utf-8') as f:
            result[seed] = [json.loads(line) for line in f]
    return result


def prepare(out):
    trial_sets = trials()
    enroll, target = set(), set()
    for ts in trial_sets.values():
        for t in ts:
            assert t['enroll_session_id'].rsplit(':', 1)[0] != t['target_session_id'].rsplit(':', 1)[0]
            assert len(t['enroll_utt_ids']) == len(set(t['enroll_utt_ids'])) == 15
            assert len(t['target_utt_ids']) == len(set(t['target_utt_ids'])) == 15
            assert (t['enroll_speaker'] == t['target_speaker']) == bool(t['label'])
            enroll.update(t['enroll_utt_ids'])
            target.update(t['target_utt_ids'])
    old = read_csv(BASE / 'manifests/session_baseline_anonymization_plan.csv')
    assert {r['utt_id'] for r in old} == target
    by_session = {}
    for row in old:
        previous = by_session.setdefault(row['session_id'], row)
        assert previous['reference_utt_id'] == row['reference_utt_id']
        assert previous['reference_audio_path'] == row['reference_audio_path']
    missing = enroll - target
    plan_audit = json.loads((BASE / 'manifests/session_baseline_anonymization_plan.audit.json').read_text(encoding='utf-8'))
    references = sorted(read_csv('artifacts/metadata/librispeech_target_pool.csv'), key=lambda r:r['utt_id'])
    seed = int(plan_audit['seed'])
    assert len(references) == plan_audit['reference_pool_utterances']
    for session, template in by_session.items():
        assert references[_stable_reference_index(seed, session, len(references))]['utt_id'] == template['reference_utt_id']
    corpus_root = old[0]['audio_path'].replace('\\', '/').split('/fisher/', 1)[0]
    audio_root = old[0]['output_audio_path'].replace('\\', '/').rsplit('/', 2)[0]
    added_sessions = set()
    source = {r['utt_id']: r for r in read_csv('artifacts/metadata/fisher_manifest.csv') if r['utt_id'] in missing}
    assert set(source) == missing
    extra = []
    for uid in sorted(missing):
        raw = source[uid]
        session = f"{raw['call_id']}:{raw['channel']}"
        if session not in by_session:
            reference = references[_stable_reference_index(seed, session, len(references))]
            assert float(reference['duration']) > 4
            template = dict(old[0])
            template.update(speaker_id=raw['speaker_id'],
                audio_path=corpus_root + '/fisher/' + raw['audio_path'].replace('\\','/').split('/fisher/',1)[1],
                reference_utt_id=reference['utt_id'], reference_speaker_id=reference['speaker_id'],
                reference_duration=reference['duration'],
                reference_audio_path=corpus_root + '/LibriSpeech/' + reference['audio_path'].replace('\\','/').split('/LibriSpeech/',1)[1],
                output_audio_path=audio_root + '/' + raw['speaker_id'] + '/' + uid + '.flac')
            by_session[session] = template
            added_sessions.add(session)
        template = by_session[session]
        row = dict(template)
        for key in ('utt_id', 'speaker_id', 'call_id', 'channel', 'start', 'end', 'duration', 'transcript'):
            row[key] = raw[key]
        assert row['speaker_id'] == template['speaker_id']
        # Same call-side source file; this also preserves already remapped Linux paths.
        row['session_id'] = session
        old_path = template['output_audio_path'].replace('\\', '/')
        row['output_audio_path'] = old_path.rsplit('/', 1)[0] + '/' + uid + '.flac'
        extra.append(row)
    write_csv(out / 'missing_enrollment_plan.csv', extra)
    write_csv(out / 'union_plan.csv', old + extra)
    audit = dict(enrollment=len(enroll), target=len(target), intersection=len(enroll & target),
                 missing_enrollment=len(missing), union=len(enroll | target),
                 mapping_preserved=True, new_sessions=len(added_sessions), mapping_seed=seed,
                 reference_pool_sha256=sha('artifacts/metadata/librispeech_target_pool.csv'),
                 baseline_plan_sha256=sha(BASE / 'manifests/session_baseline_anonymization_plan.csv'))
    dump(out / 'preparation.json', audit)
    print(json.dumps(audit), flush=True)
    return audit


def weighted_eer(labels, scores, weights):
    """Threshold-grouped ROC: tied scores cannot be split by sort order."""
    labels, scores, weights = map(np.asarray, (labels, scores, weights))
    order = np.argsort(-scores, kind='stable')
    y, s, w = labels[order], scores[order], weights[order]
    pos, neg = (w * (y == 1)).sum(), (w * (y == 0)).sum()
    if min(pos, neg) <= 0:
        raise ValueError('Bootstrap replicate has an empty class')
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    fnr = np.r_[1., 1 - np.cumsum(w * (y == 1))[ends] / pos]
    fpr = np.r_[0., np.cumsum(w * (y == 0))[ends] / neg]
    diff = fpr - fnr
    i = np.flatnonzero(diff >= 0)[0]
    if i == 0:
        return float(fpr[0])
    alpha = -diff[i - 1] / (diff[i] - diff[i - 1])
    return float(fpr[i - 1] + alpha * (fpr[i] - fpr[i - 1]))


def cluster_weights(ts, speaker_index, multiplicity, method):
    a = np.array([multiplicity[speaker_index[t['enroll_speaker']]] for t in ts])
    if method == 'enrollment_speaker':
        return a
    b = np.array([multiplicity[speaker_index[t['target_speaker']]] for t in ts])
    # Shared speaker resample in both roles; genuine pairs are one cluster.
    return np.where(np.array([t['label'] for t in ts]) == 1, a, a * b)


def bootstrap(score_root, ts_by_seed, condition, out, replicates):
    speakers = sorted({t[k] for ts in ts_by_seed.values() for t in ts for k in ('enroll_speaker', 'target_speaker')})
    index = {s: i for i, s in enumerate(speakers)}
    data = {}
    prefix = condition.replace('-', '').lower()
    for seed, ts in ts_by_seed.items():
        for n in NS:
            rows = read_csv(score_root / f'{prefix}_mean_N{n}_seed{seed}.csv')
            assert [r['trial_id'] for r in rows] == [t['trial_id'] for t in ts]
            assert [int(r['label']) for r in rows] == [t['label'] for t in ts]
            data[seed, n] = np.array([float(r['score']) for r in rows])
    output = {}
    for method in ('enrollment_speaker', 'shared_speaker_dyadic'):
        rng = np.random.default_rng(2027)
        values = np.empty((replicates, len(NS)))
        for rep in range(replicates):
            m = rng.multinomial(len(speakers), np.full(len(speakers), 1 / len(speakers)))
            per_seed = []
            for seed, ts in ts_by_seed.items():
                labels = np.array([t['label'] for t in ts])
                w = cluster_weights(ts, index, m, method)
                per_seed.append([weighted_eer(labels, data[seed, n], w) for n in NS])
            values[rep] = np.mean(per_seed, axis=0) * 100
            if rep % 100 == 0:
                print(f'bootstrap {condition} {method} {rep}/{replicates}', flush=True)
        delta = values[:, 0] - values[:, -1]
        output[method] = dict(replicates=replicates, seed=2027, clusters=len(speakers),
            delta_mean_pp=float(delta.mean()), delta_ci95_pp=np.quantile(delta, [.025, .975]).tolist(),
            per_n=[dict(n=n, mean=float(values[:, i].mean()), ci95=np.quantile(values[:, i], [.025, .975]).tolist()) for i, n in enumerate(NS)])
        write_csv(out / f'{method}_replicates.csv', [dict(replicate=i, **{f'eer_n{n}':float(row[j]) for j,n in enumerate(NS)}, delta_pp=float(delta[i])) for i,row in enumerate(values)])
    output['limitations'] = ('Conditional on fixed trial graph, one pseudo mapping and fixed attacker. '
        'Enrollment-only clusters do not capture reused impostor targets; dyadic resampling is a sensitivity analysis, '
        'not a guarantee of full population coverage. Same speaker weights are shared across all five seeds and N.')
    dump(out / 'cluster_bootstrap.json', output)
    return output


def evaluate(out, attacker, original_path, anonymous_paths, conditions, replicates):
    original = load_embeddings(original_path)
    anonymous = {}
    for path in anonymous_paths:
        part = load_embeddings(path)
        if anonymous.keys() & part.keys():
            raise ValueError('Overlapping embedding input shards')
        anonymous.update(part)
    ts = trials()
    for matrix in (original, anonymous):
        if any(not np.isfinite(v).all() for v in matrix.values()):
            raise ValueError('Non-finite embeddings')
    root = out / attacker
    root.mkdir(parents=True, exist_ok=True)
    provenance = dict(checkpoint='results/runs/audio_corrected_p1/last.pt' if attacker == 'lazy' else 'results/runs/audio_semi_corrected/last.pt',
        original=dict(path=str(original_path), sha256=sha(original_path)),
        anonymized=[dict(path=str(p), sha256=sha(p)) for p in anonymous_paths])
    checkpoint = Path(provenance['checkpoint'])
    provenance['checkpoint_sha256'] = sha(checkpoint) if checkpoint.is_file() else None
    result = dict(attacker=attacker, inputs=provenance, conditions={})
    for condition in conditions:
        score_root = root / condition / 'scores'
        metric_paths = []
        for seed in SEEDS:
            for n in NS:
                path = score_root / f"{condition.replace('-', '').lower()}_mean_N{n}_seed{seed}.csv"
                score_session_trials(BASE / f'manifests/sampling_seed_{seed}.jsonl', original, anonymous, condition, n, path)
                metric_paths.append(path.with_suffix('.metrics.json'))
        summary = summarize_privacy_curve(metric_paths, root / condition / 'privacy.csv', system=f'Session-fixed {condition}',
            attacker=attacker, checkpoint=provenance['checkpoint'], git_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
        ci = bootstrap(score_root, ts, condition, root / condition, replicates)
        result['conditions'][condition] = dict(summary=summary, clustered_ci=ci)
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    dump(root / 'summary.json', result)
    return result


def run_command(args, gpu, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with Path(log).open('a', encoding='utf-8') as f:
        subprocess.run([sys.executable, *map(str, args)], env=env, stdout=f, stderr=f, check=True)


def run(out, gpus, replicates):
    for required in ('results/runs/audio_corrected_p1/last.pt',
                     'results/runs/audio_semi_corrected/last.pt',
                     'artifacts/embeddings/original_evaluation_semi_corrected.npz'):
        if not Path(required).is_file():
            raise FileNotFoundError(required)
    audit = prepare(out)
    for row in read_csv(out/'union_plan.csv'):
        for key in ('audio_path', 'reference_audio_path'):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(f'{key}: {row[key]} (check server path mapping)')
    workers = []
    handles = []
    manifests = []
    try:
        for i, gpu in enumerate(gpus):
            start = audit['missing_enrollment'] * i // len(gpus)
            stop = audit['missing_enrollment'] * (i + 1) // len(gpus)
            manifest = out / f'worker{i}.csv'
            manifests.append(manifest)
            f = (out / f'worker{i}.log').open('a', encoding='utf-8')
            handles.append(f)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
            workers.append(subprocess.Popen([sys.executable, '-u','-m','mmsv.cli','anonymize-streamvoice',
                '--plan',str(out/'missing_enrollment_plan.csv'),'--output-manifest',str(manifest),
                '--streamvoice-root','third_party/StreamVoiceAnon','--delay','2','--alpha','1.0',
                '--start-index',str(start),'--limit',str(stop-start)], env=env, stdout=f, stderr=f))
        for p in workers:
            if p.wait() != 0:
                raise RuntimeError('Anonymization worker failed; see worker logs, existing FLACs retained')
    finally:
        for p in workers:
            if p.poll() is None:
                p.terminate()
                p.wait()
        for f in handles:
            f.close()
    cli_log = out/'postprocess.log'
    run_command(['scripts/merge_anonymization_manifests.py','--plan',out/'missing_enrollment_plan.csv',
        '--manifests',*manifests,'--output',out/'extra_manifest.csv'],gpus[0],cli_log)
    run_command(['scripts/merge_anonymization_manifests.py','--plan',out/'union_plan.csv',
        '--manifests',BASE/'manifests/session_baseline_anonymized_manifest.csv',out/'extra_manifest.csv',
        '--output',out/'union_manifest.csv'],gpus[0],cli_log)
    run_command(['scripts/validate_anonymization_outputs.py','--plan',out/'union_plan.csv','--manifest',out/'union_manifest.csv',
        '--expected',audit['union'],'--finite-check-limit',100,'--output',out/'validation.json'],gpus[0],cli_log)
    def extract(checkpoint, manifest, destination, gpu):
        run_command(['-m','mmsv.cli','extract-embeddings','--checkpoint',checkpoint,'--manifest',manifest,
                     '--output',destination],gpu,cli_log)
    extract('results/runs/audio_corrected_p1/last.pt',out/'extra_manifest.csv',out/'lazy_extra.npz',gpus[0])
    evaluate(out,'lazy',Path('artifacts/embeddings/original_evaluation_corrected.npz'),
             [BASE/'embeddings/anonymized_evaluation_corrected.npz',out/'lazy_extra.npz'],['O-A','A-A'],replicates)
    # This attacker was adapted on utterance-random anonymous training audio.
    # Its results measure transfer to session-fixed anonymization, not a fully matched attacker.
    checkpoint = Path('results/runs/audio_semi_corrected/last.pt')
    if not checkpoint.is_file():
        raise FileNotFoundError('Semi-informed checkpoint required for the second attacker')
    extract(checkpoint,out/'union_manifest.csv',out/'semi_union.npz',gpus[-1])
    evaluate(out,'semi_transfer',Path('artifacts/embeddings/original_evaluation_semi_corrected.npz'),
             [out/'semi_union.npz'],['O-A','A-A'],replicates)
    hashes = {str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.suffix in ('.json','.csv','.npz')}
    dump(out/'file_hashes.json',hashes)
    with Path('EXPERIMENT_RESULTS.md').open('a', encoding='utf-8') as ledger:
        ledger.write('\n\n### SAAR robustness server completion\n\n')
        ledger.write('Completed UTC: ' + datetime.now(timezone.utc).isoformat() + '\n\n')
        ledger.write('Run code: `scripts/saar_robustness.py`; outputs: `' + str(out) + '`\n\n')
        for attacker in ('lazy', 'semi_transfer'):
            payload = (out/attacker/'summary.json').read_text(encoding='utf-8')
            ledger.write('```json\n' + payload + '```\n\n')
        ledger.write('File SHA256 inventory: `' + str(out/'file_hashes.json') + '`\n')
    print('robustness_pipeline_completed=' + datetime.now(timezone.utc).isoformat(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['prepare','local-oa','run'], required=True)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; keep prior experiments intact')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--replicates', type=int, default=1000)
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error('replicates must be positive')
    if args.output.exists():
        parser.error('Output directory already exists; select a new run directory (audio cache is still reused)')
    args.output.mkdir(parents=True)
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'local-oa':
        evaluate(args.output,'lazy',Path('artifacts/embeddings/original_evaluation_corrected.npz'),
            [BASE/'embeddings/anonymized_evaluation_corrected.npz'],['O-A'],args.replicates)
    else:
        if os.name == 'nt':
            parser.error('run mode is intended for the Linux server with all audio/checkpoints')
        gpus = args.gpus.split(',')
        if len(gpus) != len(set(gpus)) or any(not g.isdigit() for g in gpus):
            parser.error('GPU IDs must be unique integers')
        run(args.output,gpus,args.replicates)


if __name__ == '__main__':
    main()
