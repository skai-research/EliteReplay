"""Random IID retention with production descriptors and export re-verification.

Descriptors are assigned only after independent generation; they never condition
sampling or rank retention. Failed verification/label gates trigger deterministic
refill from the same finite IID pool, never evolutionary candidates.
"""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .config import offline_dir, path, fingerprint
from .offline import atomic_json, select_iid, qd_config
from elitereplay.EliteReplay.providers import make_litellm_labeler
from elitereplay.EliteReplay.schema import PipelineConfig, Taxonomy
from elitereplay.EliteReplay.subtypes import cell_id, structural_subtype


def export_verified_iid(c, records, by_id, adapter, *, labeler=None):
    taxonomy = Taxonomy.from_dict(json.loads(path(c['offline']['iid_taxonomy_path']).read_text()))
    if taxonomy.domain != c['domain']:
        raise ValueError('IID taxonomy domain mismatch')
    root = offline_dir(c)
    cache = root / 'export_checks'
    cache.mkdir(exist_ok=True)
    if labeler is None:
        raw = qd_config(c)
        labeler = make_litellm_labeler(config=PipelineConfig.from_mapping(raw), settings=raw['providers']['labeler'])
    def inspect(record):
        cid, pid, code = record['candidate_id'], record['problem_id'], record['canonical_response']
        binding = fingerprint([taxonomy.hash, record, c['offline']['label_confidence']])
        file = cache / f'{cid}.json'
        if file.exists():
            result = json.loads(file.read_text())
            if result['binding'] != binding:
                raise ValueError('IID export check inputs changed')
            return result
        result = {'binding': binding, 'candidate_id': cid, 'accepted': False}
        if adapter.sanitize_for_sft(code) != code:
            raise ValueError('IID SFT sanitizer changed the verified artifact')
        verified = adapter.verify(by_id[pid], code)
        if not verified.executable or not verified.fully_correct:
            result['reason'] = 'export_reverification_failed'
        else:
            decision = labeler.classify(problem_id=pid,
                labeler_input=adapter.format_labeler_input(by_id[pid], code), taxonomy=taxonomy,
                seed=int(fingerprint([c['seed'], cid, 'classify'])[:8], 16), deterministic=False)
            if decision.confidence < c['offline']['label_confidence']:
                decision = labeler.classify(problem_id=pid,
                    labeler_input=adapter.format_labeler_input(by_id[pid], code), taxonomy=taxonomy,
                    seed=int(fingerprint([c['seed'], cid, 'classify_retry'])[:8], 16), deterministic=True)
            taxonomy.require_label(decision.primary_strategy or '')
            if decision.secondary_strategy is not None:
                taxonomy.require_label(decision.secondary_strategy)
            if decision.confidence < c['offline']['label_confidence']:
                result['reason'] = 'label_confidence_below_threshold'
            else:
                subtype = structural_subtype(c['domain'], code)
                result.update(accepted=True, metadata={
                    'primary_strategy': decision.primary_strategy, 'structural_subtype': subtype,
                    'mode_id': cell_id(decision.primary_strategy, subtype),
                    'label_confidence': decision.confidence, 'taxonomy_hash': taxonomy.hash})
        atomic_json(file, result)
        return result
    excluded, inspected = set(), {}
    target = c['offline']['export_target']
    while True:
        chosen = select_iid([r for r in records if r['candidate_id'] not in excluded],
            domain=c['domain'], seed=c['seed'], target=target,
            per_question_cap=c['offline']['per_question_cap'])
        if len(chosen) != target:
            raise RuntimeError(f'IID has {len(chosen)} eligible examples, target {target}; no duplicate filling')
        pending = [r for r in chosen if r['candidate_id'] not in inspected]
        with ThreadPoolExecutor(max_workers=min(c['offline']['label_workers'], c['offline']['verifier_workers'])) as pool:
            for result in pool.map(inspect, pending):
                inspected[result['candidate_id']] = result
                if not result['accepted']:
                    excluded.add(result['candidate_id'])
        if all(r['candidate_id'] not in excluded for r in chosen):
            break
    exports = [{'messages': adapter.format_prompt(by_id[r['problem_id']]) +
        [{'role': 'assistant', 'content': r['canonical_response']}], 'metadata': {
        'problem_id': r['problem_id'], 'candidate_id': r['candidate_id'],
        'archive_method': 'iid', 'fully_correct': True, 'sample_source': 'frozen_sft',
        'response_tokens': r['generation']['output_tokens'],
        **inspected[r['candidate_id']]['metadata']}} for r in chosen]
    counts = Counter(r['problem_id'] for r in chosen)
    audit = {'selection': 'random_correct_deduplicated_question_balanced',
        'proposals': len(records), 'attempted_questions': len(by_id), 'examples': len(exports),
        'represented_questions': len(counts), 'max_per_question': max(counts.values()),
        'export_rejected': len(excluded), 'taxonomy_hash': taxonomy.hash,
        'question_mode_pairs': len({(r['metadata']['problem_id'], r['metadata']['mode_id']) for r in exports}),
        'export_verified': True, 'independent_generation': True}
    return exports, audit
