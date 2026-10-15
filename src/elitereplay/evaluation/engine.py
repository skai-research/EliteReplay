"""Restartable direct sampling and full-split pass@k, without search or reranking."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import math
import time
from pathlib import Path

from tqdm.auto import tqdm

from .config import KS, atomic_json, digest, sha256


def pass_at_k(n, correct, k):
    if any(type(v) is not int for v in (n, correct, k)) or not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("pass@k requires 0 <= correct <= n and 1 <= k <= n")
    return 1.0 - math.comb(n - correct, k) / math.comb(n, k) if n - correct >= k else 1.0


def sample_seed(seed, problem_id, sample_id):
    return int(digest([seed, problem_id, sample_id])[:8], 16) % (2**31)


def adapter(c, output):
    v = c["verifier"]
    if c["domain"] == "bird":
        from elitereplay.archive.bird import BirdAdapter
        return BirdAdapter(max_response_bytes=v["max_response_bytes"])
    if c["domain"] == "spider":
        from .spider import SpiderAdapter
        return SpiderAdapter(database_root=v["spider_database_root"],
                             timeout=v["spider_timeout_seconds"], max_response_bytes=v["max_response_bytes"])
    raise ValueError("This release supports BIRD and Spider evaluation")


class Generator:
    def __init__(self, c, name, verifier, items):
        from transformers import AutoTokenizer
        from vllm import LLM
        self.c, self.tokens = c, {}
        model = c["models"][name]["path"]
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        # Audit every prompt before loading/generating; never truncate or drop it.
        for item in items:
            ids = tokenizer.apply_chat_template(verifier.format_prompt(item), tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if len(ids) > c["sampling"]["max_prompt_tokens"]:
                raise ValueError(f"Prompt overflow for {item['problem_id']}: {len(ids)} tokens; increase the common limit in a new protocol")
            self.tokens[item["problem_id"]] = ids
        r, s = c["runtime"], c["sampling"]
        self.model = LLM(model=model, tokenizer=model, trust_remote_code=True,
            tensor_parallel_size=r["tensor_parallel_size"], dtype="bfloat16",
            enforce_eager=r["enforce_eager"], enable_prefix_caching=r["enable_prefix_caching"],
            gpu_memory_utilization=r["gpu_memory_fraction"], max_num_seqs=r["max_num_seqs"],
            max_model_len=s["max_prompt_tokens"] + s["max_response_tokens"],
            seed=s["seed"],
            limit_mm_per_prompt={"image": 0, "video": 0})

    def generate(self, requests):
        from vllm import SamplingParams
        s = self.c["sampling"]
        parameters = [SamplingParams(n=1, seed=r["seed"], temperature=s["temperature"], top_p=s["top_p"],
            top_k=s["top_k"], max_tokens=s["max_response_tokens"], presence_penalty=0, repetition_penalty=1)
            for r in requests]
        outputs = self.model.generate([{"prompt_token_ids": self.tokens[r["problem_id"]]} for r in requests],
                                      parameters, use_tqdm=False)
        if len(outputs) != len(requests) or any(len(o.outputs) != 1 for o in outputs):
            raise RuntimeError("Generator returned the wrong number of samples")
        return [{"text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason,
                 "response_tokens": len(o.outputs[0].token_ids), "prompt_tokens": len(o.prompt_token_ids)} for o in outputs]


def score_sample(verifier, item, record):
    # A response hitting the cap is an attempted sample, never an omitted row.
    if record["finish_reason"] == "length":
        return {"correct": False, "error": "response_truncated", "executable": False}
    if record["finish_reason"] != "stop":
        raise RuntimeError(f"Unexpected generation finish reason: {record['finish_reason']}")
    parsed = verifier.parse_strict(record["text"])
    if not parsed.accepted:
        return {"correct": False, "error": f"parse:{parsed.error}", "executable": False}
    result = verifier.verify(item, parsed.canonical_response)
    # Verifier environment failures cannot become model failures in a table.
    error = result.error or ""
    if error.startswith(("worker_exit_", "malformed_worker_output", "worker:", "sandbox_exit_")):
        raise RuntimeError(f"Verifier infrastructure failure: {error}")
    return {"correct": bool(result.fully_correct), "executable": bool(result.executable),
            "error": result.error, "verifier_metadata": dict(result.metadata)}


def files_for(output, question_index, start):
    directory = Path(output) / "samples" / f"q{question_index:05d}"
    return directory / f"b{start:03d}.generated.json", directory / f"b{start:03d}.scored.json"


def block(c, item, start, count, records):
    return {"task_sha256": digest(item), "problem_id": item["problem_id"],
            "sample_start": start, "sample_count": count, "records": records}


def validate_block(c, item, start, count, value):
    if (value["task_sha256"], value["problem_id"], value["sample_start"], value["sample_count"]) != (digest(item), item["problem_id"], start, count):
        raise ValueError("Cached block has different question/protocol identities")
    records = value["records"]
    if len(records) != count:
        raise ValueError("Incomplete sample block")
    for sample_id, r in zip(range(start, start + count), records):
        if r["sample_id"] != sample_id or r["seed"] != sample_seed(c["sampling"]["seed"], item["problem_id"], sample_id):
            raise ValueError("Cached samples have incorrect IDs/seeds")
        if r["problem_id"] != item["problem_id"]:
            raise ValueError("Cached sample belongs to a different question")
        if not isinstance(r["text"], str) or r["finish_reason"] not in ("stop", "length"):
            raise ValueError("Invalid cached generation")
        if type(r["response_tokens"]) is not int or not 0 <= r["response_tokens"] <= c["sampling"]["max_response_tokens"]:
            raise ValueError("Invalid cached response length")
        if type(r["prompt_tokens"]) is not int or not 0 < r["prompt_tokens"] <= c["sampling"]["max_prompt_tokens"]:
            raise ValueError("Invalid cached prompt length")
    return records


def read_scores(scored, generated, count):
    value = json.loads(Path(scored).read_text())
    if value["generated_sha256"] != digest(generated) or len(value["scores"]) != count:
        raise ValueError("Scores do not belong to this complete generation block")
    if any(type(s.get("correct")) is not bool for s in value["scores"]):
        raise ValueError("Scores must be binary correctness judgments")
    return value["scores"]


def evaluation_items(c, items, smoke):
    if not smoke:
        return items
    ids = c["runtime"].get("smoke_problem_ids")
    if ids is None:
        return items[:2]
    by_id = {item["problem_id"]: item for item in items}
    if any(pid not in by_id for pid in ids):
        raise ValueError("Smoke problem IDs must belong to the prepared official evaluation set")
    return [by_id[pid] for pid in ids]


class VerificationQueue:
    """Bound candidate work and persist ready blocks without FIFO barriers."""

    def __init__(self, pool, verifier, limit, failure_path):
        self.pool, self.verifier, self.limit = pool, verifier, limit
        self.failure_path = failure_path
        self.pending = {}
        self.peak = self.completed = 0
        self.failed = []
        self.started = time.monotonic()

    def drain(self, block=False):
        if not self.pending:
            return
        if block:
            wait(self.pending, return_when=FIRST_COMPLETED)
        for future in list(self.pending):
            if not future.done():
                continue
            state, index, record = self.pending.pop(future)
            try:
                state['scores'][index] = future.result()
            except Exception as exc:
                self.failed.append({'problem_id': record['problem_id'],
                                    'sample_id': record['sample_id'], 'seed': record['seed'],
                                    'scored_path': str(state['path']),
                                    'error': f'{type(exc).__name__}: {exc}'})
                continue
            self.completed += 1
            if all(score is not None for score in state['scores']):
                atomic_json(state['path'], {'generated_sha256': state['digest'], 'scores': state['scores']})
        if self.failed:
            atomic_json(self.failure_path, {'failures': self.failed})
            raise RuntimeError(self.failed[0]['error'])

    def submit(self, path, value, records, item):
        state = {'path': path, 'digest': digest(value), 'scores': [None] * len(records)}
        for index, record in enumerate(records):
            self.drain()
            while len(self.pending) >= self.limit:
                self.drain(block=True)
            future = self.pool.submit(score_sample, self.verifier, item, record)
            self.pending[future] = state, index, record
            self.peak = max(self.peak, len(self.pending))

    def finish(self):
        while self.pending:
            self.drain(block=True)

    def report(self):
        return {'scored_candidates_this_process': self.completed,
                'pending_candidates': len(self.pending), 'peak_pending_candidates': self.peak,
                'max_pending_candidates': self.limit,
                'elapsed_seconds': time.monotonic() - self.started}


def run(c, name, items, output, *, smoke=False, smoke_samples=2, num_shards=1, shard_index=0,
        generator_factory=Generator, verifier=None):
    output = Path(output)
    verifier = verifier or adapter(c, output)
    if smoke_samples not in (2, 64, 128):
        raise ValueError("Smoke tests support 2, 64 or 128 samples per question")
    n = smoke_samples if smoke else c["sampling"]["samples_per_question"]
    items = evaluation_items(c, items, smoke)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid evaluation shard")
    if smoke and num_shards != 1:
        raise ValueError("Diagnostic smoke runs cannot be sharded")
    width = min(n, c["sampling"]["samples_per_block"])
    generator = None
    qbatch = c["sampling"]["questions_per_batch"]
    groups = [list(enumerate(items[offset:offset + qbatch], start=offset))
              for offset in range(0, len(items), qbatch)]
    groups = [group for number, group in enumerate(groups) if number % num_shards == shard_index]
    shard_items = [item for group in groups for _, item in group]
    total_batches = len(groups) * math.ceil(n / width)
    with (ThreadPoolExecutor(max_workers=c["runtime"]["verifier_workers"]) as pool,
          tqdm(total=total_batches, desc=f"{c['domain']} {name}", unit="batch",
               mininterval=1.0, dynamic_ncols=False) as progress):
        overlap = c['runtime'].get('overlap_verification', False)
        queue = VerificationQueue(pool, verifier,
            max(4 * qbatch * width, 4 * c['runtime']['verifier_workers']),
            output / f'verifier_failure.shard-{shard_index:03d}.json')
        for group in groups:
            for start in range(0, n, width):
                count = min(width, n - start)
                pending, requests = [], []
                for index, item in group:
                    generated, scored = files_for(output, index, start)
                    if scored.exists() and not generated.exists():
                        raise ValueError("Scores exist without generation provenance")
                    if generated.exists():
                        value = json.loads(generated.read_text())
                        validate_block(c, item, start, count, value)
                        continue
                    pending.append((index, item))
                    requests.extend({"problem_id": item["problem_id"], "sample_id": sample_id,
                        "seed": sample_seed(c["sampling"]["seed"], item["problem_id"], sample_id)}
                        for sample_id in range(start, start + count))
                if requests:
                    if generator is None:
                        generator = generator_factory(c, name, verifier, shard_items)
                    responses = generator.generate(requests)
                    if len(responses) != len(requests):
                        raise RuntimeError("Generator response count mismatch")
                    for j, (index, item) in enumerate(pending):
                        records = [{**r, **answer} for r, answer in zip(requests[j*count:(j+1)*count], responses[j*count:(j+1)*count])]
                        value = block(c, item, start, count, records)
                        validate_block(c, item, start, count, value)
                        atomic_json(files_for(output, index, start)[0], value)
                # Drain ready work in any order; slow candidates do not impose
                # a barrier on unrelated blocks. Saved generations are reused.
                queue.drain()
                for index, item in group:
                    generated, scored = files_for(output, index, start)
                    value = json.loads(generated.read_text())
                    records = validate_block(c, item, start, count, value)
                    if scored.exists():
                        read_scores(scored, value, count)
                    else:
                        queue.submit(scored, value, records, item)
                if not overlap:
                    queue.finish()
                print(json.dumps({"question_indices": [index for index, _ in group],
                                  "samples_through": start + count, "total_questions": len(items),
                                  "shard_index": shard_index, "num_shards": num_shards,
                                  "smoke": smoke, "verification_queue": queue.report()}), flush=True)
                progress.update()
        queue.finish()
        atomic_json(output / f'verification_queue.shard-{shard_index:03d}.json', queue.report())
    if num_shards > 1:
        return {"status": "shard_complete", "shard_index": shard_index,
                "num_shards": num_shards, "questions": len(shard_items)}
    return summarize(c, name, items, output, smoke=smoke, smoke_samples=smoke_samples)


def summarize(c, name, items, output, *, smoke=False, smoke_samples=2):
    output = Path(output)
    items = evaluation_items(c, items, smoke)
    n = smoke_samples if smoke else c["sampling"]["samples_per_question"]
    width = min(n, c["sampling"]["samples_per_block"])
    ks = [k for k in KS if k <= n]
    tasks, errors, truncated = [], Counter(), 0
    for index, item in enumerate(items):
        outcomes = []
        for start in range(0, n, width):
            count = min(width, n - start)
            generated, scored = files_for(output, index, start)
            value = json.loads(generated.read_text())
            records = validate_block(c, item, start, count, value)
            scores = read_scores(scored, value, count)
            for r, score in zip(records, scores):
                if r["finish_reason"] == "length" and score["correct"]:
                    raise ValueError("Truncated samples must count as failures")
                outcomes.append(score["correct"])
                truncated += r["finish_reason"] == "length"
                if score.get("error"):
                    errors[score["error"]] += 1
        correct = sum(outcomes)
        tasks.append({"problem_id": item["problem_id"], "n": n, "correct": correct,
            "pass_at_k": {str(k): pass_at_k(n, correct, k) for k in ks}, "extra_info": item["extra_info"]})
    if not tasks or (not smoke and len(tasks) != c["data"]["expected_questions"]):
        raise ValueError("Refusing to report an incomplete evaluation split")
    result = {"status": "smoke_complete" if smoke else "complete", "domain": c["domain"],
        "split": c["data"]["split"], "release": c["data"]["release"], "model": name,
        "questions": len(tasks), "samples_per_question": n, "total_samples": n * len(tasks),
        "pass_at_k": {str(k): sum(t["pass_at_k"][str(k)] for t in tasks) / len(tasks) for k in ks},
        "estimator": "mean_q [1 - choose(n-c_q,k)/choose(n,k)]",
        "truncated_samples": truncated, "truncation_rate": truncated / (n * len(tasks)),
        "errors": dict(errors), "sampling": c["sampling"],
        "verifier": "pinned_bird_execution_accuracy" if c["domain"] == "bird" else "pinned_DPH_Spider_execution_accuracy_direct_SQL",
        "output_contract": "strict_raw_SQL_or_Python; truncated_responses_fail; no_deduplication_or_reranking"}
    manifest = output / "bundle/manifest.json"
    if manifest.exists():
        result["bundle_manifest_sha256"] = sha256(manifest)
    extension = output / "extension.json"
    if extension.exists() and not smoke:
        provenance = json.loads(extension.read_text())
        if provenance["config_sha256"] != digest(c):
            raise ValueError("Extension provenance does not match evaluation configuration")
        result["extension"] = {"manifest_sha256": sha256(extension),
            "parent_summary_sha256": provenance["parent_summary_sha256"],
            "reused_samples_per_question": provenance["reused_samples_per_question"]}
    atomic_json(output / "per_question.json", tasks)
    atomic_json(output / "summary.json", result)
    lines = [f"# {c['domain']} / {c['data']['split']} / {name}", "",
        "DIAGNOSTIC SMOKE ONLY" if smoke else "Complete main evaluation", "",
        f"{len(tasks)} questions; {n} samples/question; {truncated} truncated samples (counted as failures).", "",
        "| k | pass@k (%) |", "| --- | ---: |"]
    lines += [f"| {k} | {100 * result['pass_at_k'][str(k)]:.2f} |" for k in ks]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return result
