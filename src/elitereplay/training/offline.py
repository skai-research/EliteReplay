"""Fresh IID sampling, DPH partitioning, and the existing MAP-Elites pipeline.

IID never reads evolutionary candidates. Each request is journaled separately,
so interrupted generation resumes without spending its successful requests twice.
"""
from __future__ import annotations

import importlib
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from .config import DPH_METHODS, ROOT, path, offline_dir, adapter_args, fingerprint, offline_fingerprint
from .data import sha256
from .objectives.dph_js import is_preserved
from elitereplay.archive.schema import DecodingConfig, GenerationResponse, PipelineConfig
from elitereplay.archive.providers import ChatSolutionGenerator, make_bedrock_labeler, make_litellm_labeler
from elitereplay.archive.archive_diversity import canonical_artifact_key
from elitereplay.archive.trinity_support import render_sft_with_terminal_eos


def adapter(c, *, offline=False):
    module = importlib.import_module(f"elitereplay.archive.{c['domain']}")
    args = offline_adapter_args(c) if offline else adapter_args(c)
    return getattr(module, {"bird": "BirdAdapter", "taco": "TacoAdapter", "optmath": "OptmathAdapter"}[c["domain"]])(**args)


def offline_adapter_args(c):
    args = adapter_args(c)
    if c["domain"] in ("taco", "optmath") and "source_dir" in c["offline"]:
        args["sandbox_config"]["workspace_root"] = str(offline_dir(c) / "sandbox")
    return args


def atomic_json(filename, data):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_suffix(filename.suffix + ".tmp")
    temporary.write_text(json.dumps(data, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(filename)


def write_jsonl(filename, rows):
    filename = Path(filename)
    temporary = filename.with_suffix(".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(filename)


class LocalGenerator:
    """A single batched vLLM engine, shared across seed/mutation requests."""
    def __init__(self, c):
        from vllm import LLM
        self.c = c
        self.model_id = str(path(c["model"]["path"]))
        self.engine = LLM(model=self.model_id, tensor_parallel_size=c["runtime"]["rollout_gpus"],
            dtype="bfloat16", trust_remote_code=True, enforce_eager=True,
            gpu_memory_utilization=c["runtime"]["rollout_memory_fraction"],
            max_model_len=c["model"]["max_prompt_tokens"] + c["model"]["max_response_tokens"],
            max_num_seqs=c["runtime"]["max_num_seqs"], seed=c["seed"], logprobs_mode="raw_logprobs",
            limit_mm_per_prompt={"image": 0, "video": 0})
        self.tokenizer = self.engine.get_tokenizer()

    @property
    def max_prompt_tokens(self):
        return self.c["model"]["max_prompt_tokens"]

    def render_prompt_ids(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, enable_thinking=False, return_dict=False)

    def prompt_token_count(self, messages):
        return len(self.render_prompt_ids(messages))

    def complete(self, **call):
        return self.complete_batch([call])[0]

    def complete_batch(self, calls):
        from vllm import SamplingParams
        prompts, parameters = [], []
        capture = self.c["method"] in DPH_METHODS
        for call in calls:
            ids = self.render_prompt_ids(call["messages"])
            if len(ids) > self.c["model"]["max_prompt_tokens"]:
                raise ValueError(f"Offline prompt has {len(ids)} tokens; configured maximum exceeded")
            prompts.append({"prompt_token_ids": ids})
            d = call["decoding"]
            parameters.append(SamplingParams(n=1, seed=call["seed"], temperature=d.temperature,
                top_p=d.top_p, top_k=self.c["training"]["top_k"], max_tokens=d.max_tokens,
                min_p=self.c["training"]["min_p"], presence_penalty=self.c["training"]["presence_penalty"],
                repetition_penalty=self.c["training"]["repetition_penalty"],
                logprobs=0 if capture else None))
        outputs = self.engine.generate(prompts, parameters, use_tqdm=False)
        results = []
        for request in outputs:
            out = request.outputs[0]
            metadata = {}
            if capture:
                ids = list(out.token_ids)
                if out.logprobs is None or len(out.logprobs) != len(ids):
                    raise RuntimeError("vLLM did not return aligned reference log probabilities")
                metadata = {"tokens": list(request.prompt_token_ids) + ids,
                    "prompt_length": len(request.prompt_token_ids),
                    "reference_logprobs": [float(lp[token].logprob) for token, lp in zip(ids, out.logprobs)]}
            results.append(GenerationResponse(out.text, out.finish_reason or "other",
                len(request.prompt_token_ids), len(out.token_ids), self.model_id, metadata))
        return results


def qd_config(c):
    o, m, t = c["offline"], c["model"], c["training"]
    root, data = offline_dir(c), path(c["data"]["root"])
    result = {"experiment": {"name": f"{c['domain']}_qd_s{c['seed']}", "seed": c["seed"], "output_dir": str(root / "map_elites")},
        "models": {"base_model": str(path(m["path"])), "tokenizer": str(path(m["path"])),
                   "base_model_revision": m["revision"], "tokenizer_revision": m["revision"], "labeler_model": o["labeler_model"]},
        "data": {"domain": c["domain"], "aux_prompt_count": c["data"]["archive_questions"],
            "aux_search_split": str(data / "aux_search.parquet"), "problem_id_field": "extra_info.problem_id",
            "split_manifest": str(data / "split_manifest.json")},
        "taxonomy": {"version": "optmath_lp_milp_formulations_v1" if c["domain"] == "optmath" else f"{c['domain']}_v1", "size": o["taxonomy_size"], "sample_cap": o["taxonomy_sample_cap"],
                     "feasible_limit": o["feasible_labels"], "confidence_min": o["label_confidence"]},
        "search": {"method": "qd", "seed_samples_per_problem": o["seed_samples"], "rounds": o["rounds"],
            "children_per_round_per_problem": o["children_per_round"], "target_elites_per_problem": o["per_question_cap"],
            "temperature": t["temperature"], "top_p": t["top_p"], "max_response_tokens": m["max_response_tokens"],
            "generation_batch_size": o["generation_batch_size"], "label_batch_size": o["generation_batch_size"],
            "verifier_workers": o["verifier_workers"], "cell_descriptor": o.get("cell_descriptor", "primary"),
            "attempt_aware": o.get("attempt_aware", False)},
        "export": {"max_examples": o["export_target"], "publish_path": str(root / "archive.jsonl"), "deduplication": "canonical_ast"},
        "domain_adapter": offline_adapter_args(c)}
    if o.get("labeler_provider") == "litellm":
        result["providers"] = {"labeler": {"factory": "elitereplay.archive.providers:make_litellm_labeler",
            "model": o["labeler_model"], "api_base": o["labeler_api_base"],
            "max_workers": o["label_workers"], "attempts": 3 if c["domain"] == "optmath" else 5, "timeout_seconds": 600, "max_tokens": 4096}}
    return result


def select_iid(records, *, domain, seed, target, per_question_cap):
    """Question-balanced export; random correct, canonical-unique candidates."""
    groups, seen = defaultdict(list), set()
    for record in records:
        if not record["fully_correct"]:
            continue
        key = (record["problem_id"], canonical_artifact_key(domain, record["canonical_response"]))
        if key in seen:
            continue
        seen.add(key)
        groups[record["problem_id"]].append(record)
    rng = random.Random(seed)
    pids = sorted(groups)
    rng.shuffle(pids)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    for rank in range(per_question_cap):
        for pid in pids:
            if rank < len(groups[pid]) and len(selected) < target:
                selected.append(groups[pid][rank])
    return selected


def generate_journal(c, rows, generator, domain_adapter, *, count=None, index_offset=0):
    root = offline_dir(c)
    journal = root / "requests"
    journal.mkdir(exist_ok=True)
    if count is None:
        count = c["method_options"]["presample_rollouts"] if c["method"] in DPH_METHODS else c["offline"]["candidates_per_question"]
    decoding = DecodingConfig(c["training"]["temperature"], c["training"]["top_p"], c["model"]["max_response_tokens"])
    tasks = [(row, i) for row in rows for i in range(index_offset, index_offset + count)]
    records = []
    batch_size = c["offline"]["generation_batch_size"]
    def identity(row, index):
        return fingerprint([c["seed"], c["method"], row["extra_info"]["problem_id"], index])
    def score(task_response):
        (row, index), response = task_response
        parsed = domain_adapter.parse_strict(response.raw_response)
        correct = False
        if parsed.accepted and response.finish_reason != "length":
            canonical = parsed.canonical_response
            # Match QD's artifact round trip and the actual supervised EOS length.
            tok = generator.tokenizer
            roundtrip = tok.decode(tok.encode(canonical, add_special_tokens=False), skip_special_tokens=False)
            verification = domain_adapter.verify(row, canonical) if roundtrip == canonical else None
            correct = bool(verification and verification.fully_correct and verification.executable)
            if correct and c["method"] not in DPH_METHODS:
                msgs = domain_adapter.format_prompt(row) + [{"role": "assistant", "content": canonical}]
                tokens, length, _ = render_sft_with_terminal_eos(tok, msgs)
                correct = len(tokens) - length <= c["model"]["max_response_tokens"]
        record = {"candidate_id": identity(row, index), "problem_id": row["extra_info"]["problem_id"],
            "index": index, "fully_correct": correct, "canonical_response": parsed.canonical_response,
            "generation": response.to_dict()}
        atomic_json(journal / (record["candidate_id"] + ".json"), record)
        return record
    for start in range(0, len(tasks), batch_size):
        pending = []
        for task in tasks[start:start + batch_size]:
            file = journal / (identity(*task) + ".json")
            if file.exists():
                records.append(json.loads(file.read_text()))
            else:
                pending.append(task)
        if pending:
            calls = [{"messages": domain_adapter.format_prompt(row), "decoding": decoding,
                      "seed": int(identity(row, i)[:8], 16)} for row, i in pending]
            responses = generator.complete_batch(calls)
            if len(responses) != len(pending):
                raise RuntimeError("Generator returned the wrong number of responses")
            with ThreadPoolExecutor(max_workers=c["offline"]["verifier_workers"]) as pool:
                records.extend(pool.map(score, zip(pending, responses)))
        print(f"offline: {min(start + batch_size, len(tasks))}/{len(tasks)} proposals verified", flush=True)
    return records


def prepare(c, *, seeds_only=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from elitereplay.archive.pipeline import EliteReplayPipeline
    if seeds_only and c["method"] != "EliteReplay":
        raise ValueError("Seed-only preparation is supported only for EliteReplay")
    if c["method"] == "EliteReplay" and not seeds_only and c["offline"].get("labeler_provider") == "litellm":
        from elitereplay.integrations.labeler import require_api_key
        require_api_key()  # Fail before loading the GPU generator if auth is missing.
    root, data = offline_dir(c), path(c["data"]["root"])
    if c["method"] not in ("EliteReplay", "IID_Replay", *DPH_METHODS):
        return {"status": "No offline preparation required"}
    root.mkdir(parents=True, exist_ok=True)
    contract = {"config_sha256": offline_fingerprint(c), "train_sha256": sha256(data / "train.parquet"),
                "aux_sha256": sha256(data / "aux_search.parquet")}
    if "iid_taxonomy_path" in c["offline"]:
        contract["taxonomy_sha256"] = sha256(path(c["offline"]["iid_taxonomy_path"]))
    contract_file = root / "contract.json"
    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError("Offline inputs/config changed; use a new output_root")
    atomic_json(contract_file, contract)
    if (root / "complete.json").exists():
        return {"status": "already prepared", **json.loads((root / "complete.json").read_text())}
    filename = "train.parquet" if c["method"] in DPH_METHODS else "aux_search.parquet"
    rows = pq.read_table(data / filename).to_pylist()
    domain_adapter = adapter(c, offline=True)
    generator = LocalGenerator(c)
    if c["method"] == "EliteReplay":
        raw = qd_config(c)
        (root / "map_elites.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
        config = PipelineConfig.from_mapping(raw)
        if c["offline"].get("labeler_provider") == "litellm":
            labeler = make_litellm_labeler(config=config, settings=raw["providers"]["labeler"])
        else:  # Explicit compatibility for historical Bedrock configurations.
            labeler = make_bedrock_labeler(config=config, settings={"model": c["offline"]["labeler_model"],
                "region": c["offline"].get("labeler_region", "us-east-1"), "max_workers": c["offline"]["label_workers"],
                "attempts": 3 if c["domain"] == "optmath" else 5, "timeout_seconds": 600, "max_tokens": 4096})
        pipeline = EliteReplayPipeline(config=config, adapter=domain_adapter,
            generator=ChatSolutionGenerator(generator), labeler=labeler, project_root=ROOT,
            trinity_root=ROOT / "Trinity-RFT")
        if c["offline"].get("seed_source_config"):
            from .seed_reuse import reuse_seeds
            reuse_seeds(c, pipeline, rows)
        if seeds_only:
            result = {**contract, **pipeline.collect_seeds(rows),
                      "seed_journal_sha256": sha256(pipeline.seed_observations_path)}
            atomic_json(root / "seeds_complete.json", result)
            return result
        pipeline.induce_taxonomy(rows)
        pipeline.build_archive(rows)
        pipeline.export_sft(rows)
        from elitereplay.archive.archive_diversity import audit_archive, write_audit
        summary, questions, review = audit_archive(root / "map_elites", export_path=root / "archive.jsonl",
            target_min=c["offline"]["minimum_examples"] if c["domain"] == "optmath" else c["offline"]["export_target"], target_max=c["offline"]["export_target"],
            per_problem_cap=c["offline"]["per_question_cap"],
            replay_draws=c["training"]["rollout_iterations"] * c["training"]["replay_per_iteration"])
        write_audit(root / "audit", summary, questions, review)
        if not summary["integrity_pass"] or not summary["target"]["met"]:
            raise RuntimeError(f"Archive audit failed; inspect {root / 'audit/README.md'}")
        exports = [json.loads(line) for line in (root / "archive.jsonl").read_text().splitlines()]
        result = {"examples": len(exports), "archive_sha256": sha256(root / "archive.jsonl"),
                  "integrity_pass": True, "audit_summary_sha256": sha256(root / "audit/summary.json")}
    else:
        records = generate_journal(c, rows, generator, domain_adapter)
        by_id = {r["extra_info"]["problem_id"]: r for r in rows}
        if c["method"] == "IID_Replay":
            chosen = select_iid(records, domain=c["domain"], seed=c["seed"], target=c["offline"]["export_target"],
                                per_question_cap=c["offline"]["per_question_cap"])
            exports = [{"messages": domain_adapter.format_prompt(by_id[r["problem_id"]]) +
                [{"role": "assistant", "content": r["canonical_response"]}],
                "metadata": {"problem_id": r["problem_id"], "candidate_id": r["candidate_id"],
                    "archive_method": "iid", "fully_correct": True, "sample_source": "frozen_sft"}} for r in chosen]
            if c["method_options"].get("replay_sampling") == "mode_cycles":
                from .iid_replay import export_verified_iid
                exports, audit = export_verified_iid(c, records, by_id, domain_adapter)
                atomic_json(root / "iid_audit.json", audit)
            write_jsonl(root / "archive.jsonl", exports)
            result = {"examples": len(exports), "archive_sha256": sha256(root / "archive.jsonl"),
                      "integrity_pass": True}
        else:
            groups = defaultdict(list)
            for record in records:
                groups[record["problem_id"]].append(record)
            preserved = {pid for pid, group in groups.items() if is_preserved(
                [r["fully_correct"] for r in group], c["method_options"]["preservation_threshold"])}
            reference_records = records
            extra_proposals = 0
            if c["method"] == "dph_f" and preserved:
                # Authors' protocol: an independent reference response after
                # the 8/8 screen. A failed ninth sample returns to exploration.
                reference_records = generate_journal(c, [by_id[pid] for pid in sorted(preserved)],
                    generator, domain_adapter, count=1,
                    index_offset=c["method_options"]["presample_rollouts"])
                extra_proposals = len(reference_records)
                preserved = {r["problem_id"] for r in reference_records if r["fully_correct"]}
                records = records + reference_records
            if not preserved or len(preserved) == len(rows):
                raise RuntimeError("DPH needs non-empty preservation and exploration sets; inspect presampling results")
            references = [{**r["generation"]["metadata"], "metadata": {
                "problem_id": r["problem_id"], "candidate_id": r["candidate_id"],
                "archive_method": c["method"], "fully_correct": True}}
                for r in reference_records if r["problem_id"] in preserved]
            write_jsonl(root / "references.jsonl", references)
            exploration = [row for pid, row in by_id.items() if pid not in preserved]
            pq.write_table(pa.Table.from_pylist(exploration), root / "exploration.parquet")
            result = {"preserved_questions": len(preserved), "exploration_questions": len(exploration),
                "independent_reference_proposals": extra_proposals,
                "reference_responses": len(references), "references_sha256": sha256(root / "references.jsonl"),
                "exploration_sha256": sha256(root / "exploration.parquet")}
        result.update(proposals=len(records), generated_tokens=sum(r["generation"]["output_tokens"] for r in records))
    if c["domain"] == "optmath" and c["method"] == "EliteReplay":
        if not c["offline"]["minimum_examples"] <= result["examples"] <= c["offline"]["export_target"]:
            raise RuntimeError("OptMATH requires a nonempty verified archive within its configured ceiling")
        effective = {"schema_version": "optmath.effective_archive.v1",
            "generation_config_sha256": offline_fingerprint(c),
            "requested_max_examples": c["offline"]["export_target"],
            "effective_examples": result["examples"], "archive_sha256": result["archive_sha256"],
            "shortfall": c["offline"]["export_target"] - result["examples"],
            "duplicate_fill": False, "verification_weakened": False}
        effective_path = root / "effective_archive.json"
        if effective_path.exists() and json.loads(effective_path.read_text()) != effective:
            raise ValueError("Frozen effective archive identity changed")
        atomic_json(effective_path, effective)
        result.update(effective_archive_sha256=sha256(effective_path), effective_examples=result["examples"])
    elif c["method"] not in DPH_METHODS and result["examples"] != c["offline"]["export_target"]:
        raise RuntimeError(f"Export yielded {result['examples']} examples, target {c['offline']['export_target']}. "
                           "Do not duplicate-fill. Review both QD/IID budgets or matched targets in new run configs.")
    result.update(contract)
    atomic_json(root / "complete.json", result)
    return result
