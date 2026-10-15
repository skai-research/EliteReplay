"""Two-rank maximum-length update check; never writes a model checkpoint.

Run with torchrun inside the pinned container before a recovered run's smoke.
Restores real optimizer/model state, performs two disposable updates, and writes
only diagnostic JSON. The production training state remains untouched.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[int(os.environ["LOCAL_RANK"])]

import json
from pathlib import Path
import sys
import time

import torch
import torch.distributed as dist
import yaml
from tensordict import TensorDict
from evorl.baselines.config import run_dir
from evorl.baselines.kernels import require_fast_path
from evorl.baselines import integration
from trinity.common.config import load_config
from trinity.trainer.verl.config import build_verl_config
from trinity.trainer.verl.monkey_patch import patch_verl_engine
from trinity.trainer.verl.losses import build_trinity_loss
from verl.utils.config import omega_conf_to_dataclass
from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def main():
    bundle = Path(sys.argv[1])
    c = yaml.safe_load((bundle / "experiment.yaml").read_text())
    cfg = load_config(str(bundle / "trinity.yaml")); cfg.check_and_update()
    root = run_dir(c)
    job = root / "train/checkpoints/evorl-baselines" / c["domain"] / f"{c['method']}-s{c['seed']}"
    step = int((job / "latest_checkpointed_iteration.txt").read_text())
    checkpoint = job / f"global_step_{step}"
    assert (checkpoint / ".full_checkpoint").is_file()
    torch.cuda.set_device(0); torch.set_num_threads(4)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    require_fast_path(); patch_verl_engine()
    vc = build_verl_config(cfg)
    ac = omega_conf_to_dataclass(vc.actor)
    engine = FSDPEngineWithLMHead(omega_conf_to_dataclass(vc.model), ac.engine, ac.optim, ac.checkpoint)
    engine.initialize(); engine.load_checkpoint(str(checkpoint / "actor"), del_local_after_load=False)
    def optimizer_steps():
        return sorted({int(s["step"].item()) for s in engine.optimizer.state.values() if "step" in s})
    report = {"checkpoint_step": step, "optimizer_steps_before": optimizer_steps(), "cases": []}
    length = c["model"]["max_prompt_tokens"] + c["model"]["max_response_tokens"]
    prompt = c["model"]["max_prompt_tokens"]; response = length - prompt
    ids = [42] * length
    def nested(values, dtype):
        return torch.nested.as_nested_tensor([torch.tensor(v, dtype=dtype) for v in values], layout=torch.jagged)
    data = TensorDict({
        "input_ids": nested([ids] * 2, torch.long),
        "position_ids": nested([list(range(length))] * 2, torch.long),
        "prompts": nested([ids[:prompt]] * 2, torch.long),
        "responses": nested([ids[prompt:]] * 2, torch.long),
        "response_mask": nested([[True] * response] * 2, torch.bool),
        "loss_mask": nested([[True] * response] * 2, torch.bool),
        "old_log_probs": nested([[-12.] * response] * 2, torch.float32),
        "advantages": nested([[1.] * response, [-1.] * response], torch.float32),
        "expert_mask": torch.tensor([False, c["method"] == "dph_f"]),
        "online_token_count": torch.full((2,), float(response * 4)),
        "auxiliary_token_count": torch.full((2,), float(response * 2)),
        "auxiliary_response_count": torch.full((2,), 2.)}, batch_size=[2])
    tu.assign_non_tensor(data, use_remove_padding=False, use_dynamic_bsz=False,
        max_token_len_per_gpu=c["runtime"]["max_tokens_per_gpu"], micro_batch_size_per_gpu=1,
        use_fused_kernels=False, calculate_entropy=False, temperature=1., global_batch_size=4, pad_token_id=0)
    loss = build_trinity_loss(cfg.algorithm)
    for repetition in range(2):
        dist.barrier(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        with engine.train_mode():
            output = engine.train_batch(data.clone(), loss_function=loss)
        torch.cuda.synchronize(); dist.barrier()
        grad = float(output["metrics"]["grad_norm"])
        assert torch.isfinite(torch.tensor(grad)), "Non-finite diagnostic gradient"
        case = {"tokens": length, "response_tokens": response, "seconds": time.monotonic() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30, "grad_norm": grad}
        report["cases"].append(case)
        print(json.dumps({"recovery_stress": case}), flush=True)
    report.update(status="passed", optimizer_steps_after=optimizer_steps(), production_checkpoints_modified=False)
    output_dir = root / "recovery-validation"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"stress-rank{dist.get_rank()}.json").write_text(json.dumps(report, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
