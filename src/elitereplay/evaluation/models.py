"""Validate complete, explicitly selected Hugging Face model exports."""
import json
from .config import path

def model_files(model):
    root = path(model["path"])
    files = [root / "config.json", root / "tokenizer_config.json"]
    if not (root / "tokenizer.json").exists():
        raise FileNotFoundError(f"Missing full tokenizer at {root}/tokenizer.json")
    files.append(root / "tokenizer.json")
    indices = [root / n for n in ("model.safetensors.index.json", "pytorch_model.bin.index.json") if (root / n).exists()]
    if indices:
        index = indices[0]
        files.append(index)
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards:
            raise ValueError("Empty model weight index")
        for shard in sorted(shards):
            # HF snapshots legitimately symlink shards into ../../blobs. Check
            # the index's filename, not the symlink destination.
            if not isinstance(shard, str) or shard in ("", ".", "..") or "/" in shard or "\\" in shard:
                raise ValueError("Unsafe checkpoint shard path")
            files.append(root / shard)
    else:
        weights = [root / n for n in ("model.safetensors", "pytorch_model.bin") if (root / n).is_file()]
        if not weights:
            raise FileNotFoundError(f"No Hugging Face weights at {root}; config-only FSDP directories are not evaluable. Export/merge weights explicitly.")
        files.append(weights[0])
    files += sorted(p for p in root.glob("*.json") if p not in files)
    # Newer Transformers saves chat templates separately; tokenization inputs
    # must be bound to a run just as model weights are.
    for pattern in ("*.jinja", "*.txt", "*.model"):
        files += sorted(p for p in root.glob(pattern) if p not in files)
    if model["checkpoint_marker"]:
        files.append(path(model["checkpoint_marker"]))
    for f in files:
        if not f.is_file() or (f.suffix != ".full_checkpoint" and f.name != ".full_checkpoint" and f.stat().st_size == 0):
            raise FileNotFoundError(f"Missing/empty checkpoint component: {f}")
    return files
