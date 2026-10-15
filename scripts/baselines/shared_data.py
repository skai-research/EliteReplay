#!/usr/bin/env python3
"""Restore/verify the Git-distributed training release; no downloads or resampling."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evorl.baselines.config import ROOT
from evorl.baselines.shared_data import materialize_domain, verify_domain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["materialize", "verify"], default="verify", nargs="?")
    parser.add_argument("--root", type=Path, default=ROOT / "data/baselines/v1")
    args = parser.parse_args()
    for domain in ("bird", "taco"):
        result = (verify_domain(args.root / domain) if args.action == "verify"
                  else {"domain": domain, "restored": materialize_domain(args.root / domain)})
        print(json.dumps(result, indent=2), flush=True)
