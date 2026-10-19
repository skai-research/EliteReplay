#!/usr/bin/env python3
"""Prepare the BIRD training, archive-search, and validation splits."""
import argparse
import json
import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from elitereplay.training.data import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(yaml.safe_load(args.config.read_text())), indent=2))
