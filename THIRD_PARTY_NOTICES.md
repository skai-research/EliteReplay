# Third-party notices

Original EliteReplay code is released under the MIT License in `LICENSE`.
Included Trinity-RFT, veRL, and DPH SQL evaluation code remains under Apache-2.0.
The MIT license does not replace
licenses or terms attached to datasets, model weights, or third-party code.

| Component | Source / provenance | Included notice |
| --- | --- | --- |
| Trinity-RFT modified files | agentscope-ai/Trinity-RFT; commit and hashes in `runtime/locks/runtime.lock.json` | `vendor/runtime/trinity/LICENSE` |
| veRL modified files | verl-project/verl; commit and hashes in the same lock | `vendor/runtime/verl/LICENSE` |
| DPH SQL comparison code | seamoke/DPH-RL at `fea6990642b072a59a1ecc653da31f9b876bd775` | `src/elitereplay/evaluation/vendor/DPH_LICENSE` |

The existing released elite archive contains model-generated SQL paired with
BIRD-derived task prompts. Its provenance is in `elite_archive/source_complete.json`.
BIRD and Spider data and the Qwen base model are obtained separately
under their upstream licenses and access terms. This repository grants no new
rights over those third-party inputs. No base-model or trained-model weights
are included in this code release.
