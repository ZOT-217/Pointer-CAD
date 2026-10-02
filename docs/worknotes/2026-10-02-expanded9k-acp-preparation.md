# Expanded-9K ACP Preparation

The authoritative source population is the existing 9,767-row operation-set
manifest. The CPU job runs current Pipeline V2 into a new output root, freezes
accepted rows, prepares native FACE/EDGE sidecars, and performs a no-CadQuery
release check. Every record uses the exact fixed conditioning text
`Construct the CAD model.`.

The GPU job reads only frozen JSON and native sidecars. It uses Qwen 0.5B,
bf16, rank-8 LoRA, native geometry, four-rank NCCL DDP, bounded preflight,
atomic checkpoints, and JSONL metrics. Historical EDGE and SplitFeature remain
reported coverage debt and do not block this diagnostic.
