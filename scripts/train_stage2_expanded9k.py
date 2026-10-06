#!/usr/bin/env python3
"""Distributed diagnostic trainer for the frozen expanded-9K Stage2 corpus.

The input is JSON/native sidecars only.  This entrypoint never imports CadQuery
or reconstructs a CRS trajectory during training.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, random, socket, subprocess, time, sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models.crs_pointercad import (CRSExpandedPointerCAD, PreparedStage2Corpus,
    Stage2QwenCollator, frozen_grammar_vocabulary, load_prepared_state,
    training_action_loss)

FIXED_TEXT = "Construct the CAD model."
BUCKETS = ("<=16", "17-32", "33-64", "65-128", ">128")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _init_dist() -> tuple[int, int, int, torch.device]:
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed expanded-9K training requires CUDA/NCCL")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
        device = torch.device("cuda", local)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, local, device


def _cleanup() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _records(root: Path, prepared: Path, *, native_backend: str = "v1") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    native = PreparedStage2Corpus(root, prepared, native_backend=native_backend)
    split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    rows, by_id = [], {}
    for entry in manifest["entries"]:
        identity = tuple(entry[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id"))
        if identity not in native.entries:
            continue
        supervision = json.loads((root / entry["supervision_artifact"]).read_text(encoding="utf-8"))
        rows.append({"identity": identity, "command": supervision["command"], "supervision": supervision,
                     "entry": entry})
        by_id[identity] = rows[-1]
    train_ids = {tuple(x[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id")) for x in split["train"]}
    valid_ids = {tuple(x[k] for k in ("dataset", "sample_id", "source_variant_id", "approved_variant_id")) for x in split["validation"]}
    if train_ids & valid_ids:
        raise ValueError("train/validation source split overlaps")
    for row in rows:
        row["split"] = "validation" if row["identity"] in valid_ids else "train" if row["identity"] in train_ids else "unassigned"
        row["supervision"] = {**row["supervision"], "conditioning": {"text": FIXED_TEXT, "source": "expanded9k_fixed_constant"}}
    return rows, {"manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
                  "train": len([r for r in rows if r["split"] == "train"]),
                  "validation": len([r for r in rows if r["split"] == "validation"])}


class ActionLossModule(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, sequence, supervision, action_index, state, brep):
        return training_action_loss(self.model, sequence, supervision, action_index, state, brep)


def _checkpoint(path: Path, model, optimizer, scheduler, epoch: int, step: int, config: dict[str, Any], split_sha: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "epoch": epoch, "global_step": step, "config": config, "split_sha256": split_sha,
                "rng_state": torch.get_rng_state()}, tmp)
    os.replace(tmp, path)


def _load_checkpoint(path: Path, model, optimizer, scheduler, split_sha: str) -> tuple[int, int]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("split_sha256") != split_sha:
        raise ValueError("checkpoint split membership does not match corpus")
    model.load_state_dict(value["model"]); optimizer.load_state_dict(value["optimizer"]); scheduler.load_state_dict(value["scheduler"])
    if "rng_state" in value: torch.set_rng_state(value["rng_state"])
    return int(value["epoch"]), int(value["global_step"])


def _loss_one(ddp, row, collator, corpus, device):
    sequence = collator(row["supervision"])
    total = None; parts = {name: 0.0 for name in ("grammar", "pointer", "scalar", "record")}; actions = 0
    pointer_stats = {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0, "best_positive_rank_sum": 0, "by_type": {}, "by_bank_bucket": {}}
    for action in sequence.action_boundaries:
        state, brep = corpus.state_for(row["identity"], action.action_index)
        value, metrics = ddp(sequence, row["supervision"], action.action_index, state, brep)
        total = value if total is None else total + value
        for name in parts: parts[name] += float(metrics[name].detach().item())
        stats = metrics.get("pointer_stats", pointer_stats)
        for name in ("slots", "any_positive_at_1", "top3", "top5", "best_positive_rank_sum"): pointer_stats[name] += stats[name]
        for group in ("by_type", "by_bank_bucket"):
            for key, item in stats[group].items():
                target = pointer_stats[group].setdefault(key, {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0, "best_positive_rank_sum": 0})
                for name in target: target[name] += item[name]
        actions += 1
    if total is None: total = torch.zeros((), device=device, requires_grad=True)
    return total / max(actions, 1), parts, actions, pointer_stats


def _pointer_report(stats):
    slots = max(stats["slots"], 1)
    def row(item):
        n = max(item["slots"], 1)
        return {"slots": item["slots"], "any-positive@1": item["any_positive_at_1"] / n,
                "top3": item["top3"] / n, "top5": item["top5"] / n,
                "best-positive-rank": item["best_positive_rank_sum"] / n}
    return {"any-positive@1": stats["any_positive_at_1"] / slots, "top3": stats["top3"] / slots,
            "top5": stats["top5"] / slots, "best-positive-rank": stats["best_positive_rank_sum"] / slots,
            "by_type": {key: row(value) for key, value in stats["by_type"].items()},
            "by_bank_bucket": {key: row(value) for key, value in stats["by_bank_bucket"].items()}}


@torch.no_grad()
def _evaluate(ddp, rows, collator, corpus, device, limit=None):
    values, parts, count = [], {k: 0.0 for k in ("grammar", "pointer", "scalar", "record")}, 0
    for row in rows[:limit] if limit else rows:
        value, item, actions, _ = _loss_one(ddp, row, collator, corpus, device)
        values.append(float(value.detach().item())); count += actions
        for key in parts: parts[key] += item[key]
    return {"total_loss": sum(values) / max(len(values), 1), "L_g": parts["grammar"] / max(len(values), 1),
            "L_p": parts["pointer"] / max(len(values), 1), "L_s": parts["scalar"] / max(len(values), 1),
            "L_r": parts["record"] / max(len(values), 1), "actions": count}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--native-backend", choices=("v1", "v2"), default="v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--preflight-steps", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    rank, world, local, device = _init_dist()
    try:
        random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
        rows, split_info = _records(args.corpus.resolve(), args.prepared.resolve(), native_backend=args.native_backend)
        if not split_info["train"] or not split_info["validation"]:
            raise ValueError("frozen corpus needs nonempty train and validation splits")
        raw = [r["supervision"] for r in rows]
        grammar = frozen_grammar_vocabulary(raw)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        collator = Stage2QwenCollator(tokenizer, grammar_vocabulary=grammar)
        model = CRSExpandedPointerCAD(qwen_model=args.model, dtype="bf16", use_lora=True, lora_rank=8,
            hidden_dim=896, pointer_dim=128, grammar_vocab_size=max(1, len(grammar)), registry_context_mode="TYPE_POOLED",
            use_native_brep=True).to(device)
        module = ActionLossModule(model)
        ddp = DDP(module, device_ids=[local] if device.type == "cuda" and world > 1 else None,
                  find_unused_parameters=False) if world > 1 else module
        optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs * len(rows)))
        args.output.mkdir(parents=True, exist_ok=True)
        if rank == 0:
            for name in ("checkpoints", "logs"): (args.output / name).mkdir(exist_ok=True)
            config = vars(args).copy(); config.update({"world_size": world, "rank": rank, "dtype": "bf16", "lora_rank": 8,
                "fixed_conditioning": FIXED_TEXT, "split": split_info})
            _json(args.output / "training_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in config.items()})
            _json(args.output / "corpus_manifest_reference.json", {"corpus": str(args.corpus.resolve()), "prepared": str(args.prepared.resolve()), **split_info})
            _json(args.output / "environment.json", {"hostname": socket.gethostname(), "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "world_size": world})
            (args.output / "command.txt").write_text(" ".join(os.sys.argv) + "\n", encoding="utf-8")
        split_sha = split_info["manifest_sha256"]
        start_epoch, global_step = 0, 0
        if args.resume:
            start_epoch, global_step = _load_checkpoint(args.resume, module, optimizer, scheduler, split_sha)
        corpus = PreparedStage2Corpus(args.corpus, args.prepared, native_backend=args.native_backend)
        train = [r for r in rows if r["split"] == "train"]
        valid = [r for r in rows if r["split"] == "validation"]
        if not args.skip_preflight:
            probe = train[(rank + global_step) % len(train)]
            for _ in range(args.preflight_steps):
                optimizer.zero_grad(set_to_none=True); loss, _, _, _ = _loss_one(ddp, probe, collator, corpus, device)
                if not torch.isfinite(loss): raise FloatingPointError("preflight loss is not finite")
                loss.backward(); optimizer.step(); scheduler.step()
            if rank == 0:
                preflight_path = args.output / "checkpoints" / "preflight.pt"
                _checkpoint(preflight_path, module, optimizer, scheduler, 0, global_step, vars(args), split_sha)
            if world > 1: dist.barrier()
            preflight_path = args.output / "checkpoints" / "preflight.pt"
            _load_checkpoint(preflight_path, module, optimizer, scheduler, split_sha)
            if rank == 0: _json(args.output / "preflight_results.json", {"status": "PASS", "world_size": world, "steps": args.preflight_steps})
        started = time.perf_counter(); optimizer.zero_grad(set_to_none=True)
        for epoch in range(start_epoch, args.epochs):
            random.Random(args.seed + epoch).shuffle(train)
            for offset in range(0, len(train), args.per_device_batch_size):
                batch = train[offset:offset + args.per_device_batch_size]
                total = None; aggregate = {k: 0.0 for k in ("grammar", "pointer", "scalar", "record")}; action_count = 0
                batch_pointer = {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0, "best_positive_rank_sum": 0, "by_type": {}, "by_bank_bucket": {}}
                for row in batch:
                    value, parts, count, stats = _loss_one(ddp, row, collator, corpus, device); total = value if total is None else total + value
                    action_count += count
                    for key in aggregate: aggregate[key] += parts[key]
                    for key in ("slots", "any_positive_at_1", "top3", "top5", "best_positive_rank_sum"): batch_pointer[key] += stats[key]
                    for group in ("by_type", "by_bank_bucket"):
                        for name, item in stats[group].items():
                            target = batch_pointer[group].setdefault(name, {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0, "best_positive_rank_sum": 0})
                            for key in target: target[key] += item[key]
                total = total / len(batch) / args.gradient_accumulation_steps
                if not torch.isfinite(total): raise FloatingPointError("training loss is not finite")
                total.backward()
                optimizer_step = ((offset // args.per_device_batch_size + 1) % args.gradient_accumulation_steps == 0)
                if optimizer_step or offset + len(batch) >= len(train):
                    torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0); optimizer.step(); optimizer.zero_grad(set_to_none=True); scheduler.step(); global_step += 1
                    if rank == 0:
                        elapsed = max(time.perf_counter() - started, 1e-9)
                        _append(args.output / "metrics.jsonl", {"split": "train", "epoch": epoch, "global_step": global_step,
                            "learning_rate": optimizer.param_groups[0]["lr"], "total_loss": float(total.detach().item() * args.gradient_accumulation_steps),
                            "L_g": aggregate["grammar"] / len(batch), "L_p": aggregate["pointer"] / len(batch), "L_s": aggregate["scalar"] / len(batch), "L_r": aggregate["record"] / len(batch),
                            "pointer": _pointer_report(batch_pointer),
                            "examples_per_second": (global_step * args.per_device_batch_size * world) / elapsed,
                            "optimizer_steps_per_second": global_step / elapsed, "peak_allocated_vram": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                            "peak_reserved_vram": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0, "actions": action_count})
                    if rank == 0 and global_step % args.checkpoint_every == 0:
                        _checkpoint(args.output / "checkpoints" / f"step-{global_step:08d}.pt", module, optimizer, scheduler, epoch, global_step, vars(args), split_sha)
                    if rank == 0 and global_step % args.validate_every == 0:
                        validation = _evaluate(ddp, valid, collator, corpus, device)
                        _append(args.output / "metrics.jsonl", {"split": "validation", "epoch": epoch, "global_step": global_step,
                            **validation, "pointer": {"any-positive@1": None, "top3": None, "top5": None,
                            "best_positive_rank": None, "by_type": {}, "by_bank_bucket": {}}})
            if rank == 0:
                _checkpoint(args.output / "checkpoints" / f"epoch-{epoch + 1:03d}.pt", module, optimizer, scheduler, epoch + 1, global_step, vars(args), split_sha)
            if world > 1: dist.barrier()
        if rank == 0:
            _json(args.output / "result.json", {"status": "complete", "diagnostic_only": True, "epochs": args.epochs, "global_step": global_step, "world_size": world, "wall_seconds": time.perf_counter() - started})
        return 0
    except Exception as exc:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True); _append(args.output / "metrics.jsonl", {"status": "failure", "failure_kind": type(exc).__name__, "message": str(exc), "global_step": locals().get("global_step", 0)})
        raise
    finally:
        _cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
