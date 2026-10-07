"""Long trajectories backpropagate action by action and save adapter states."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from scripts import train_stage2_expanded9k as trainer


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Linear(1, 1, bias=False)
        self.base_model.requires_grad_(False)
        self.head = nn.Linear(1, 1, bias=False)


def test_action_streaming_gradient_and_trainable_checkpoint(monkeypatch, tmp_path):
    model = _TinyModel()
    module = trainer.ActionLossModule(model)
    zero_stats = {"slots": 0, "any_positive_at_1": 0, "top3": 0, "top5": 0,
                  "best_positive_rank_sum": 0, "by_type": {}, "by_bank_bucket": {}}

    def fake_loss(model, sequence, supervision, action_index, state, brep, visual_features=None):
        value = model.head.weight.sum() * (action_index + 1)
        return value, {"grammar": value.detach(), "pointer": value.detach() * 0,
                       "scalar": value.detach() * 0, "record": value.detach() * 0,
                       "pointer_stats": zero_stats}

    monkeypatch.setattr(trainer, "training_action_loss", fake_loss)
    sequence = SimpleNamespace(pixel_values=None, image_grid_thw=None,
                               action_boundaries=[SimpleNamespace(action_index=0), SimpleNamespace(action_index=1)])
    class _Collator:
        stage2_conditioning = "text_only"
        def __call__(self, row):
            return sequence
    corpus = SimpleNamespace(state_for=lambda identity, action: (None, None))
    row = {"identity": ("dataset", "sample", "000", "v000"), "supervision": {}}
    loss, _, actions, _ = trainer._loss_one(module, row, _Collator(), corpus, torch.device("cpu"),
                                            backward_scale=0.5)
    assert actions == 2 and torch.isfinite(loss)
    assert torch.allclose(model.head.weight.grad, torch.full_like(model.head.weight, 1.5))
    optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    path = tmp_path / "checkpoint.pt"
    trainer._checkpoint(path, module, optimizer, scheduler, 0, 1, {}, "frozen-sha")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "model.head.weight" in payload["model"]
    assert "model.base_model.weight" not in payload["model"]
    trainer._load_checkpoint(path, module, optimizer, scheduler, "frozen-sha")
