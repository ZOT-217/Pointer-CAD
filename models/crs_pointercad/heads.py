"""Typed pointer scoring, causal feedback, and v0.4 loss."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .candidates import CandidateBank
from .contracts import PointerType


class TypedPointerHeads(nn.Module):
    def __init__(self, hidden_dim: int, pointer_dim: int = 128, typed_scales: bool = True, max_tau: float = 100.0):
        super().__init__()
        self.pointer_dim = pointer_dim
        self.queries = nn.ModuleDict({ptype.value: nn.Linear(hidden_dim, pointer_dim, bias=False) for ptype in PointerType})
        self.log_tau = nn.ParameterDict(
            {ptype.value: nn.Parameter(torch.log(torch.tensor(1 / 0.07))) for ptype in PointerType}
            if typed_scales else {"shared": nn.Parameter(torch.log(torch.tensor(1 / 0.07)))}
        )
        self.typed_scales = typed_scales
        self.max_log_tau = float(torch.log(torch.tensor(max_tau)))

    def clamp_parameters(self) -> None:
        with torch.no_grad():
            for value in self.log_tau.values():
                value.clamp_(max=self.max_log_tau)

    def query(self, hidden: torch.Tensor, pointer_type: PointerType) -> torch.Tensor:
        return self.queries[pointer_type.value](hidden)

    def score(self, hidden: torch.Tensor, bank: CandidateBank, pointer_type: PointerType | None = None) -> torch.Tensor:
        pointer_type = pointer_type or bank.pointer_type
        query = F.normalize(self.query(hidden, pointer_type), dim=-1, eps=1e-6)
        embeddings = F.normalize(bank.embeddings(device=query.device, dtype=query.dtype), dim=-1, eps=1e-6)
        if embeddings.numel() == 0:
            return query.new_empty((0,))
        tau_key = pointer_type.value if self.typed_scales else "shared"
        tau = self.log_tau[tau_key].exp().clamp(max=float(torch.exp(torch.tensor(self.max_log_tau))))
        return torch.matmul(embeddings, query) * tau


class PointerFeedback(nn.Module):
    """Inject a selected embedding only after its producing decoder position."""

    def __init__(self, hidden_dim: int, pointer_dim: int = 128):
        super().__init__()
        self.type_embeddings = nn.Embedding(len(PointerType), hidden_dim)
        self.projections = nn.ModuleDict({ptype.value: nn.Linear(pointer_dim, hidden_dim, bias=False) for ptype in PointerType})

    def forward(self, pointer_type: PointerType, selected_embedding: torch.Tensor) -> torch.Tensor:
        index = torch.tensor([list(PointerType).index(pointer_type)], device=selected_embedding.device)
        typed = self.type_embeddings(index).squeeze(0)
        return typed + self.projections[pointer_type.value](selected_embedding)

    def apply_to_future(self, decoder_inputs: torch.Tensor, position: int, pointer_type: PointerType, selected_embedding: torch.Tensor) -> torch.Tensor:
        result = decoder_inputs.clone()
        result[position + 1:] = result[position + 1:] + self.forward(pointer_type, selected_embedding)
        return result


RECORD_DIMS = {
    "POINT3": 3,
    "VECTOR3": 3,
    "DIRECTION3": 3,
    "AXIS3": 7,
    "PLANE3": 9,
    "FRAME3": 12,
}


class NumericHeads(nn.Module):
    """Trainable MODEL_OUTPUT_SCALAR and MODEL_STRUCTURED_NUMERIC heads."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scalar = nn.Linear(hidden_dim, 1)
        self.records = nn.ModuleDict({name: nn.Linear(hidden_dim, dim) for name, dim in RECORD_DIMS.items()})

    def forward(self, hidden_states: torch.Tensor):
        return self.scalar(hidden_states).squeeze(-1), {
            name: head(hidden_states) for name, head in self.records.items()
        }


def pointer_bce_per_slot(logits: Iterable[torch.Tensor], positives: Iterable[torch.Tensor]) -> tuple[torch.Tensor, int]:
    """Mean BCE within each slot, then mean across slots; multi-positive safe."""
    def flatten(values):
        for value in values:
            if isinstance(value, (list, tuple)):
                yield from flatten(value)
            else:
                yield value

    logits = tuple(flatten(logits))
    positives = tuple(flatten(positives))
    losses = []
    for slot_logits, positive in zip(logits, positives):
        if slot_logits.numel() == 0:
            continue
        positive = positive.to(device=slot_logits.device, dtype=torch.bool)
        if positive.numel() != slot_logits.numel():
            raise ValueError("positive mask and pointer logits have different sizes")
        targets = positive.to(dtype=slot_logits.dtype)
        losses.append(F.binary_cross_entropy_with_logits(slot_logits, targets, reduction="mean"))
    if not losses:
        anchor = logits[0] if logits else torch.zeros((), requires_grad=True)
        return anchor.sum() * 0.0, 0
    return torch.stack(losses).mean(), len(losses)


class ExpandedLoss(nn.Module):
    def __init__(self, lambda_g: float = 1.0, lambda_p: float = 1.0, lambda_s: float = 1.0, lambda_r: float = 1.0):
        super().__init__()
        self.weights = (lambda_g, lambda_p, lambda_s, lambda_r)

    def forward(self, grammar_logits=None, grammar_targets=None, pointer_logits=(), pointer_positive=(), scalar_loss=None, record_loss=None,
                scalar_predictions=None, scalar_targets=None, record_predictions=None, record_targets=None):
        if grammar_logits is None or grammar_targets is None or grammar_logits.numel() == 0:
            grammar = torch.zeros((), device=self._device(pointer_logits, scalar_loss, record_loss))
        else:
            if grammar_logits.ndim > 2:
                grammar_logits = grammar_logits.reshape(-1, grammar_logits.shape[-1])
                grammar_targets = grammar_targets.reshape(-1)
            grammar = F.cross_entropy(grammar_logits, grammar_targets)
        pointer, slots = pointer_bce_per_slot(pointer_logits, pointer_positive)
        if scalar_loss is None:
            scalar = self._numeric_mse(scalar_predictions, scalar_targets, grammar)
        else:
            scalar = scalar_loss
        if record_loss is None:
            record = self._record_mse(record_predictions, record_targets, grammar)
        else:
            record = record_loss
        total = self.weights[0] * grammar + self.weights[1] * pointer + self.weights[2] * scalar + self.weights[3] * record
        return total, {"grammar": grammar, "pointer": pointer, "scalar": scalar, "record": record, "valid_pointer_slots": slots}

    @staticmethod
    def _numeric_mse(predictions, targets, anchor):
        if predictions is None or targets is None:
            return anchor.sum() * 0.0
        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets, device=predictions.device, dtype=predictions.dtype)
        mask = torch.isfinite(targets)
        if not mask.any():
            return predictions.sum() * 0.0
        return F.mse_loss(predictions[mask], targets[mask])

    @staticmethod
    def _record_mse(predictions, targets, anchor):
        if not predictions or not targets:
            return anchor.sum() * 0.0
        losses = []
        for name, prediction in predictions.items():
            target = targets.get(name) if isinstance(targets, Mapping) else None
            if target is None:
                continue
            target = torch.as_tensor(target, device=prediction.device, dtype=prediction.dtype)
            mask = torch.isfinite(target)
            if mask.any():
                losses.append(F.mse_loss(prediction[mask], target[mask]))
        return torch.stack(losses).mean() if losses else anchor.sum() * 0.0

    @staticmethod
    def _zero_like(value):
        return value.sum() * 0.0

    @staticmethod
    def _device(pointer_logits, scalar_loss, record_loss):
        for value in (scalar_loss, record_loss):
            if isinstance(value, torch.Tensor):
                return value.device
        for value in pointer_logits:
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")
