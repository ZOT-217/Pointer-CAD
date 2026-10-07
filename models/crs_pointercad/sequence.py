"""Authoritative frozen-command to Qwen sequence materialization.

The CRS command positions are semantic positions.  A tokenizer may split one
command atom into several Qwen tokens, so this module keeps the complete span
map and routes every supervision target through it once.  Consumers must use
the returned maps; searching decoded text is intentionally unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


# Exact system instruction from the original Pointer-CAD train.py input path.
POINTERCAD_SYSTEM_INSTRUCTION = (
    "You are an expert mechanical engineer. Based on the user's text requirements, "
    "generate the corresponding CAD model design."
)
LEGACY_ADAPTER_INSTRUCTION = "Construct the CAD model."


@dataclass(frozen=True)
class TokenSpan:
    command_position: int
    start: int
    end: int
    token: str
    action_index: int | None


@dataclass(frozen=True)
class TargetPosition:
    kind: str
    command_position: int
    model_position: int
    action_index: int
    slot: str
    subposition: Any = None
    target_index: int | None = None
    target_candidate: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ActionSpan:
    action_index: int
    start: int
    end: int


@dataclass(frozen=True)
class Stage2Sequence:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    conditioning_text: str
    conditioning_source: str
    conditioning_end: int
    token_spans: tuple[TokenSpan, ...]
    action_boundaries: tuple[ActionSpan, ...]
    grammar_positions: tuple[TargetPosition, ...]
    pointer_positions: tuple[TargetPosition, ...]
    scalar_positions: tuple[TargetPosition, ...]
    structured_record_positions: tuple[TargetPosition, ...]
    feedback_positions: tuple[int, ...]
    grammar_targets: torch.Tensor
    scalar_targets: tuple[float, ...]
    structured_record_targets: tuple[Mapping[str, Any], ...]
    decoder_substates: tuple[Mapping[str, Any], ...]
    stage2_conditioning: str = "text_only"
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    visual_token_positions: tuple[int, ...] = ()
    context_position: int = 0


class Stage2QwenCollator:
    """Turn one frozen supervision record into the exact model sequence.

    ``record`` is the JSON object emitted by the frozen supervision stage.  It
    must contain command tokens and action boundaries.  Target positions are
    resolved against command atom indices and are rejected when they do not
    identify exactly one atom span.
    """

    def __init__(self, tokenizer, *, grammar_vocabulary: Mapping[str, int],
                 max_length: int | None = None,
                 stage2_conditioning: str = "multiview_vlm"):
        if stage2_conditioning not in {"text_only", "multiview_vlm"}:
            raise ValueError("unknown Stage2 conditioning mode")
        if stage2_conditioning == "multiview_vlm" and not all(
            hasattr(tokenizer, field) for field in ("tokenizer", "image_processor", "apply_chat_template")
        ):
            raise TypeError("multiview Stage2 requires an official image-capable processor")
        self.stage2_conditioning = stage2_conditioning
        self.processor = tokenizer if stage2_conditioning == "multiview_vlm" else None
        self.tokenizer = tokenizer.tokenizer if self.processor is not None else tokenizer
        self.grammar_vocabulary = dict(grammar_vocabulary)
        if sorted(self.grammar_vocabulary.values()) != list(range(len(self.grammar_vocabulary))):
            raise ValueError("grammar vocabulary must have contiguous frozen indices")
        self.max_length = max_length

    def _conditioning_prefix(self, record: Mapping[str, Any]) -> tuple[str, str, list[int], torch.Tensor | None, torch.Tensor | None, tuple[int, ...], int]:
        conditioning = record.get("conditioning")
        if not isinstance(conditioning, Mapping):
            raise ValueError("frozen Stage2 record requires explicit conditioning X")
        text = conditioning.get("text")
        source = conditioning.get("source")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("conditioning X must be nonempty source text")
        if source not in {"dataset_annotation", "legacy_pointercad_adapter_constant", "expanded9k_fixed_constant"}:
            raise ValueError("conditioning X needs an approved source annotation")
        if source == "legacy_pointercad_adapter_constant" and text != LEGACY_ADAPTER_INSTRUCTION:
            raise ValueError("legacy adapter instruction does not match its recorded source")
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise TypeError("Stage2 tokenizer must implement the Pointer-CAD chat template")
        if self.stage2_conditioning == "multiview_vlm":
            if text != LEGACY_ADAPTER_INSTRUCTION:
                raise ValueError("multiview baseline requires the exact fixed instruction")
            paths = record.get("image_paths")
            supplied_images = record.get("images")
            if (paths is None) == (supplied_images is None):
                raise ValueError("multiview conditioning requires exactly one eight-view image source")
            views = paths if paths is not None else supplied_images
            if not isinstance(views, (list, tuple)) or len(views) != 8:
                raise ValueError("multiview conditioning requires exactly eight ordered images")
            from PIL import Image
            images = []
            if paths is not None:
                for path in paths:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
            else:
                for image in supplied_images:
                    if not isinstance(image, Image.Image):
                        raise TypeError("multiview images must be PIL images")
                    images.append(image.convert("RGB"))
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": text})
            messages = [{"role": "system", "content": POINTERCAD_SYSTEM_INSTRUCTION},
                        {"role": "user", "content": content}]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processed = self.processor(text=[rendered], images=images, return_tensors="pt")
            values = processed["input_ids"][0].tolist()
            grid = processed.get("image_grid_thw")
            if grid is None or grid.shape[0] != 8 or "pixel_values" not in processed:
                raise ValueError("Qwen2.5-VL processor did not produce eight image grids")
            image_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
            visual_positions = tuple(i for i, value in enumerate(values) if value == image_id)
            if not visual_positions:
                raise ValueError("Qwen2.5-VL processor produced no visual tokens")
            # One constant assistant-side token is reserved for current S_t.
            # It follows all visual tokens and precedes every command atom.
            slot = self._encode_atom(" ")
            if len(slot) != 1:
                raise ValueError("Stage2 context slot must be exactly one tokenizer token")
            values.extend(slot)
            return text, source, values, processed["pixel_values"], grid, visual_positions, len(values) - 1
        messages = [
            {"role": "system", "content": POINTERCAD_SYSTEM_INSTRUCTION},
            {"role": "user", "content": [{"type": "brep"}, {"type": "text", "text": text}]},
        ]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("Pointer-CAD chat template produced an empty conditioning prefix")
        return text, source, self._encode_atom(rendered), None, None, (), 0

    def _encode_atom(self, atom: str) -> list[int]:
        encoded = self.tokenizer(atom, add_special_tokens=False)["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        result = [int(value) for value in encoded]
        if not result:
            raise ValueError(f"tokenizer produced no tokens for command atom {atom!r}")
        return result

    @staticmethod
    def _position(target: Mapping[str, Any], spans: Sequence[TokenSpan], *, kind: str) -> TargetPosition:
        if not isinstance(target.get("position"), int):
            raise ValueError(f"{kind} target has no integer command position")
        command_position = int(target["position"])
        if not 0 <= command_position < len(spans):
            raise ValueError(f"{kind} target command position is outside command atoms")
        span = spans[command_position]
        action_index = target.get("action_index")
        if not isinstance(action_index, int):
            raise ValueError(f"{kind} target has no integer action index")
        if span.action_index != action_index:
            raise ValueError(f"{kind} target action index does not match command boundary")
        return TargetPosition(
            kind=kind,
            command_position=command_position,
            model_position=span.start,
            action_index=action_index,
            slot=str(target.get("slot", "")),
            subposition=target.get("subposition"),
            target_index=target.get("target_index"),
            target_candidate=target.get("target_candidate"),
        )

    @staticmethod
    def _located_record_targets(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        """Locate positionless frozen numeric records at unique typed markers."""
        command = record["command"]
        atoms = command["tokens"]
        markers = {"FRAME3": ("<frame>",), "VECTOR3": ("<vec>",),
                   "PLANE3": ("<plane>", "<frame>")}
        located = []
        for target in record.get("structured_record_targets", ()):
            if isinstance(target.get("position"), int):
                located.append(target)
                continue
            kind = str(target["record_type"])
            if kind not in markers:
                raise ValueError(f"structured_record {kind} has no position or known typed marker")
            action_index = target["action_index"]
            boundaries = [item for item in command["action_boundaries"] if item["action_index"] == action_index]
            if len(boundaries) != 1:
                raise ValueError("structured_record action boundary is not unique")
            boundary = boundaries[0]
            positions = [index for index in range(boundary["start"], boundary["end"])
                         if atoms[index] in markers[kind]]
            if len(positions) != 1:
                raise ValueError(f"structured_record {kind} typed marker is not unique in action {action_index}")
            located.append({**target, "position": positions[0]})
        return tuple(located)

    def __call__(self, records: Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> Stage2Sequence | dict[str, Any]:
        if isinstance(records, Mapping):
            return self.collate_one(records)
        rows = [self.collate_one(record) for record in records]
        if not rows:
            raise ValueError("Stage2 collator requires at least one record")
        width = max(int(row.input_ids.shape[0]) for row in rows)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        positions = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            length = row.input_ids.shape[0]
            ids[index, :length] = row.input_ids
            mask[index, :length] = row.attention_mask
            positions[index, :length] = row.position_ids
        return {"input_ids": ids, "attention_mask": mask, "position_ids": positions, "rows": tuple(rows)}

    def collate_one(self, record: Mapping[str, Any]) -> Stage2Sequence:
        command = record.get("command")
        if not isinstance(command, Mapping):
            raise ValueError("frozen record requires command")
        atoms = command.get("tokens")
        boundaries = command.get("action_boundaries")
        if not isinstance(atoms, list) or not all(isinstance(atom, str) for atom in atoms):
            raise ValueError("command.tokens must be a list of strings")
        if not isinstance(boundaries, list):
            raise ValueError("command.action_boundaries must be present")

        (conditioning_text, conditioning_source, values, pixel_values,
         image_grid_thw, visual_positions, context_position) = self._conditioning_prefix(record)
        conditioning_end = len(values)
        spans: list[TokenSpan] = []
        for command_position, atom in enumerate(atoms):
            start = len(values)
            values.extend(self._encode_atom(atom))
            action_index = next((int(b["action_index"]) for b in boundaries if b["start"] <= command_position < b["end"]), None)
            spans.append(TokenSpan(command_position, start, len(values), atom, action_index))
        if self.max_length is not None and len(values) > self.max_length:
            raise ValueError("frozen command exceeds collator max_length")

        action_spans = []
        for boundary in boundaries:
            start, end, action_index = boundary.get("start"), boundary.get("end"), boundary.get("action_index")
            if not all(isinstance(value, int) for value in (start, end, action_index)) or not 0 <= start <= end <= len(spans):
                raise ValueError("invalid action boundary")
            model_start = spans[start].start if start < end else (spans[start - 1].end if start else conditioning_end)
            model_end = spans[end - 1].end if start < end else model_start
            action_spans.append(ActionSpan(action_index, model_start, model_end))

        located_records = self._located_record_targets(record)
        grammar = []
        grammar_labels = []
        supervised_atoms = {
            int(target["position"])
            for category in (record.get("pointer_targets", ()), record.get("parameter_targets", ()), located_records)
            for target in category
            if isinstance(target.get("position"), int)
        }
        for span in spans:
            if span.command_position in supervised_atoms or _numeric_atom(span.token):
                continue
            if span.token not in self.grammar_vocabulary:
                raise ValueError(f"grammar atom is absent from frozen vocabulary: {span.token!r}")
            grammar.append(TargetPosition("grammar", span.command_position, span.start - 1, span.action_index if span.action_index is not None else len(boundaries), span.token))
            grammar_labels.append(self.grammar_vocabulary[span.token])
        pointers = tuple(self._position(target, spans, kind="pointer") for target in record.get("pointer_targets", ()))
        scalars = tuple(self._position(target, spans, kind="scalar") for target in record.get("parameter_targets", ()))
        records = tuple(self._position(target, spans, kind="structured_record") for target in located_records)
        feedback = tuple(spans[item.command_position].end for item in pointers)
        input_ids = torch.tensor(values, dtype=torch.long)
        attention = torch.ones_like(input_ids)
        position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)
        grammar_targets = torch.tensor(grammar_labels, dtype=torch.long)
        scalar_targets = tuple(float(target["value"]) for target in record.get("parameter_targets", ()))
        decoder_substates = tuple(target.get("decoder_substate", {}) for target in record.get("pointer_targets", ()))
        result = Stage2Sequence(
            input_ids, attention, position_ids, conditioning_text, conditioning_source, conditioning_end,
            tuple(spans), tuple(action_spans), tuple(grammar),
            pointers, scalars, records, feedback, grammar_targets, scalar_targets,
            tuple(record.get("structured_record_targets", ())), decoder_substates,
            self.stage2_conditioning, pixel_values, image_grid_thw,
            visual_positions, context_position,
        )
        self._assert_one_to_one(result)
        return result

    @staticmethod
    def _assert_one_to_one(sequence: Stage2Sequence) -> None:
        all_targets = sequence.grammar_positions + sequence.pointer_positions + sequence.scalar_positions + sequence.structured_record_positions
        for target in all_targets:
            if not 0 <= target.model_position < sequence.input_ids.shape[0]:
                raise AssertionError("supervision target is outside model sequence")
        pointer_positions = [target.model_position for target in sequence.pointer_positions]
        if len(pointer_positions) != len(sequence.feedback_positions):
            raise AssertionError("pointer and feedback maps are not aligned")
        if pointer_positions != sorted(set(pointer_positions)):
            raise ValueError("pointer slots must occupy unique increasing causal positions")
        keys = [(target.action_index, target.slot, repr(target.subposition), target.command_position) for target in sequence.pointer_positions]
        if len(keys) != len(set(keys)):
            raise AssertionError("pointer target map contains duplicate slot entries")


def frozen_grammar_vocabulary(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Freeze the discrete grammar head labels before any training batches."""
    tokens = set()
    for record in records:
        supervised = {int(target["position"])
                      for category in ("pointer_targets", "parameter_targets", "structured_record_targets")
                      for target in record.get(category, ()) if isinstance(target.get("position"), int)}
        for position, token in enumerate(record["command"]["tokens"]):
            if position not in supervised and not _numeric_atom(token):
                tokens.add(token)
    return {token: index for index, token in enumerate(sorted(tokens))}


def _numeric_atom(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def collate_stage2(record, tokenizer, **kwargs):
    """Convenience entry point used by training and integration probes."""
    return Stage2QwenCollator(tokenizer, **kwargs)(record)
