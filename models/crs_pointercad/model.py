"""CRS-native Pointer-CAD model integrated with the real decoder path."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .candidates import CandidateBank, CandidateView, ContextView
from .contracts import ExecutionState, ExternalKey, ModelMode, PointerType
from .encoders import BodyCandidateEncoder, RegistryContextEncoder, default_encoders
from .grammar import ActionAST
from .heads import ExpandedLoss, NumericHeads, PointerFeedback, TypedPointerHeads


def _resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    values = {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if value not in values:
        raise ValueError(f"unsupported Qwen dtype {value!r}")
    return values[value]


class _TinyBackbone(nn.Module):
    """Dependency-light local backbone used only when no Qwen is requested."""

    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, nhead=max(1, min(8, hidden_dim // 8)), batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        del input_ids, kwargs
        padding = None if attention_mask is None else ~attention_mask.to(dtype=torch.bool)
        length = inputs_embeds.shape[1]
        causal = torch.triu(torch.ones((length, length), device=inputs_embeds.device, dtype=torch.bool), diagonal=1)
        return SimpleNamespace(last_hidden_state=self.encoder(inputs_embeds, mask=causal, src_key_padding_mask=padding))


@dataclass
class ExpandedForward:
    grammar_logits: torch.Tensor
    pointer_logits: tuple[torch.Tensor, ...]
    candidate_banks: tuple[CandidateBank, ...]
    hidden_states: torch.Tensor
    context_features: torch.Tensor | None = None
    pointer_logits_by_example: tuple[tuple[torch.Tensor, ...], ...] = ()
    candidate_banks_by_example: tuple[tuple[CandidateBank, ...], ...] = ()
    scalar_predictions: torch.Tensor | None = None
    record_predictions: Mapping[str, torch.Tensor] | None = None
    pointer_hidden_by_example: tuple[tuple[torch.Tensor, ...], ...] = ()


class CRSExpandedPointerCAD(nn.Module):
    mode = ModelMode.CRS_EXPANDED_POINTERCAD

    def __init__(self, hidden_dim: int = 256, grammar_vocab_size: int = 512, pointer_dim: int = 128,
                 qwen_model: str | None = None, base_model: nn.Module | None = None,
                 load_base_model: bool = True, vocab_size: int = 2048,
                 dtype: str | torch.dtype = "bf16", use_lora: bool = True,
                 lora_rank: int = 8, lora_alpha: int = 32, lora_dropout: float = 0.1,
                 use_body_relative_age: bool = True, body_pooling: str = "mean",
                 registry_context_mode: str = "NONE", enable_historical_face: bool = True,
                 enable_historical_edge: bool = True, pointer_feedback: bool = True,
                 typed_pointer_scales: bool = True, use_native_brep: bool = True,
                 crv_channels: int = 12, surf_channels: int = 8,
                 numeric_heads: bool = True, ragged_pointer_batching: bool = True,
                 autoregressive_pointer_feedback: bool = True):
        super().__init__()
        self.configured_base_model = qwen_model
        self.vlm_backbone = False
        self.vision_forward_count = 0
        if base_model is not None:
            self.base_model = base_model
            self.vlm_backbone = all(hasattr(base_model, name) for name in ("visual", "model", "get_rope_index"))
            self.base_model_source = "injected"
            self.base_model_dtype = str(next(base_model.parameters()).dtype)
            self.peft_lora = False
        elif qwen_model is not None and load_base_model:
            try:
                torch_dtype = _resolve_dtype(dtype)
                from transformers import AutoConfig
                self.vlm_backbone = AutoConfig.from_pretrained(qwen_model).model_type == "qwen2_5_vl"
                if self.vlm_backbone:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        qwen_model, torch_dtype=torch_dtype, attn_implementation="flash_attention_2")
                    self.base_model.requires_grad_(False)
                    self.base_model.visual.eval()
                else:
                    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
                    self.base_model = Qwen2Model.from_pretrained(qwen_model, torch_dtype=torch_dtype)
                self.base_model_dtype = str(torch_dtype)
                self.peft_lora = False
                if use_lora:
                    from peft import LoraConfig, TaskType, get_peft_model, inject_adapter_in_model
                    lora_config = LoraConfig(
                        task_type=None if self.vlm_backbone else TaskType.CAUSAL_LM,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        inference_mode=False, r=lora_rank, lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    if self.vlm_backbone:
                        self.base_model.model = inject_adapter_in_model(lora_config, self.base_model.model)
                    else:
                        self.base_model = get_peft_model(self.base_model, lora_config)
                    self.peft_lora = True
                self.base_model_source = "pretrained"
            except Exception as exc:
                raise RuntimeError(f"failed to load configured base_model {qwen_model!r}") from exc
        else:
            self.base_model = _TinyBackbone(vocab_size, hidden_dim)
            self.base_model_source = "tiny_smoke"
            self.base_model_dtype = str(next(self.base_model.parameters()).dtype)
            self.peft_lora = False
        self.hidden_dim = int(self.base_model.config.hidden_size)
        self.grammar_head = nn.Linear(self.hidden_dim, grammar_vocab_size)
        self.hidden_projection = nn.Identity()
        self.encoders = nn.ModuleDict()
        encoders = default_encoders()
        encoders[PointerType.BODY] = BodyCandidateEncoder(pooling=body_pooling, use_relative_age=use_body_relative_age)
        self.encoders.update({ptype.value: encoder for ptype, encoder in encoders.items()})
        self.registry_context = RegistryContextEncoder(self.hidden_dim, registry_context_mode)
        self.context_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.brep_projection = nn.Linear(pointer_dim, self.hidden_dim)
        self.query_heads = TypedPointerHeads(self.hidden_dim, pointer_dim, typed_pointer_scales)
        self.numeric_heads = NumericHeads(self.hidden_dim)
        self.feedback = PointerFeedback(self.hidden_dim, pointer_dim) if pointer_feedback else None
        self.loss_fn = ExpandedLoss()
        self.numeric_heads_enabled = numeric_heads
        self.ragged_pointer_batching = ragged_pointer_batching
        self.autoregressive_pointer_feedback = autoregressive_pointer_feedback
        self.enable_historical_face = enable_historical_face
        self.enable_historical_edge = enable_historical_edge
        self.use_native_brep = use_native_brep
        self.brep = None
        if use_native_brep:
            try:
                from models.brep_embed import UVNetEmbedder
                self.brep = UVNetEmbedder(crv_channels, surf_channels, pointer_dim, pointer_dim, self.hidden_dim)
            except ImportError:
                # Import is optional for dependency-light unit tests; production
                # configs with a DGL graph fail closed in _native_maps.
                self.brep = None

    def training_config(self) -> dict[str, Any]:
        return {
            "dtype": self.base_model_dtype,
            "peft_lora": self.peft_lora,
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "frozen_parameters": sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad),
            "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if self.peft_lora else [],
            "vlm_backbone": self.vlm_backbone,
            "vision_tower": "frozen" if self.vlm_backbone else None,
            "vision_merger": "frozen" if self.vlm_backbone else None,
            "trainable_groups": {name: sum(p.numel() for p in module.parameters() if p.requires_grad)
                                 for name, module in (("language_lora", self.base_model),
                                                      ("context_adapter", self.context_projection),
                                                      ("brep_adapter", self.brep_projection),
                                                      ("feedback_adapter", self.feedback),
                                                      ("uvnet_gnn", self.brep),
                                                      ("typed_pointer_heads", self.query_heads),
                                                      ("semantic_candidate_encoders", self.encoders),
                                                      ("grammar_head", self.grammar_head),
                                                      ("scalar_record_heads", self.numeric_heads),
                                                      ("registry_context", self.registry_context))
                                 if module is not None},
        }

    def _native_maps(self, states: Sequence[ExecutionState], breps=None):
        maps = [(dict(state.native_face_embeddings), dict(state.native_edge_embeddings), dict(state.native_body_face_embeddings)) for state in states]
        prepared = breps if isinstance(breps, (list, tuple)) and breps and hasattr(breps[0], "face_keys") else None
        if prepared is not None:
            if self.brep is None:
                raise RuntimeError("native Pointer-CAD UV-Net encoder is unavailable")
            if len(prepared) != len(states):
                raise ValueError("prepared BRep batch does not align with execution states")
            if len(prepared) == 1 and not prepared[0].face_keys and not prepared[0].edge_keys:
                return maps
            import dgl
            device = next(self.brep.parameters()).device
            graphs = [item.graph.to(device) for item in prepared]
            pointer_edge, pointer_face, _, _ = self.brep(dgl.batch(graphs))
            for batch_index, (state, item) in enumerate(zip(states, prepared)):
                if len(item.face_keys) != len(pointer_face[batch_index]) or len(item.edge_keys) != len(pointer_edge[batch_index]):
                    raise ValueError("prepared BRep rows do not match UV-Net output rows")
                face_rows = {key: pointer_face[batch_index][index] for index, key in enumerate(item.face_keys)}
                edge_rows = {key: pointer_edge[batch_index][index] for index, key in enumerate(item.edge_keys)}
                if item.loose_edge_keys:
                    features = torch.from_numpy(item.loose_edge_features).to(device)
                    if features.shape[0] == 1 and self.training:
                        # UVNetCurveEncoder contains BatchNorm1d; duplicate a
                        # singleton only for its batch statistics.
                        loose = self.brep.curv_encoder(torch.cat((features, features), dim=0))[:1]
                    else:
                        loose = self.brep.curv_encoder(features)
                    edge_rows.update({key: loose[index] for index, key in enumerate(item.loose_edge_keys)})
                maps[batch_index][0].update(face_rows)
                maps[batch_index][1].update(edge_rows)
                for key, embedding in face_rows.items():
                    owner = getattr(key, "owner", None)
                    if owner is not None:
                        maps[batch_index][2].setdefault(owner, []).append(embedding)
                        maps[batch_index][2].setdefault(ExternalKey(str(owner)), []).append(embedding)
            return maps
        if breps is None or self.brep is None:
            for batch_index, state in enumerate(states):
                snapshot = state.active_brep_state
                if snapshot is None or not hasattr(snapshot, "candidates"):
                    continue
                try:
                    from cadquery2crs.representation import RefKind
                    face_candidates = tuple(snapshot.candidates(RefKind.FACE, include_historical=True))
                    edge_candidates = tuple(snapshot.candidates(RefKind.EDGE, include_historical=True))
                except Exception:
                    face_candidates = tuple(snapshot.candidates("FACE", include_historical=True))
                    edge_candidates = tuple(snapshot.candidates("EDGE", include_historical=True))
                if state.historical_face_provider is not None:
                    for candidate in face_candidates:
                        if getattr(candidate, "metadata", {}).get("historical"):
                            maps[batch_index][0][candidate.key] = state.historical_face_provider(candidate)
                if state.historical_edge_provider is not None:
                    for candidate in edge_candidates:
                        if getattr(candidate, "metadata", {}).get("historical"):
                            maps[batch_index][1][candidate.key] = state.historical_edge_provider(candidate)
            return maps
        try:
            pointer_edge, pointer_face, _, _ = self.brep(breps)
        except Exception as exc:
            raise RuntimeError("native Pointer-CAD UV-Net B-Rep encoding failed") from exc
        for batch_index, state in enumerate(states):
            snapshot = state.active_brep_state
            if snapshot is None or not hasattr(snapshot, "candidates"):
                continue
            try:
                from cadquery2crs.representation import RefKind
                face_candidates = tuple(snapshot.candidates(RefKind.FACE, include_historical=True))
                edge_candidates = tuple(snapshot.candidates(RefKind.EDGE, include_historical=True))
            except Exception:
                face_candidates = tuple(snapshot.candidates("FACE", include_historical=True))
                edge_candidates = tuple(snapshot.candidates("EDGE", include_historical=True))
            active_faces = [candidate for candidate in face_candidates if not getattr(candidate, "metadata", {}).get("historical")]
            active_edges = [candidate for candidate in edge_candidates if not getattr(candidate, "metadata", {}).get("historical")]
            for candidate, embedding in zip(active_faces, pointer_face[batch_index]):
                maps[batch_index][0][candidate.key] = embedding
                if getattr(candidate, "owner", None) is not None:
                    maps[batch_index][2].setdefault(candidate.owner, []).append(embedding)
                    maps[batch_index][2].setdefault(ExternalKey(str(candidate.owner)), []).append(embedding)
            for candidate, embedding in zip(active_edges, pointer_edge[batch_index]):
                maps[batch_index][1][candidate.key] = embedding
            if state.historical_face_provider is not None:
                for candidate in face_candidates:
                    if getattr(candidate, "metadata", {}).get("historical") and candidate.key not in maps[batch_index][0]:
                        maps[batch_index][0][candidate.key] = state.historical_face_provider(candidate)
            if state.historical_edge_provider is not None:
                for candidate in edge_candidates:
                    if getattr(candidate, "metadata", {}).get("historical") and candidate.key not in maps[batch_index][1]:
                        maps[batch_index][1][candidate.key] = state.historical_edge_provider(candidate)
        return maps

    def _inject_native_brep_tokens(self, embeddings: torch.Tensor, input_ids: torch.Tensor, breps) -> torch.Tensor:
        """Preserve legacy Pointer-CAD graph-aware face/edge token context."""
        if isinstance(breps, (list, tuple)) and breps and hasattr(breps[0], "face_keys"):
            # Prepared graphs have already passed through UV-Net in _native_maps.
            return embeddings
        if breps is None or self.brep is None:
            return embeddings
        edge_pointer, face_pointer, _, _ = self.brep(breps)
        edge_values = torch.cat(tuple(edge_pointer), dim=0) if edge_pointer else embeddings.new_empty((0, 128))
        face_values = torch.cat(tuple(face_pointer), dim=0) if face_pointer else embeddings.new_empty((0, 128))
        edge_mask = input_ids == 151667
        face_mask = input_ids == 151670
        result = embeddings.clone()
        if int(edge_mask.sum()) == edge_values.shape[0] and edge_values.shape[0]:
            result = result.masked_scatter(edge_mask.unsqueeze(-1), self.brep_projection(edge_values.to(result.dtype)).expand_as(result[edge_mask]))
        if int(face_mask.sum()) == face_values.shape[0] and face_values.shape[0]:
            result = result.masked_scatter(face_mask.unsqueeze(-1), self.brep_projection(face_values.to(result.dtype)).expand_as(result[face_mask]))
        return result

    def candidate_view(self, state: ExecutionState, pointer_type: PointerType, decoder_substate=None, *, native_maps=None) -> CandidateBank:
        encoders = {ptype: self.encoders[ptype.value] for ptype in PointerType}
        if native_maps is not None:
            face_map, edge_map, body_face_map = native_maps
            state = state.clone()
            state.native_face_embeddings = face_map
            state.native_edge_embeddings = edge_map
            state.native_body_face_embeddings = {key: tuple(value) for key, value in body_face_map.items()}
        bank = CandidateView(state, encoders=encoders).bank(pointer_type, decoder_substate)
        if pointer_type is PointerType.FACE and not self.enable_historical_face:
            bank = CandidateBank.build(pointer_type, [entry for entry in bank if not entry.legal_metadata.get("historical")])
        if pointer_type is PointerType.EDGE and not self.enable_historical_edge:
            bank = CandidateBank.build(pointer_type, [entry for entry in bank if not entry.legal_metadata.get("historical")])
        return bank

    def _banks(self, state: ExecutionState, pointer_slots, decoder_substates=(), native_maps=None):
        return tuple(self.candidate_view(state, pointer_type, decoder_substates[index] if index < len(decoder_substates) else None, native_maps=native_maps) for index, pointer_type in enumerate(pointer_slots))

    def _context(self, state: ExecutionState, native_maps=None):
        encoders = {ptype: self.encoders[ptype.value] for ptype in PointerType}
        view_state = state
        if native_maps is not None:
            face_map, edge_map, body_face_map = native_maps
            view_state = state.clone()
            view_state.native_face_embeddings = face_map
            view_state.native_edge_embeddings = edge_map
            view_state.native_body_face_embeddings = {key: tuple(value) for key, value in body_face_map.items()}
        context_view = ContextView(view_state, CandidateView(view_state, encoders=encoders), self.registry_context)
        features = context_view.features()
        if features.numel() == 0:
            return features
        return features.to(next(self.parameters()).device)

    def forward(self, hidden_states: torch.Tensor | None = None, state: ExecutionState | None = None,
                pointer_slots: Sequence[PointerType] = (), decoder_substates: Sequence[Mapping[str, Any] | None] = (),
                input_ids: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                breps=None, states: Sequence[ExecutionState] | None = None,
                pixel_values=None, image_grid_thw=None, context_position: int = 0,
                visual_token_positions: Sequence[int] = (), **kwargs) -> ExpandedForward:
        del kwargs
        state = state or (states[0] if states else ExecutionState())
        states = tuple(states or (state,))
        if states and len(states) > 1 and pointer_slots and isinstance(pointer_slots[0], (list, tuple)):
            return self.forward_ragged(
                input_ids=input_ids, attention_mask=attention_mask, states=states,
                pointer_specs=pointer_slots, decoder_substates=decoder_substates,
                breps=breps,
            )
        native_maps = self._native_maps(states, breps)
        context = self._context(state, native_maps=native_maps[0] if native_maps else None)
        if input_ids is not None:
            embeddings, context = self._prepare_embeddings(
                input_ids, state, attention_mask=attention_mask, breps=breps,
                native_maps=native_maps[0] if native_maps else None,
                pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                visual_token_positions=visual_token_positions, context_position=context_position)
            position_ids = self._vlm_position_ids(input_ids, attention_mask, image_grid_thw)
            hidden_states = self._backbone_forward(embeddings, attention_mask, position_ids).last_hidden_state
        elif hidden_states is None:
            raise ValueError("CRSExpandedPointerCAD requires input_ids or hidden_states")
        hidden_states = self.hidden_projection(hidden_states)
        grammar_logits = self.grammar_head(hidden_states.to(self.grammar_head.weight.dtype))
        scalar_predictions, record_predictions = self.numeric_heads(hidden_states)
        slots = tuple(pointer_slots)
        if slots and len(states) > 1:
            raise ValueError("batched pointer slots must be nested per-example; use forward_ragged")
        banks = self._banks(state, slots, decoder_substates, native_maps=native_maps[0] if native_maps else None)
        def slot_hidden(index):
            return hidden_states[min(index, hidden_states.shape[0] - 1)] if hidden_states.ndim == 2 else hidden_states[min(index, hidden_states.shape[0] - 1), -1]
        pointer_logits = tuple(self.query_heads.score(slot_hidden(index), bank, ptype) for index, (ptype, bank) in enumerate(zip(slots, banks)))
        return ExpandedForward(grammar_logits, pointer_logits, banks, hidden_states, context,
                               scalar_predictions=scalar_predictions, record_predictions=record_predictions)

    def encode_visual(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> torch.Tensor:
        """Use Qwen2.5-VL's frozen visual tower and merger exactly once."""
        if not self.vlm_backbone:
            raise ValueError("visual features require the Qwen2.5-VL backbone")
        self.base_model.visual.eval()
        device = next(self.base_model.visual.parameters()).device
        with torch.no_grad():
            features = self.base_model.visual(pixel_values.to(device=device, dtype=self.base_model.visual.dtype),
                                              grid_thw=image_grid_thw.to(device))
        self.vision_forward_count += 1
        return features

    def _vlm_position_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                          image_grid_thw: torch.Tensor | None):
        if not self.vlm_backbone:
            return None
        grid = image_grid_thw.to(input_ids.device) if image_grid_thw is not None else None
        position_ids, _ = self.base_model.get_rope_index(input_ids, grid, None, None, attention_mask)
        return position_ids.to(input_ids.device)

    def _backbone_forward(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None,
                          position_ids: torch.Tensor | None = None, **kwargs):
        if self.vlm_backbone:
            kwargs.setdefault("use_cache", False)
            return self.base_model.model(inputs_embeds=embeddings, attention_mask=attention_mask,
                                         position_ids=position_ids, **kwargs)
        return self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask, **kwargs)

    def _prepare_embeddings(self, input_ids: torch.Tensor, state: ExecutionState, *, attention_mask=None,
                            breps=None, native_maps=None, pixel_values=None, image_grid_thw=None,
                            visual_token_positions=(), context_position=0,
                            visual_features=None):
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        if self.vlm_backbone:
            image_mask = input_ids == self.base_model.config.image_token_id
            actual_positions = tuple(torch.nonzero(image_mask[0], as_tuple=False).flatten().tolist())
            if visual_token_positions and actual_positions != tuple(visual_token_positions):
                raise ValueError("visual token positions changed after action slicing")
            if actual_positions:
                if image_grid_thw is None or (pixel_values is None and visual_features is None):
                    raise ValueError("visual tokens require real image pixels and image grids")
                features = visual_features if visual_features is not None else self.encode_visual(pixel_values, image_grid_thw)
                if len(actual_positions) != features.shape[0]:
                    raise ValueError("Qwen2.5-VL image features and image tokens do not match")
                embeddings = embeddings.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(embeddings), features.to(embeddings.dtype).flatten())
            elif pixel_values is not None:
                raise ValueError("image pixels were supplied without Qwen visual tokens")
        else:
            embeddings = self._inject_native_brep_tokens(embeddings, input_ids, breps)
        if not 0 <= context_position < embeddings.shape[1] or context_position in visual_token_positions:
            raise ValueError("Stage2 context position overlaps or escapes visual tokens")
        context = self._context(state, native_maps=native_maps)
        if context.numel():
            embeddings = embeddings.clone()
            embeddings[:, context_position, :] = embeddings[:, context_position, :] + self.context_projection(context.mean(0)).to(embeddings.dtype)
        if native_maps is not None and native_maps[0]:
            native = torch.stack(tuple(native_maps[0].values())).mean(0).to(self.brep_projection.weight.device, self.brep_projection.weight.dtype)
            embeddings = embeddings.clone()
            embeddings[:, context_position, :] = embeddings[:, context_position, :] + self.brep_projection(native).to(embeddings.dtype)
        return embeddings, context

    def forward_ragged(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None,
                       states: Sequence[ExecutionState], pointer_specs: Sequence[Sequence[Any]],
                       decoder_substates: Sequence[Sequence[Mapping[str, Any] | None]] = (),
                       teacher_indices: Sequence[Sequence[int | None]] = (),
                       feedback_positions: Sequence[Sequence[int]] = (), breps=None,
                       pixel_values=None, image_grid_thw=None,
                       context_positions: Sequence[int] = (),
                       visual_token_positions: Sequence[Sequence[int]] = (),
                       visual_features=None) -> ExpandedForward:
        """Loss-bearing autoregressive path for ragged pointer slots.

        Each example owns its state, banks, slot types, substates and candidate
        counts. GT feedback is placed at the explicit post-span boundaries
        before one causal backbone forward per action.
        """
        if len(states) != input_ids.shape[0] or len(pointer_specs) != input_ids.shape[0]:
            raise ValueError("states and pointer_specs must align with input batch")
        hidden_rows, grammar_rows, scalar_rows = [], [], []
        record_rows = []
        pointer_rows, bank_rows, pointer_hidden_rows = [], [], []
        for batch_index, state in enumerate(states):
            ids = input_ids[batch_index:batch_index + 1]
            mask = attention_mask[batch_index:batch_index + 1] if attention_mask is not None else None
            example_breps = [breps[batch_index]] if isinstance(breps, (list, tuple)) and breps and hasattr(breps[0], "face_keys") else breps
            native_maps = self._native_maps((state,), example_breps)[0] if example_breps is not None else self._native_maps((state,))[0]
            row_pixels = pixel_values[batch_index] if isinstance(pixel_values, (list, tuple)) else pixel_values
            row_grid = image_grid_thw[batch_index] if isinstance(image_grid_thw, (list, tuple)) else image_grid_thw
            row_features = visual_features[batch_index] if isinstance(visual_features, (list, tuple)) else visual_features
            row_visual_positions = visual_token_positions[batch_index] if batch_index < len(visual_token_positions) else ()
            context_position = context_positions[batch_index] if batch_index < len(context_positions) else 0
            embeddings, context = self._prepare_embeddings(
                ids, state, attention_mask=mask, breps=example_breps, native_maps=native_maps,
                pixel_values=row_pixels, image_grid_thw=row_grid,
                visual_token_positions=row_visual_positions, context_position=context_position,
                visual_features=row_features)
            slots = pointer_specs[batch_index]
            substates = decoder_substates[batch_index] if batch_index < len(decoder_substates) else ()
            targets = teacher_indices[batch_index] if batch_index < len(teacher_indices) else ()
            example_logits, example_banks, example_hidden = [], [], []
            # Banks and teacher feedback are known before the training forward.
            # Build the complete causal embedding stream first, then execute the
            # backbone once. This removes the old one-forward-per-pointer loop.
            banks = []
            feedback_by_position = {}
            for slot_index, slot in enumerate(slots):
                if isinstance(slot, Mapping):
                    position = int(slot["position"])
                    feedback_position = slot.get("feedback_position")
                    pointer_type = slot["pointer_type"]
                    substate = slot.get("decoder_substate")
                    target_index = slot.get("target_index")
                    target_key = slot.get("target_key")
                else:
                    position, pointer_type = slot
                    row_feedback = feedback_positions[batch_index] if batch_index < len(feedback_positions) else ()
                    feedback_position = row_feedback[slot_index] if slot_index < len(row_feedback) else None
                    substate = substates[slot_index] if slot_index < len(substates) else None
                    target_index = targets[slot_index] if slot_index < len(targets) else None
                    target_key = None
                pointer_type = pointer_type if isinstance(pointer_type, PointerType) else PointerType(pointer_type)
                if not 0 <= position < ids.shape[1]:
                    raise ValueError("pointer slot position is outside action sequence")
                bank = self.candidate_view(state, pointer_type, substate, native_maps=native_maps)
                banks.append((position, pointer_type, bank))
                if target_key is not None:
                    if target_index is not None:
                        raise ValueError("pointer slot cannot specify both target_key and target_index")
                    target_index = bank.external_to_index(target_key)
                if target_index is not None:
                    if not 0 <= int(target_index) < len(bank):
                        raise IndexError(f"teacher target {target_index} is outside {pointer_type.value} bank")
                    feedback = self.feedback_for(pointer_type, bank, int(target_index))
                    if self.feedback is not None:
                        if not isinstance(feedback_position, int) or not position < feedback_position <= ids.shape[1]:
                            raise ValueError("teacher pointer requires explicit post-span feedback_position")
                        feedback_by_position[feedback_position] = feedback_by_position.get(feedback_position, 0) + feedback.to(embeddings.dtype)
            if feedback_by_position:
                embeddings = embeddings.clone()
            for start, feedback in feedback_by_position.items():
                embeddings[:, start:, :] = embeddings[:, start:, :] + feedback
            position_ids = self._vlm_position_ids(ids, mask, row_grid)
            final_hidden = self._backbone_forward(embeddings, mask, position_ids).last_hidden_state
            final_hidden = final_hidden.to(self.grammar_head.weight.dtype)
            for position, pointer_type, bank in banks:
                h_j = final_hidden[:, position, :].squeeze(0)
                example_hidden.append(h_j)
                example_logits.append(self.query_heads.score(h_j, bank, pointer_type))
                example_banks.append(bank)
            hidden_rows.append(final_hidden.squeeze(0))
            grammar_rows.append(self.grammar_head(final_hidden.squeeze(0)))
            scalar_prediction, record_prediction = self.numeric_heads(final_hidden.squeeze(0))
            scalar_rows.append(scalar_prediction)
            record_rows.append(record_prediction)
            pointer_rows.append(tuple(example_logits))
            bank_rows.append(tuple(example_banks))
            pointer_hidden_rows.append(tuple(example_hidden))
        return ExpandedForward(
            grammar_logits=torch.stack(grammar_rows), pointer_logits=tuple(logit for row in pointer_rows for logit in row),
            candidate_banks=tuple(bank for row in bank_rows for bank in row), hidden_states=torch.stack(hidden_rows),
            context_features=None, pointer_logits_by_example=tuple(pointer_rows),
            candidate_banks_by_example=tuple(bank_rows), scalar_predictions=torch.stack(scalar_rows),
            record_predictions={name: torch.stack([row[name] for row in record_rows]) for name in record_rows[0]},
            pointer_hidden_by_example=tuple(pointer_hidden_rows),
        )

    def feedback_for(self, pointer_type: PointerType, bank: CandidateBank, selected_index: int) -> torch.Tensor:
        embedding = bank.embeddings()[selected_index]
        return embedding.new_zeros(self.hidden_dim) if self.feedback is None else self.feedback(pointer_type, embedding)

    def teacher_forced_decode(self, input_ids, attention_mask, state: ExecutionState, pointer_positions: Sequence[int], pointer_types: Sequence[PointerType], recorded_indices: Sequence[int], feedback_positions: Sequence[int], decoder_substates=(), breps=None):
        """Run real base-LM hidden states and inject recorded GT feedback."""
        if not len(pointer_positions) == len(pointer_types) == len(recorded_indices) == len(feedback_positions):
            raise ValueError("teacher pointer positions and feedback boundaries must align")
        embeddings = self.base_model.get_input_embeddings()(input_ids)
        native_maps = self._native_maps((state,), breps)[0]
        before = self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        selected = []
        for slot_index, (position, pointer_type, target_index) in enumerate(zip(pointer_positions, pointer_types, recorded_indices)):
            boundary = feedback_positions[slot_index]
            if not position < boundary <= input_ids.shape[1]:
                raise ValueError("teacher feedback must follow its complete pointer span")
            substate = decoder_substates[slot_index] if slot_index < len(decoder_substates) else None
            bank = self.candidate_view(state, pointer_type, substate, native_maps=native_maps)
            if not 0 <= target_index < len(bank):
                raise IndexError("recorded GT pointer index is not in causal bank")
            selected.append(self.feedback_for(pointer_type, bank, target_index))
            if self.feedback is not None:
                embeddings = embeddings.clone()
                embeddings[:, boundary:, :] = embeddings[:, boundary:, :] + selected[-1].to(embeddings.dtype)
        after = self.base_model(inputs_embeds=embeddings, attention_mask=attention_mask).last_hidden_state
        return {"hidden_before": before, "hidden_after": after, "feedback": tuple(selected)}

    @torch.no_grad()
    def inference_pointer_decode(self, input_ids, attention_mask, state: ExecutionState, pointer_positions: Sequence[int], pointer_types: Sequence[PointerType], decoder_substates=(), breps=None, *, feedback_positions: Sequence[int], use_cache: bool | None = None, return_logits: bool = False, pixel_values=None, image_grid_thw=None, context_position: int = 0, visual_token_positions: Sequence[int] = (), visual_features=None):
        """Incremental inference using ``past_key_values`` when available."""
        native_maps = self._native_maps((state,), breps)[0]
        embeddings, _ = self._prepare_embeddings(
            input_ids, state, attention_mask=attention_mask, breps=breps, native_maps=native_maps,
            pixel_values=pixel_values, image_grid_thw=image_grid_thw,
            context_position=context_position, visual_token_positions=visual_token_positions,
            visual_features=visual_features)
        position_ids = self._vlm_position_ids(input_ids, attention_mask, image_grid_thw)
        selections = []
        logit_rows = []
        if len(pointer_positions) != len(pointer_types) or len(set(pointer_positions)) != len(pointer_positions):
            raise ValueError("pointer positions/types must be unique and aligned")
        if tuple(pointer_positions) != tuple(sorted(pointer_positions)):
            raise ValueError("pointer positions must be in causal order")
        if any(not 0 <= int(position) < input_ids.shape[1] for position in pointer_positions):
            raise ValueError("pointer position is outside input sequence")
        if len(feedback_positions) != len(pointer_positions):
            raise ValueError("pointer feedback boundaries must align with pointer positions")
        if any(not position < boundary <= input_ids.shape[1]
               for position, boundary in zip(pointer_positions, feedback_positions)):
            raise ValueError("predicted feedback must follow its complete pointer span")
        if any(boundary > next_position for boundary, next_position in zip(feedback_positions, pointer_positions[1:])):
            raise ValueError("pointer spans overlap or are out of causal order")
        pointer_by_position = {int(position): (index, pointer_type) for index, (position, pointer_type) in enumerate(zip(pointer_positions, pointer_types))}
        past = None
        running_feedback = None
        scheduled_feedback = {}
        cached = bool(getattr(self.base_model.config, "use_cache", True)) and not isinstance(self.base_model, _TinyBackbone) if use_cache is None else use_cache
        if cached and isinstance(self.base_model, _TinyBackbone):
            raise ValueError("tiny smoke backbone does not support KV cache")
        uncached_embeddings = embeddings.clone() if not cached else None
        for position in range(input_ids.shape[1]):
            if position in scheduled_feedback:
                feedback = scheduled_feedback[position]
                running_feedback = feedback if running_feedback is None else running_feedback + feedback
            current = embeddings[:, position:position + 1, :]
            if running_feedback is not None:
                current = current + running_feedback.to(current.dtype)
            if cached:
                kwargs = {"inputs_embeds": current, "past_key_values": past, "use_cache": True}
                try:
                    kwargs["cache_position"] = torch.tensor([position], device=current.device)
                    outputs = self._backbone_forward(current, None,
                        position_ids[..., position:position + 1] if position_ids is not None else None,
                        past_key_values=past, use_cache=True, cache_position=kwargs["cache_position"])
                except TypeError:
                    outputs = self._backbone_forward(current, None,
                        position_ids[..., position:position + 1] if position_ids is not None else None,
                        past_key_values=past, use_cache=True)
                hidden = outputs.last_hidden_state[:, -1, :]
                past = getattr(outputs, "past_key_values", None)
            else:
                uncached_embeddings[:, position:position + 1, :] = current
                if position not in pointer_by_position:
                    continue
                prefix_mask = attention_mask[:, :position + 1] if attention_mask is not None else None
                reference = self._backbone_forward(
                    uncached_embeddings[:, :position + 1, :], prefix_mask,
                    position_ids[..., :position + 1] if position_ids is not None else None).last_hidden_state
                hidden = reference[:, -1, :]
            if position not in pointer_by_position:
                continue
            slot_index, pointer_type = pointer_by_position[position]
            substate = decoder_substates[slot_index] if slot_index < len(decoder_substates) else None
            bank = self.candidate_view(state, pointer_type, substate, native_maps=native_maps)
            logits = self.query_heads.score(hidden.squeeze(0), bank, pointer_type)
            logit_rows.append(logits)
            selected_index = int(logits.argmax())
            selections.append((pointer_type, selected_index, bank.index_to_external(selected_index)))
            if self.feedback is not None:
                feedback = self.feedback_for(pointer_type, bank, selected_index)
                boundary = feedback_positions[slot_index]
                scheduled_feedback[boundary] = scheduled_feedback.get(boundary, 0) + feedback
        return (selections, tuple(logit_rows)) if return_logits else selections

    @torch.no_grad()
    def bounded_decode_and_execute(self, input_ids, attention_mask, state: ExecutionState,
                                   pointer_positions: Sequence[int], pointer_types: Sequence[PointerType],
                                   operation: str, fields_builder, executor, decoder_substates=(), breps=None,
                                   *, feedback_positions: Sequence[int]):
        """Inference smoke boundary: predicted pointers -> AST -> causal executor."""
        selections = self.inference_pointer_decode(
            input_ids, attention_mask, state, pointer_positions, pointer_types,
            decoder_substates=decoder_substates, breps=breps,
            feedback_positions=feedback_positions,
        )
        action = self.decode_action(operation, fields_builder(selections))
        execution = self.execute_decoded(executor, state, action)
        return {"selections": selections, "action": action, "execution": execution}

    def decode_action(self, operation: str, fields: Mapping[str, Any], declarations=()) -> ActionAST:
        action = ActionAST(operation, fields, tuple(declarations))
        action.validate()
        return action

    def execute_decoded(self, executor, state: ExecutionState, action: ActionAST):
        action.validate()
        return executor(state, action.to_crs_semantics())

    def loss(self, *, grammar_logits=None, grammar_targets=None, pointer_logits=(), pointer_positive=(), scalar_loss=None, record_loss=None,
             scalar_predictions=None, scalar_targets=None, record_predictions=None, record_targets=None):
        return self.loss_fn(grammar_logits, grammar_targets, pointer_logits, pointer_positive, scalar_loss, record_loss,
                            scalar_predictions, scalar_targets, record_predictions, record_targets)

    def training_loss(self, output: ExpandedForward, *, grammar_targets, pointer_positive=(),
                      scalar_targets=None, record_targets=None):
        """Canonical loss entry point for a model forward result."""
        logits = output.pointer_logits_by_example or (output.pointer_logits,)
        return self.loss(
            grammar_logits=output.grammar_logits,
            grammar_targets=grammar_targets,
            pointer_logits=logits,
            pointer_positive=pointer_positive,
            scalar_predictions=output.scalar_predictions,
            scalar_targets=scalar_targets,
            record_predictions=output.record_predictions,
            record_targets=record_targets,
        )


def build_pointercad(mode: str = ModelMode.LEGACY_POINTERCAD, **kwargs):
    mode = mode if isinstance(mode, ModelMode) else ModelMode(mode)
    if mode is ModelMode.CRS_EXPANDED_POINTERCAD:
        return CRSExpandedPointerCAD(**kwargs)
    from models.pointercad import PointerCAD
    return PointerCAD(**kwargs)
