import torch
import yaml
from types import SimpleNamespace

from models.crs_pointercad import (
    ActionAST,
    BodyRecord,
    BodyVersionRegistry,
    CandidateBank,
    CandidateEntry,
    CandidateView,
    ConstructionRecord,
    ConstructionRegistry,
    CRSExpandedPointerCAD,
    ExecutionState,
    ExternalKey,
    ModelMode,
    Operation,
    PointerFeedback,
    PointerType,
    pointer_bce_per_slot,
)
from models.crs_pointercad.encoders import BodyCandidateEncoder, ProfileCandidateEncoder


def test_prefix_candidate_bank_and_round_trip():
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", active=True)
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    bank = CandidateView(state).bank(PointerType.BODY)
    assert bank.external_to_index(body.key) == 0
    assert bank.index_to_external(0) == body.key


def test_body_candidate_view_passes_causal_metadata_to_encoder():
    body = BodyRecord(ExternalKey("body_a"), 2, "ExtrudeFeature", active=False, geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]}, spatial={"extent": [1, 2, 3]})
    state = ExecutionState(action_index=5, body_version_registry=BodyVersionRegistry({body.key: body}))
    encoder = BodyCandidateEncoder(use_relative_age=True)
    bank = CandidateView(state, encoders={PointerType.BODY: encoder}).bank(PointerType.BODY)
    assert bank.entries[0].legal_metadata["relative_age"] == 3
    assert bank.entries[0].legal_metadata["creator_operation_type"] == "ExtrudeFeature"


def test_same_action_declaration_is_not_committed():
    state = ExecutionState()
    record = ConstructionRecord(ExternalKey("profile_t"), PointerType.PROFILE, 0, {"loop": []}, committed=False)
    try:
        state.construction_registry.add(record)
    except ValueError:
        pass
    else:
        raise AssertionError("uncommitted declaration entered prefix registry")


def test_storage_permutation_preserves_targets():
    bank = CandidateBank.build(PointerType.BODY, [CandidateEntry("a", {"x": 1}), CandidateEntry("b", {"x": 2})])
    permuted = bank.permute_storage([1, 0])
    assert permuted.remap_targets(["a", "b"]) == (1, 0)


def test_action_ast_round_trip_and_pointer_type():
    action = ActionAST(Operation.BOOLEAN, {"operation": "CutFeatureOperation", "target_bodies": [{"pointer_type": "PTR_BODY", "external_key": "a"}], "tool_bodies": [], "keep_tools": False, "cleanup": True})
    assert ActionAST.from_dict(action.to_dict()).to_dict() == action.to_dict()


def test_feedback_only_changes_future_positions():
    feedback = PointerFeedback(8)
    inputs = torch.zeros(4, 8)
    updated = feedback.apply_to_future(inputs, 1, PointerType.FACE, torch.ones(128))
    assert torch.equal(updated[0], inputs[0]) and torch.equal(updated[1], inputs[1])
    assert not torch.equal(updated[2], inputs[2])


def test_multi_positive_and_zero_pointer_loss_are_finite():
    logits = [torch.zeros(3, requires_grad=True)]
    positives = [torch.tensor([True, False, True])]
    loss, slots = pointer_bce_per_slot(logits, positives)
    assert slots == 1 and torch.isfinite(loss)
    zero, slots = pointer_bce_per_slot([], [])
    assert slots == 0 and torch.isfinite(zero)


def test_expanded_forward_backward_smoke():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=32)
    hidden = torch.randn(3, 16)
    output = model(hidden, ExecutionState(), ())
    loss, metrics = model.loss(grammar_logits=output.grammar_logits, grammar_targets=torch.tensor([1, 2, 3]))
    assert torch.isfinite(loss) and all(torch.isfinite(value) for value in metrics.values() if torch.is_tensor(value))
    loss.backward()


def test_native_face_edge_embeddings_and_historical_share_bank_space():
    class Snapshot:
        def candidates(self, kind, include_historical=False):
            return (
                SimpleNamespace(key="face_active", owner="body_a", geometry=None, metadata={"historical": False}),
                SimpleNamespace(key="face_old", owner="body_a", geometry=None, metadata={"historical": True}),
            ) if str(kind).endswith("FACE") else (
                SimpleNamespace(key="edge_active", owner="body_a", geometry=None, metadata={"historical": False}),
                SimpleNamespace(key="edge_old", owner="body_a", geometry=None, metadata={"historical": True}),
            )
    state = ExecutionState(active_brep_state=Snapshot(), native_face_embeddings={"face_active": torch.ones(128), "face_old": torch.ones(128) * 2}, native_edge_embeddings={"edge_active": torch.ones(128), "edge_old": torch.ones(128) * 2})
    from models.crs_pointercad.encoders import FaceCandidateEncoder, EdgeCandidateEncoder
    view = CandidateView(state, encoders={PointerType.FACE: FaceCandidateEncoder(), PointerType.EDGE: EdgeCandidateEncoder()})
    face, edge = view.bank(PointerType.FACE), view.bank(PointerType.EDGE)
    assert face.entries[0].embedding.shape == edge.entries[0].embedding.shape == (128,)
    assert torch.equal(face.entries[1].embedding, torch.ones(128) * 2)


def test_historical_owner_locator_provider_uses_same_128d_space():
    class Snapshot:
        def candidates(self, kind, include_historical=False):
            return (SimpleNamespace(key="old_face", owner="body_a", geometry=None, metadata={"historical": True, "locator": {"index": 2}}),)
    state = ExecutionState(active_brep_state=Snapshot(), historical_face_provider=lambda candidate: torch.full((128,), float(candidate.metadata["locator"]["index"])))
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False)
    maps = model._native_maps((state,))[0]
    assert maps[0]["old_face"].shape == (128,) and maps[0]["old_face"][0] == 2


def test_native_brep_tokens_feed_decoder_context_path():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False)
    class StubBRep:
        def __call__(self, graph):
            del graph
            return (torch.ones(1, 128),), (torch.ones(1, 128) * 2,), (), ()
    model.brep = StubBRep()
    embeddings = torch.zeros(1, 2, 16)
    tokens = torch.tensor([[151667, 151670]])
    injected = model._inject_native_brep_tokens(embeddings, tokens, object())
    assert not torch.equal(injected[:, 0], embeddings[:, 0]) and not torch.equal(injected[:, 1], embeddings[:, 1])


def test_body_geometry_and_age_ablation_are_real():
    geometry = {"faces": [{"center": [0.0, 0.0, 0.0], "area": 1.0}, {"center": [1.0, 0.0, 0.0], "area": 2.0}]}
    with_age = BodyCandidateEncoder(use_relative_age=True)
    without_age = BodyCandidateEncoder(use_relative_age=False)
    semantic = {"geometry": geometry}
    first = with_age(semantic, metadata={"relative_age": 1})
    second = with_age(semantic, metadata={"relative_age": 5})
    off_first = without_age(semantic, metadata={"relative_age": 1})
    off_second = without_age(semantic, metadata={"relative_age": 5})
    assert not torch.allclose(first, second)
    assert torch.allclose(off_first, off_second)


def test_execution_state_uses_real_creation_lineage():
    node = SimpleNamespace(body_id="body_a", producer_feature_id="feature_2", producer_operation="ExtrudeFeature", predecessors=("body_prev",), creation_action_index=2)
    snapshot = SimpleNamespace(step_index=4, active_bodies=("body_a",), body_graph=SimpleNamespace(nodes=(node,)), _bodies={"body_a": object()}, profiles={}, sketch_references={}, references={}, brep=None)
    state = ExecutionState.from_runtime_snapshot(snapshot)
    record = state.body_version_registry.get("body_a")
    assert record.creation_action_index == 2 and record.creator_operation == "ExtrudeFeature"
    assert record.age(state.action_index) == 2 and len(record.predecessor_keys) == 1


def test_profile_preserves_loop_and_nurbs_structure():
    encoder = ProfileCandidateEncoder()
    value = {"loops": [{"is_outer": True, "curves": [{"type": "NurbsCurve3D", "control_points": [[0, 0, 0], [1, 1, 0]], "knots": [0, 1], "weights": [1, 1]}]}, {"is_outer": False, "curves": [{"type": "Circle3D", "center": [0, 0, 0], "radius": 0.2}]}], "transform": {"origin": [0, 0, 0]}}
    output = encoder(value)
    assert output.shape == (128,) and torch.isfinite(output).all()


def test_registry_context_mode_is_live():
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    from models.crs_pointercad.encoders import BodyCandidateEncoder, RegistryContextEncoder
    bank = CandidateView(state, encoders={PointerType.BODY: BodyCandidateEncoder()}).bank(PointerType.BODY)
    none = RegistryContextEncoder(8, "NONE")({PointerType.BODY: bank})
    pooled = RegistryContextEncoder(8, "TYPE_POOLED")({PointerType.BODY: bank})
    assert none.numel() == 0 and pooled.shape == (1, 8)


def test_qwen_base_model_path_and_real_forward():
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Model
    from models.crs_pointercad.model import CRSExpandedPointerCAD
    base = Qwen2Model(Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4))
    model = CRSExpandedPointerCAD(base_model=base, hidden_dim=32, grammar_vocab_size=16, use_native_brep=False)
    ids = torch.randint(0, 128, (1, 5))
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), state=ExecutionState())
    assert output.hidden_states.shape == (1, 5, 32)


def test_context_summary_reaches_decoder_input():
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    none = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, registry_context_mode="NONE", use_native_brep=False).eval()
    pooled = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, registry_context_mode="TYPE_POOLED", use_native_brep=False).eval()
    pooled.load_state_dict(none.state_dict(), strict=False)
    ids = torch.randint(0, 128, (1, 5))
    mask = torch.ones_like(ids)
    out_none = none(input_ids=ids, attention_mask=mask, state=state)
    out_pooled = pooled(input_ids=ids, attention_mask=mask, state=state)
    assert out_none.context_features.numel() == 0
    assert out_pooled.context_features.numel() > 0
    assert not torch.allclose(out_none.hidden_states, out_pooled.hidden_states)


def test_teacher_feedback_uses_recorded_candidate_and_changes_only_future():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False)
    model.eval()
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    ids = torch.randint(0, 128, (1, 5))
    result = model.teacher_forced_decode(ids, torch.ones_like(ids), state, [1], [PointerType.BODY], [0])
    assert torch.equal(result["hidden_before"][:, 1, :], result["hidden_after"][:, 1, :])
    assert not torch.equal(result["hidden_before"][:, 2, :], result["hidden_after"][:, 2, :])


def test_inference_pointer_selection_and_predicted_feedback_path():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False).eval()
    body = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]})
    state = ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body.key: body}))
    ids = torch.randint(0, 128, (1, 5))
    selected = model.inference_pointer_decode(ids, torch.ones_like(ids), state, [1], [PointerType.BODY])
    assert selected[0][0] is PointerType.BODY and selected[0][2] == body.key


def test_expanded_config_flags_are_live_on_model():
    config = yaml.safe_load(open("config/crs_expanded_smoke.yaml"))
    from models.model_factory import from_config
    model = from_config(config)
    assert model.mode is ModelMode.CRS_EXPANDED_POINTERCAD
    assert model.registry_context.mode == config["model"]["registry_context_mode"]
    assert model.enable_historical_face == config["model"]["enable_historical_face"]
    assert model.enable_historical_edge == config["model"]["enable_historical_edge"]
    assert model.feedback is not None and model.query_heads.typed_scales == config["model"]["typed_pointer_scales"]


def test_causal_materialization_maps_runtime_ref_kinds_without_sidecars():
    from models.crs_pointercad.materialization import materialize_causal_supervision
    target = SimpleNamespace(kind=SimpleNamespace(value="FACE"), snapshot_step=0, target_candidate="face_a", action_index=0, slot="face")
    supervision = SimpleNamespace(pointer_targets=(target,))
    class Snapshot:
        def candidates(self, kind, include_historical=False):
            return (SimpleNamespace(key="face_a", owner="body_a", geometry=torch.ones(128), metadata={"native_embedding": torch.ones(128)}),)
    trace = SimpleNamespace(at=lambda step: Snapshot())
    # Runtime adapter construction is tested separately; this assertion proves
    # the model-side mapper accepts CRS RefKind spelling without a sidecar bank.
    result = materialize_causal_supervision(supervision, trace)
    assert result["pointer_positions"] == 1 and result["unresolved_targets"] == 0


def test_field_complete_grammar_branches_and_dynamic_masks():
    from models.crs_pointercad.grammar import PointerLeaf
    ActionAST(Operation.SHELL, {"shell_mode": "BODY", "input_body": PointerLeaf("PTR_BODY", "b"), "shell_type": "Inside", "inside_thickness": 0.1}).validate()
    try:
        ActionAST(Operation.SHELL, {"shell_mode": "BODY", "input_body": PointerLeaf("PTR_BODY", "b"), "selected_faces": [], "shell_type": "Inside"}).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("Shell BODY/FACE branches must be exclusive")
    try:
        ActionAST(Operation.CHAMFER, {"chamfer_type": "TwoDistances", "edges": [], "edge_sides": [], "edge_propagation": "TangentChain"}).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("TwoDistances Chamfer must require EDGE/FACE records")


def test_dynamic_face_mask_uses_selected_edge_incidence():
    entries = [CandidateEntry("f1", torch.ones(128), legal_metadata={"owner": "b", "incident_edge_keys": ("e1",)}), CandidateEntry("f2", torch.ones(128), legal_metadata={"owner": "b", "incident_edge_keys": ("e2",)})]
    bank = CandidateBank.build(PointerType.FACE, entries)
    class Snapshot:
        def candidates(self, kind, include_historical=False):
            return ()
    state = ExecutionState(active_brep_state=Snapshot())
    # Exercise the same mask contract directly on a bank-shaped view.
    from models.crs_pointercad.candidates import CandidateView
    view = CandidateView(state)
    filtered = view._apply_masks(PointerType.FACE, tuple(entries), {"selected_edge_key": "e1"})
    assert [entry.external_key for entry in filtered] == ["f1"]


def test_dynamic_body_role_mask_keeps_boolean_targets_and_tools_distinct():
    entries = [CandidateEntry("target", torch.ones(128)), CandidateEntry("tool", torch.ones(128))]
    view = CandidateView(ExecutionState())
    filtered = view._apply_masks(PointerType.BODY, tuple(entries), {"role": "tool", "target_keys": ("target",)})
    assert [entry.external_key for entry in filtered] == ["tool"]


def _ragged_body_states():
    body_a = BodyRecord(ExternalKey("body_a"), 0, "Sketch", geometry={"faces": [{"center": [0, 0, 0], "area": 1.0}]})
    body_b = BodyRecord(ExternalKey("body_b"), 0, "Sketch", geometry={"faces": [{"center": [1, 0, 0], "area": 2.0}]})
    return [
        ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body_a.key: body_a})),
        ExecutionState(action_index=1, body_version_registry=BodyVersionRegistry({body_a.key: body_a, body_b.key: body_b})),
    ]


def test_ragged_batch_different_slot_and_bank_counts_works():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False).eval()
    states = _ragged_body_states()
    ids = torch.randint(0, 128, (2, 6))
    specs = [[], [(1, PointerType.BODY), (3, PointerType.BODY)]]
    teacher = [[], [0, 1]]
    output = model.forward_ragged(input_ids=ids, attention_mask=torch.ones_like(ids), states=states, pointer_specs=specs, teacher_indices=teacher)
    assert len(output.pointer_logits_by_example) == 2
    assert len(output.pointer_logits_by_example[0]) == 0 and len(output.pointer_logits_by_example[1]) == 2
    assert [len(bank) for bank in output.candidate_banks_by_example[1]] == [2, 2]


def test_ragged_pointer_loss_flattens_slots_and_keeps_zero_pointer_example():
    logits = [[], [torch.zeros(2, requires_grad=True), torch.zeros(1, requires_grad=True)]]
    positives = [[], [torch.tensor([True, False]), torch.tensor([True])]]
    loss, slots = pointer_bce_per_slot(logits, positives)
    assert slots == 2 and torch.isfinite(loss)
    loss.backward()


def test_real_training_feedback_changes_later_hidden_not_pointer_hidden():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False).eval()
    states = _ragged_body_states()
    ids = torch.randint(0, 128, (1, 6))
    first = model.forward_ragged(input_ids=ids, attention_mask=torch.ones_like(ids), states=[states[1]], pointer_specs=[[(1, PointerType.BODY), (3, PointerType.BODY)]], teacher_indices=[[0, 1]])
    second = model.forward_ragged(input_ids=ids, attention_mask=torch.ones_like(ids), states=[states[1]], pointer_specs=[[(1, PointerType.BODY), (3, PointerType.BODY)]], teacher_indices=[[1, 0]])
    assert torch.allclose(first.pointer_hidden_by_example[0][0], second.pointer_hidden_by_example[0][0])
    assert not torch.allclose(first.hidden_states[:, 4, :], second.hidden_states[:, 4, :])


def test_decoder_substate_is_slot_local_in_teacher_and_inference_paths():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False).eval()
    states = _ragged_body_states()
    ids = torch.randint(0, 128, (1, 6))
    output = model.forward_ragged(input_ids=ids, attention_mask=torch.ones_like(ids), states=[states[1]], pointer_specs=[[(1, PointerType.BODY), (3, PointerType.BODY)]], decoder_substates=[[{"excluded_keys": ("body_a",)}, {"excluded_keys": ("body_b",)}]], teacher_indices=[[0, 0]])
    assert len(output.candidate_banks_by_example[0][0]) == 1
    assert len(output.candidate_banks_by_example[0][1]) == 1


def test_ordered_curve_and_profile_changes_representation():
    from models.crs_pointercad.encoders import Curve3DEncoder
    curve = Curve3DEncoder()
    a = {"type": "Polyline3D", "points": [[0, 0, 0], [1, 0, 0], [2, 0, 0]]}
    b = {"type": "Polyline3D", "points": [[2, 0, 0], [1, 0, 0], [0, 0, 0]]}
    assert not torch.allclose(curve(a), curve(b))
    profile = ProfileCandidateEncoder()
    p1 = {"loops": [{"is_outer": True, "curves": [a, {"type": "Circle3D", "center": [0, 0, 0], "radius": 1}]}]}
    p2 = {"loops": [{"is_outer": True, "curves": [p1["loops"][0]["curves"][1], a]}]}
    assert not torch.allclose(profile(p1), profile(p2))


def test_numeric_heads_and_four_component_loss_are_trainable():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False)
    output = model(input_ids=torch.randint(0, 128, (1, 5)), attention_mask=torch.ones(1, 5, dtype=torch.long), state=ExecutionState())
    scalar_target = torch.zeros_like(output.scalar_predictions)
    record_targets = {"VECTOR3": torch.zeros_like(output.record_predictions["VECTOR3"])}
    loss, parts = model.loss(grammar_logits=output.grammar_logits, grammar_targets=torch.zeros((1, 5), dtype=torch.long),
                             scalar_predictions=output.scalar_predictions, scalar_targets=scalar_target,
                             record_predictions=output.record_predictions, record_targets=record_targets)
    assert all(torch.isfinite(parts[name]) for name in ("grammar", "pointer", "scalar", "record"))
    loss.backward()
    assert model.numeric_heads.scalar.weight.grad is not None
    assert model.numeric_heads.records["VECTOR3"].weight.grad is not None


def test_canonical_training_loss_consumes_ragged_forward_output():
    model = CRSExpandedPointerCAD(hidden_dim=16, grammar_vocab_size=16, use_native_brep=False)
    states = _ragged_body_states()
    ids = torch.randint(0, 128, (2, 5))
    output = model.forward_ragged(input_ids=ids, attention_mask=torch.ones_like(ids), states=states, pointer_specs=[[], [(1, PointerType.BODY)]], teacher_indices=[[], [0]])
    loss, metrics = model.training_loss(output, grammar_targets=torch.zeros((2, 5), dtype=torch.long), pointer_positive=[[], [torch.tensor([True, False])]], scalar_targets=torch.zeros_like(output.scalar_predictions), record_targets={"VECTOR3": torch.zeros_like(output.record_predictions["VECTOR3"])})
    assert torch.isfinite(loss) and all(torch.isfinite(metrics[name]) for name in ("grammar", "pointer", "scalar", "record"))
    loss.backward()


def test_grammar_routes_pointer_scalar_and_record_fields():
    action = ActionAST(Operation.EXTRUDE, {
        "profiles": [], "operation": "NewBodyFeatureOperation", "participant_bodies": [],
        "start_extent": {"type": "ProfilePlaneStartDefinition"},
        "extent_type": "OneSideFeatureExtentType", "extent_one": {"type": "DistanceExtentDefinition"},
    })
    assert action.field_route("profiles").value == "MODEL_POINTER"
    assert action.field_route("twist_angle").value == "MODEL_SCALAR"
    assert action.field_route("participant_bodies").value == "MODEL_POINTER"
    assert action.field_route("start_extent").value == "MODEL_LABEL"
