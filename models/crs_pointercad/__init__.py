"""CRS-native expanded Pointer-CAD model interfaces."""

from .candidates import CandidateBank, CandidateView, ContextView
from .contracts import (
    BodyRecord,
    BodyVersionRegistry,
    CandidateEntry,
    ConstructionRecord,
    ConstructionRegistry,
    ExecutionState,
    ExternalKey,
    ModelMode,
    PointerType,
)
from .encoders import (
    BodyCandidateEncoder,
    Curve3DEncoder,
    EdgeCandidateEncoder,
    FaceCandidateEncoder,
    ProfileCandidateEncoder,
    RegistryContextEncoder,
    ResolvedGeometryCandidateEncoder,
    SketchReferenceCandidateEncoder,
)
from .frozen_corpus import FrozenStage2Corpus, FrozenStage2Record
from .grammar import ActionAST, Operation, PointerLeaf, RecordType, StructuredRecord
from .heads import ExpandedLoss, PointerFeedback, TypedPointerHeads, pointer_bce_per_slot
from .materialization import KNOWN_EXECUTOR_TAIL, MaterializationReport, materialize_causal_supervision, materialize_prefixes
from .model import CRSExpandedPointerCAD, build_pointercad
from .sequence import Stage2QwenCollator, Stage2Sequence, TargetPosition, frozen_grammar_vocabulary
from .brep_bridge import PreparedBRep, PreparedSnapshot, FrozenCandidateKey, load_prepared_brep, load_prepared_state, save_prepared_brep, validate_candidate_alignment
from .training import PreparedStage2Corpus, training_action_loss
from .images import Stage2ImageManifest

__all__ = [
    "ActionAST", "BodyCandidateEncoder", "BodyRecord", "BodyVersionRegistry", "CRSExpandedPointerCAD", "Curve3DEncoder",
    "CandidateBank", "CandidateEntry", "CandidateView", "ContextView", "ConstructionRecord", "ConstructionRegistry",
    "EdgeCandidateEncoder", "ExecutionState", "ExpandedLoss", "ExternalKey", "FaceCandidateEncoder",
    "FrozenStage2Corpus", "FrozenStage2Record",
    "KNOWN_EXECUTOR_TAIL", "MaterializationReport", "ModelMode", "Operation", "PointerFeedback",
    "PointerLeaf", "PointerType", "ProfileCandidateEncoder", "RecordType", "RegistryContextEncoder",
    "ResolvedGeometryCandidateEncoder", "SketchReferenceCandidateEncoder", "StructuredRecord", "TypedPointerHeads",
    "build_pointercad", "materialize_causal_supervision", "materialize_prefixes", "pointer_bce_per_slot",
    "Stage2QwenCollator", "Stage2Sequence", "TargetPosition", "frozen_grammar_vocabulary",
    "PreparedBRep", "load_prepared_brep", "save_prepared_brep", "validate_candidate_alignment",
    "PreparedSnapshot", "FrozenCandidateKey", "load_prepared_state",
    "PreparedStage2Corpus", "training_action_loss",
    "Stage2ImageManifest",
]
