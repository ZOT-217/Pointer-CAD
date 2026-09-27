"""Field-complete v0.4 action grammar and typed action AST."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from .contracts import PointerType


class Operation(str, Enum):
    SKETCH = "Sketch"
    EXTRUDE = "ExtrudeFeature"
    REVOLVE = "RevolveFeature"
    SWEEP = "SweepFeature"
    LOFT = "LoftFeature"
    MIRROR = "MirrorFeature"
    SHELL = "ShellFeature"
    FILLET = "FilletFeature"
    CHAMFER = "ChamferFeature"
    BOOLEAN = "BooleanFeature"
    TRANSFORM = "TransformFeature"
    SPLIT = "SplitFeature"


class FieldRole(str, Enum):
    LABEL = "MODEL_LABEL"
    POINTER = "MODEL_POINTER"
    DECLARATION = "MODEL_DECLARATION"
    SCALAR = "MODEL_SCALAR"
    RECORD = "MODEL_STRUCTURED_NUMERIC"


class RecordType(str, Enum):
    POINT3 = "POINT3"
    VECTOR3 = "VECTOR3"
    DIRECTION3 = "DIRECTION3"
    AXIS3 = "AXIS3"
    PLANE3 = "PLANE3"
    FRAME3 = "FRAME3"


@dataclass(frozen=True)
class StructuredRecord:
    record_type: RecordType | str
    fields: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, "record_type", self.record_type if isinstance(self.record_type, RecordType) else RecordType(self.record_type))
        object.__setattr__(self, "fields", dict(self.fields))
        if not self.fields:
            raise ValueError("structured numeric record cannot be empty")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    role: FieldRole
    pointer_type: PointerType | None = None
    required: bool = True
    repeated: bool = False
    ordered: bool = False


@dataclass(frozen=True)
class OperationSpec:
    operation: Operation
    fields: tuple[FieldSpec, ...]
    branches: tuple[str, ...] = ()

    @property
    def by_name(self):
        return {field.name: field for field in self.fields}


def _f(name, role=FieldRole.LABEL, pointer_type=None, required=True, repeated=False, ordered=False):
    return FieldSpec(name, role, pointer_type, required, repeated, ordered)


GRAMMAR: dict[Operation, OperationSpec] = {
    Operation.SKETCH: OperationSpec(Operation.SKETCH, (_f("reference_plane"), _f("transform", FieldRole.RECORD), _f("profiles", FieldRole.DECLARATION, repeated=True, ordered=True), _f("references", FieldRole.DECLARATION, required=False, repeated=True, ordered=True))),
    Operation.EXTRUDE: OperationSpec(Operation.EXTRUDE, (_f("profiles", FieldRole.POINTER, PointerType.PROFILE, repeated=True, ordered=True), _f("operation"), _f("participant_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("start_extent"), _f("extent_type"), _f("extent_one"), _f("extent_two", required=False), _f("twist_angle", FieldRole.SCALAR, required=False), _f("cleanup", required=False))),
    Operation.REVOLVE: OperationSpec(Operation.REVOLVE, (_f("profiles", FieldRole.POINTER, PointerType.PROFILE, repeated=True, ordered=True), _f("axis"), _f("operation"), _f("participant_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("extent_type"), _f("extent_one"), _f("extent_two", required=False), _f("cleanup", required=False))),
    Operation.SWEEP: OperationSpec(Operation.SWEEP, (_f("profiles", FieldRole.POINTER, PointerType.PROFILE, repeated=True, ordered=True), _f("path", FieldRole.POINTER, PointerType.EDGE, repeated=True, ordered=True), _f("orientation"), _f("operation"), _f("participant_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("cleanup", required=False))),
    Operation.LOFT: OperationSpec(Operation.LOFT, (_f("sections", FieldRole.POINTER, PointerType.PROFILE, repeated=True, ordered=True), _f("interpolation"), _f("operation"), _f("participant_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("cleanup", required=False))),
    Operation.MIRROR: OperationSpec(Operation.MIRROR, (_f("input_bodies", FieldRole.POINTER, PointerType.BODY, repeated=True, ordered=True), _f("mirror_plane", FieldRole.RECORD), _f("is_combine"))),
    Operation.SHELL: OperationSpec(Operation.SHELL, (_f("shell_mode"), _f("input_body", FieldRole.POINTER, PointerType.BODY, required=False), _f("selected_faces", FieldRole.POINTER, PointerType.FACE, required=False, repeated=True, ordered=True), _f("inside_thickness", FieldRole.SCALAR, required=False), _f("outside_thickness", FieldRole.SCALAR, required=False), _f("shell_type"))),
    Operation.FILLET: OperationSpec(Operation.FILLET, (_f("edges", FieldRole.POINTER, PointerType.EDGE, repeated=True, ordered=True), _f("radius", FieldRole.SCALAR), _f("edge_propagation"))),
    Operation.CHAMFER: OperationSpec(Operation.CHAMFER, (_f("chamfer_type"), _f("edges", FieldRole.POINTER, PointerType.EDGE, repeated=True, ordered=True), _f("edge_sides", FieldRole.RECORD, required=False, repeated=True, ordered=True), _f("distance", FieldRole.SCALAR, required=False), _f("distance_one", FieldRole.SCALAR, required=False), _f("distance_two", FieldRole.SCALAR, required=False), _f("edge_propagation"))),
    Operation.BOOLEAN: OperationSpec(Operation.BOOLEAN, (_f("operation"), _f("target_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("tool_bodies", FieldRole.POINTER, PointerType.BODY, required=False, repeated=True, ordered=True), _f("keep_tools"), _f("cleanup"), _f("tolerance", FieldRole.SCALAR, required=False), _f("glue", required=False))),
    Operation.TRANSFORM: OperationSpec(Operation.TRANSFORM, (_f("input_bodies", FieldRole.POINTER, PointerType.BODY, repeated=True, ordered=True), _f("transform", FieldRole.RECORD))),
    Operation.SPLIT: OperationSpec(Operation.SPLIT, (_f("input_body", FieldRole.POINTER, PointerType.BODY), _f("split_plane", FieldRole.RECORD), _f("keep_top"), _f("keep_bottom"))),
}


@dataclass(frozen=True)
class PointerLeaf:
    pointer_type: str
    external_key: Any

    def __post_init__(self):
        if self.pointer_type not in {item.value for item in PointerType}:
            raise ValueError(f"unknown pointer type: {self.pointer_type}")


@dataclass(frozen=True)
class ActionAST:
    operation: Operation | str
    fields: Mapping[str, Any] = field(default_factory=dict)
    declarations: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "operation", self.operation if isinstance(self.operation, Operation) else Operation(self.operation))
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(self, "declarations", tuple(dict(item) for item in self.declarations))

    @property
    def spec(self) -> OperationSpec:
        return GRAMMAR[self.operation]

    def field_route(self, field_name: str) -> FieldRole:
        """Return the unique v0.4 output head route for a semantic field."""
        field = self.spec.by_name.get(field_name)
        if field is None:
            raise KeyError(f"{self.operation.value}.{field_name} is not in the frozen grammar")
        return field.role

    def validate(self) -> None:
        fields = self.fields
        spec = self.spec
        missing = [item.name for item in spec.fields if item.required and item.name not in fields]
        # Fields represented by a structured branch may be supplied under its
        # branch record; only operation-level presence is required here.
        if missing:
            raise ValueError(f"{self.operation.value} missing required fields: {', '.join(missing)}")
        for item in spec.fields:
            if item.name not in fields:
                continue
            value = fields[item.name]
            values = value if item.repeated else (value,)
            if item.repeated and not isinstance(value, (list, tuple)):
                raise ValueError(f"{self.operation.value}.{item.name} must be an ordered list")
            for candidate in values:
                self._validate_field(item, candidate)
        if self.operation is Operation.SHELL:
            mode = fields.get("shell_mode")
            if mode not in {"BODY", "FACE"}:
                raise ValueError("Shell shell_mode must be BODY or FACE")
            if mode == "BODY" and ("input_body" not in fields or "selected_faces" in fields):
                raise ValueError("Shell BODY mode is exclusive with selected faces")
            if mode == "FACE" and ("selected_faces" not in fields or "input_body" in fields):
                raise ValueError("Shell FACE mode is exclusive with input body")
            if mode == "FACE" and not fields.get("selected_faces"):
                raise ValueError("Shell FACE mode requires at least one face")
        if self.operation is Operation.CHAMFER and fields.get("chamfer_type") == "TwoDistances":
            sides = fields.get("edge_sides", ())
            if not sides or any(not isinstance(side, Mapping) or "edge" not in side or "distance_one_face" not in side for side in sides):
                raise ValueError("TwoDistances Chamfer requires ordered EDGE -> incident FACE records")
            if len(sides) != len(fields.get("edges", ())):
                raise ValueError("TwoDistances Chamfer EDGE/FACE records must be cardinality aligned")
            if "distance_one" not in fields or "distance_two" not in fields:
                raise ValueError("TwoDistances Chamfer requires distance_one and distance_two")
        if self.operation is Operation.CHAMFER and fields.get("chamfer_type") == "EqualDistance":
            if not fields.get("edges") or "distance" not in fields:
                raise ValueError("EqualDistance Chamfer requires ordered edges and distance")
        minimum_lists = {
            Operation.EXTRUDE: ("profiles",), Operation.REVOLVE: ("profiles",),
            Operation.SWEEP: ("profiles", "path"), Operation.LOFT: ("sections",),
            Operation.MIRROR: ("input_bodies",), Operation.FILLET: ("edges",),
            Operation.BOOLEAN: (), Operation.TRANSFORM: ("input_bodies",),
        }
        for field_name in minimum_lists.get(self.operation, ()):
            if not fields.get(field_name):
                raise ValueError(f"{self.operation.value}.{field_name} requires a nonempty ordered list")
        if self.operation is Operation.LOFT and len(fields.get("sections", ())) < 2:
            raise ValueError("Loft requires at least two ordered sections")
        if self.operation is Operation.BOOLEAN and not fields.get("target_bodies") and not fields.get("tool_bodies"):
            raise ValueError("Boolean requires at least one target or tool body")

    @staticmethod
    def _validate_field(spec: FieldSpec, value: Any) -> None:
        if spec.pointer_type is not None:
            if isinstance(value, PointerLeaf):
                kind = value.pointer_type
            elif isinstance(value, Mapping):
                kind = value.get("pointer_type")
            else:
                raise ValueError(f"{spec.name} requires typed pointer leaves")
            if kind != spec.pointer_type.value:
                raise ValueError(f"{spec.name} requires {spec.pointer_type.value}, got {kind!r}")
        elif spec.role is FieldRole.RECORD and not isinstance(value, (Mapping, StructuredRecord)):
            raise ValueError(f"{spec.name} requires a structured record")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionAST":
        result = cls(value["operation"], value.get("fields", {}), tuple(value.get("declarations", ())))
        result.validate()
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"operation": self.operation.value, "fields": _plain(self.fields), "declarations": _plain(self.declarations)}

    def to_crs_semantics(self) -> dict[str, Any]:
        self.validate()
        result = {"type": self.operation.value}
        result.update(_plain(self.fields))
        return result


def _plain(value: Any) -> Any:
    if isinstance(value, PointerLeaf):
        return {"pointer_type": value.pointer_type, "external_key": value.external_key}
    if isinstance(value, StructuredRecord):
        return {"record_type": value.record_type.value, "fields": _plain(value.fields)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value
