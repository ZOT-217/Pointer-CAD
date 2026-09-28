# FIRST_1K Selection/OCC Determinism Forensic

## Scope and identity

This local diagnostic ran at Pointer-CAD HEAD `645209f605477d7aec5ad7ee43023b604cc52df3`. The certified CRS population was `../cadquery2crs/outputs/current-1k-certification/crs`, generated from cadquery2crs commit `203bed7c38cb39b97fb50474edfdf53b867ee984`.

The CRS inventory covers the certified 1K population; 931 CRS artifacts are present after failed rows are excluded. The local CQ source population is only 215 samples under `../cadquery2crs/.local-data/zero2cad-test-mini`, not the complete certified 1K source population. The source hash for sample `00000` matches the certified manifest exactly.

## Inventory

The CRS inventory contains 997 multi-selection actions:

- Chamfer: 674 actions; 43 select more than 32 edges.
- Fillet: 323 actions; 9 select more than 32 edges.

No other multi-edge operation was included. Complete counts are in `docs/audits/2026-09-27-first1k-selection-occ-determinism/cardinality_distribution.json`.

## Fresh-process evidence

The policy was 10 fresh processes for sample `00000` and 5 fresh processes for two additional source-backed cases.

### sample-00000

The locally available source selects 8 edges, not the 36 edges reported by CCI. Across 10 fresh processes, raw order, physical signatures, locator mapping, selection IDs, serialized order, and reconstruction were stable. This does not disprove the CCI 36-edge result; it establishes that the local source/runtime does not reproduce that exact CCI case.

### sample-00031, Fillet

Five fresh processes selected the same two physical edges and retained the same selection IDs and physical signatures. Raw order and serialized order had two variants. Each run was internally self-consistent, but the same selection ID could refer to the other physical edge across processes. Reconstruction failed at an earlier invalid-BREP feature.

### sample-00268, Fillet plus Chamfer

The 4-edge Fillet was stable across five runs. The 28-edge Chamfer preserved the same physical signature set and selection IDs but had three raw/serialized order variants. Reconstruction failed at an invalid-BREP boundary.

## Answers

1. The CCI 36-edge sample cannot be classified as isolated or systematic from this local population because its source/runtime reproduction is missing. Local evidence shows the phenomenon in a 2-edge Fillet and a 28-edge Chamfer, so it is not limited to cardinality 36.
2. Yes. Raw OCC/CadQuery enumeration order varied in the local Fillet and Chamfer cases.
3. No physical-set variation was observed; the signature set stayed constant.
4. Yes. Stable selection IDs followed the varying order and therefore bound to different physical entities across processes.
5. Locator/physical signature sets were stable. Ordered binding was not stable.
6. STEP normalization was not run because the complete certified STEP/source population is absent locally.
7. A bounded permutation probe was not run. The observed order drift already requires a separate order-semantics decision.
8. The drift changes serialized multi-positive target order and can change selection-ID-to-physical binding; it is not established as benign storage order.
9. The first observed drift is raw enumeration order, followed by positional selection-ID allocation and serialized target order.
10. The narrowest future direction is an ID-binding investigation using stable physical/locator semantics, followed by a separate decision about operation order. No repair was implemented.

## Root cause and limitation

Supported diagnostic labels are `OCC_ENUMERATION_ORDER`, `SELECTION_ID_ALLOCATION`, and `SERIALIZATION_ORDER`. `CQ_SELECTOR_RESULT_SET` and `LOCATOR_AMBIGUITY` are not supported by local evidence because physical and locator sets remained stable. STEP effects remain unknown.

The missing complete FIRST_1K CQ source/reference population is the main limitation. The exact CCI 36-edge sample must be rerun from its authoritative source/runtime before selecting a production correction.

Final verdict: `SELECTION_NONDETERMINISM_HAS_MULTIPLE_ROOT_CAUSES`
