import json

from models.crs_pointercad.frozen_corpus import FrozenStage2Corpus


def test_frozen_stage2_reader_reloads_membership_without_source_recording(tmp_path):
    root = tmp_path / "corpus"
    (root / "NATURAL_CLEAN/dataset/sample/v000").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({"format": "stage2-clean-validation-v1", "entries": []}))
    assert len(FrozenStage2Corpus(root)) == 0
