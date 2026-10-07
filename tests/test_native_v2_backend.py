import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from models.crs_pointercad.training import PreparedStage2Corpus
from models.crs_pointercad.brep_bridge import load_prepared_v2_state


class NativeV2BackendTests(unittest.TestCase):
    def test_backend_is_explicit_and_frozen_hash_is_checked(self):
        with TemporaryDirectory() as directory:
            frozen, prepared = Path(directory) / "frozen", Path(directory) / "prepared"
            frozen.mkdir()
            prepared.mkdir()
            source = b'{"format":"stage2-clean-validation-v1","entries":[]}'
            (frozen / "manifest.json").write_bytes(source)
            manifest = {"format": "stage2a3-native-input-v2",
                        "frozen_manifest_sha256": hashlib.sha256(source).hexdigest(), "entries": []}
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                PreparedStage2Corpus(frozen, prepared)
            self.assertEqual(PreparedStage2Corpus(frozen, prepared, native_backend="v2").entries, {})
            manifest["frozen_manifest_sha256"] = "0" * 64
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "different frozen manifest"):
                PreparedStage2Corpus(frozen, prepared, native_backend="v2")

    def test_payload_reference_rejects_path_escape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text(json.dumps({"active_bodies": ["body_1"], "historical_bodies": []}))
            (root / "view.json").write_text(json.dumps({"format": "stage2a3-native-view-v2",
                                                         "owners": [{"owner": "body_1", "block": "../secret"}]}))
            with self.assertRaisesRegex(ValueError, "block reference"):
                load_prepared_v2_state(root, root)

    def test_unreleased_migration_manifest_requires_explicit_opt_in(self):
        with TemporaryDirectory() as directory:
            frozen, prepared, local = (Path(directory) / name for name in ("frozen", "prepared", "local"))
            frozen.mkdir(); prepared.mkdir(); local.mkdir()
            source = b'{"format":"stage2-clean-validation-v1","entries":[]}'
            (frozen / "manifest.json").write_bytes(source)
            manifest = {"format": "stage2a3-native-migration-v1", "migration_mode": True,
                        "source_backend": "v1_unreleased_individually_validated", "status": "COMPLETE",
                        "frozen_manifest_sha256": hashlib.sha256(source).hexdigest(),
                        "expected_identities": [], "failures": [], "entries": []}
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                PreparedStage2Corpus(frozen, prepared)
            migration = local / "v1_oracle_manifest.json"
            migration.write_text(json.dumps(manifest))
            corpus = PreparedStage2Corpus(frozen, prepared, migration_manifest=migration)
            self.assertTrue(corpus.migration_mode)
            self.assertEqual(corpus.entries, {})
            manifest["source_backend"] = "published_v1"
            migration.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "migration provenance"):
                PreparedStage2Corpus(frozen, prepared, migration_manifest=migration)

    def test_normal_manifest_rejects_migration_flag(self):
        with TemporaryDirectory() as directory:
            frozen, prepared = Path(directory) / "frozen", Path(directory) / "prepared"
            frozen.mkdir(); prepared.mkdir()
            source = b'{"format":"stage2-clean-validation-v1","entries":[]}'
            (frozen / "manifest.json").write_bytes(source)
            (prepared / "manifest.json").write_text(json.dumps({
                "format": "stage2a3-native-input-v1", "migration_mode": True,
                "frozen_manifest_sha256": hashlib.sha256(source).hexdigest(), "entries": []}))
            with self.assertRaisesRegex(ValueError, "unreleased migration"):
                PreparedStage2Corpus(frozen, prepared)


if __name__ == "__main__":
    unittest.main()
