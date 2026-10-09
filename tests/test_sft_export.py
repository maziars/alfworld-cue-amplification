from pathlib import Path
import tempfile
import json
import unittest

from research.export_sft_checkpoint import check_copied_keys, validate_catalogs, validate_index


class SFTExportTests(unittest.TestCase):
    def test_language_fallback_forbidden(self):
        with self.assertRaises(ValueError):
            check_copied_keys("add model.language_model.layers.0.weight from origin hf checkpoint\n", True)
        with self.assertRaises(ValueError):
            check_copied_keys("add lm_head.weight from origin hf checkpoint\n", True)

    def test_mtp_requires_explicit_option(self):
        log = "add model.visual.foo from origin hf checkpoint\nadd mtp.foo from origin hf checkpoint\n"
        with self.assertRaises(ValueError):
            check_copied_keys(log)
        self.assertEqual(check_copied_keys(log, True), ["model.visual.foo", "mtp.foo"])

    def test_dtype_drift_rejected(self):
        origin = {"weight": {"shape": [2], "bytes": 4, "dtype": "BF16"}}
        exported = {"weight": {"shape": [2], "bytes": 8, "dtype": "F32"}}
        with self.assertRaises(ValueError):
            validate_catalogs(origin, exported, {})

    def test_index_total_corrected_but_mapping_not_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "model.safetensors.index.json"
            index.write_text(json.dumps({"metadata": {"total_size": 999}, "weight_map": {"foo": "x"}}))
            report = validate_index(root, {"foo": {"file": "x", "bytes": 8}})
            self.assertEqual(report["actual_total_size"], 8)
            self.assertTrue(report["corrected"])
            with self.assertRaises(ValueError):
                validate_index(root, {"bar": {"file": "x", "bytes": 8}})


if __name__ == "__main__":
    unittest.main()
