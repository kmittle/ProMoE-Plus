import unittest

from models.models_ProMoE_TC_dino_route import (
    _expected_table_contract,
    _validate_table_metadata,
)
from preprocess.dino_route_table_contract import (
    CORRECTED_TABLE_METHOD,
    CORRECTED_TABLE_VERSION,
    LEGACY_TABLE_METHOD,
    LEGACY_TABLE_VERSION,
)


class DinoRouteTableContractTests(unittest.TestCase):
    TABLE_SHA256 = "a" * 64

    def test_undeclared_contract_is_exactly_legacy(self):
        self.assertEqual(
            _expected_table_contract({}),
            (LEGACY_TABLE_VERSION, LEGACY_TABLE_METHOD),
        )

    def test_corrected_contract_must_be_declared_as_a_pair(self):
        with self.assertRaisesRegex(ValueError, "declared together"):
            _expected_table_contract({"table_version": CORRECTED_TABLE_VERSION})

    def test_corrected_contract_is_supported(self):
        self.assertEqual(
            _expected_table_contract({
                "table_version": CORRECTED_TABLE_VERSION,
                "table_method": CORRECTED_TABLE_METHOD,
            }),
            (CORRECTED_TABLE_VERSION, CORRECTED_TABLE_METHOD),
        )

    def test_metadata_cannot_cross_legacy_and_corrected_contracts(self):
        metadata = {
            "version": CORRECTED_TABLE_VERSION,
            "method": CORRECTED_TABLE_METHOD,
            "num_classes": 1000,
        }
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            _validate_table_metadata(
                metadata,
                expected_num_classes=1000,
                expected_version=LEGACY_TABLE_VERSION,
                expected_method=LEGACY_TABLE_METHOD,
                actual_table_sha256=self.TABLE_SHA256,
            )

    def test_metadata_requires_exact_method(self):
        with self.assertRaisesRegex(ValueError, "method mismatch"):
            _validate_table_metadata(
                {
                    "version": CORRECTED_TABLE_VERSION,
                    "method": "wrong",
                    "num_classes": 1000,
                    "table_sha256": self.TABLE_SHA256,
                },
                expected_num_classes=1000,
                expected_version=CORRECTED_TABLE_VERSION,
                expected_method=CORRECTED_TABLE_METHOD,
                actual_table_sha256=self.TABLE_SHA256,
            )

    def test_corrected_metadata_binds_the_npz_hash(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            _validate_table_metadata(
                {
                    "version": CORRECTED_TABLE_VERSION,
                    "method": CORRECTED_TABLE_METHOD,
                    "num_classes": 1000,
                    "table_sha256": "b" * 64,
                },
                expected_num_classes=1000,
                expected_version=CORRECTED_TABLE_VERSION,
                expected_method=CORRECTED_TABLE_METHOD,
                actual_table_sha256=self.TABLE_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
