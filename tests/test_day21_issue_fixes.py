import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_data_policy import overview_exclusion_status
from day21_dataset_contract import separate_flags, recognition_route, split_fingerprint, assert_artifact_compatible
from day20_finalize_data import checked_publication_context
from day20_common import ROOT


class IssueRegressionTests(unittest.TestCase):
    def test_publication_context_accepts_new_audit_day_but_not_root_or_raw(self):
        stage, baseline = ROOT / "data/staging/day21_user_issues", ROOT / "data/audit/day21/user_issues"
        self.assertEqual(checked_publication_context(stage, baseline), (stage.resolve(), baseline.resolve()))
        with self.assertRaises(ValueError): checked_publication_context(ROOT / "data/staging", baseline)
        with self.assertRaises(ValueError): checked_publication_context(stage, ROOT / "data/audit")
        with self.assertRaises(ValueError): checked_publication_context(stage, ROOT / "data/raw")

    def test_compound_flags_are_counted_individually(self):
        flags = separate_flags({"data_quality_flags": "global_price_iqr_outlier_review;recovery_source_redirected_or_unavailable"})
        self.assertTrue(flags["flag_price_iqr_outlier"])
        self.assertTrue(flags["flag_source_unavailable"])
        self.assertFalse(flags["flag_high_annual_mileage"])

    def test_reviewed_document_is_excluded_not_pending(self):
        self.assertEqual(overview_exclusion_status({"content_review_status": "reviewed", "content_kind": "document"}), "reviewed_not_vehicle_overview")
        self.assertEqual(overview_exclusion_status({"content_kind": "vehicle_overview"}), "content_review_pending")

    def test_make_fallback_keeps_original_model_identity(self):
        row = {"make": "A", "model": "Rare", "rec_label": "A|Rare", "usable_recognition": "False", "usable_make_recognition": "True"}
        before = row.copy()
        self.assertEqual(recognition_route(row), "make_only")
        self.assertEqual(row, before)
        row["usable_make_recognition"] = "False"
        self.assertEqual(recognition_route(row), "unavailable")

    def test_split_fingerprint_is_order_independent_but_changes_with_fold(self):
        rows = [{"listing_uid": "a", "group_id": "g1", "split": "train"}, {"listing_uid": "b", "group_id": "g2", "split": "test"}]
        before = split_fingerprint(rows)
        self.assertEqual(before, split_fingerprint(list(reversed(rows))))
        rows[0]["split"] = "val"
        self.assertNotEqual(before, split_fingerprint(rows))

    def test_artifact_without_matching_dataset_identity_is_rejected(self):
        manifest = {"dataset_version": "v6", "dataset_fingerprint": "data", "split_fingerprint": "new_split"}
        self.assertTrue(assert_artifact_compatible(dict(manifest), manifest))
        with self.assertRaises(ValueError): assert_artifact_compatible({}, manifest)
        with self.assertRaises(ValueError): assert_artifact_compatible(dict(manifest, split_fingerprint="old_split"), manifest)


if __name__ == "__main__": unittest.main()
