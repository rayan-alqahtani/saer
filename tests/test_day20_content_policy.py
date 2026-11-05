import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_data_policy import reviewed_overview, assign_recognition
from day20_content_policy import labels, price_eligibility, assign_attribute_task, support_tier


class ContentPolicyTests(unittest.TestCase):
    def test_historical_clean_cannot_overwrite_reviewed_dataset(self):
        import day20_clean
        with patch.object(day20_clean, "load_json", return_value={"content_review_required": True}):
            with self.assertRaises(SystemExit):
                day20_clean.clean()

    def row(self, **kw):
        return dict({"color": "dark leaden", "body_type": "sedan", "trim": "TITANIUM", "make": "Nissan", "model_std": "Patrol"}, **kw)

    def test_source_exterior_document_is_never_overview(self):
        self.assertFalse(reviewed_overview({"is_exterior": "True", "content_review_status": "reviewed", "content_kind": "document"}))

    def test_source_unknown_reviewed_overview_can_be_used(self):
        self.assertTrue(reviewed_overview({"is_exterior": "", "content_review_status": "reviewed", "content_kind": "vehicle_overview"}))

    def test_absent_review_fails_closed(self):
        self.assertFalse(reviewed_overview({"is_exterior": "True", "content_kind": "vehicle_overview"}))

    def test_visual_body_overrides_source_without_mutating_source(self):
        row = self.row(body_type="sport")
        self.assertEqual(labels(row, {"body_visual": "sedan"})["body_type_std"], "sedan")
        self.assertEqual(row["body_type"], "sport")

    def test_manual_unknown_does_not_fall_back_to_source(self):
        self.assertEqual(labels(self.row(), {"body_visual": "unknown"})["body_type_std"], "")

    def test_color_family_and_exact_trim_case_only(self):
        result = labels(self.row())
        self.assertEqual(result["color_std"], "gray")
        self.assertEqual(result["trim_std"], "Titanium")
        self.assertEqual(labels(self.row(color="petrol"))["color_std"], "")
        self.assertEqual(labels(self.row(model_std="Other"))["trim_std"], "TITANIUM")

    def test_unconfirmed_unusual_mileage_is_retained_but_not_price_eligible(self):
        row = {"condition": "used", "data_quality_flags": "mileage_exact_100_review", "mileage_source_status": "source_unavailable"}
        old = row.copy()
        self.assertEqual(price_eligibility(row), (False, "unusual_mileage_unconfirmed"))
        self.assertEqual(row, old)
        row["mileage_source_status"] = "source_confirms_record"
        self.assertTrue(price_eligibility(row)[0])

    def test_small_test_support_is_experimental(self):
        self.assertEqual(support_tier({"train": 10, "val": 1, "test": 1}), "experimental_minimum")
        self.assertEqual(support_tier({"train": 20, "val": 5, "test": 5}), "larger_evaluation_sample")

    def test_exact_photo_conflicting_color_cannot_supply_support(self):
        listings, images = [], []
        for i, split in enumerate(["train"] * 8 + ["val"] * 2 + ["test"] * 2):
            listings.append({"listing_uid": str(i)})
            images.append({"listing_uid": str(i), "image_uid": str(i), "group_id": str(i), "sha256": str(i),
                           "split": split, "content_review_status": "reviewed", "content_kind": "vehicle_overview", "label_color_std": "white"})
        images.append(dict(images[0], image_uid="conflict", label_color_std="black"))
        assign_attribute_task(listings, images, "color", "label_color_std")
        self.assertTrue(all(not r["usable_color"] for r in images))
        self.assertEqual(images[0]["color_status"], "conflicting_exact_image_labels")


if __name__ == "__main__": unittest.main()
