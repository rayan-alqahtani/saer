import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_data_policy import assign_recognition, assign_make_recognition, descriptive_price_bounds, mileage_flags


class MileagePolicyTests(unittest.TestCase):
    def test_low_used_mileage_is_flagged_without_mutation(self):
        row = {"condition": "used", "km_mileage": "100", "year": "2026", "date_collection": "2026-10-14"}
        saved = row.copy()
        self.assertEqual(set(mileage_flags(row)), {"mileage_exact_100_review", "young_used_low_mileage_review"})
        self.assertEqual(row, saved)

    def test_new_missing_mileage_is_never_imputed(self):
        self.assertEqual(mileage_flags({"condition": "new", "km_mileage": ""}), [])

    def test_annual_proxy_uses_collection_year_and_one_year_floor(self):
        row = {"condition": "used", "km_mileage": "60001", "year": "2026", "date_collection": "2026-10-14"}
        self.assertIn("annualized_mileage_proxy_high_review", mileage_flags(row))
        row["km_mileage"] = "60000"
        self.assertNotIn("annualized_mileage_proxy_high_review", mileage_flags(row))

    def test_iqr_describes_prices_without_dropping_luxury_ads(self):
        rows = [{"condition": "used", "sar_price": p} for p in [10, 20, 30, 40, 1000]]
        self.assertEqual(descriptive_price_bounds(rows), (-10, 70))
        self.assertEqual(len(rows), 5)


class RecognitionPolicyTests(unittest.TestCase):
    def fixture(self, exterior_splits):
        listings, images = [], []
        for i, split in enumerate(["train"] * 8 + ["val"] * 2 + ["test"] * 2):
            listings.append({"listing_uid": str(i), "group_id": str(i), "make": "A", "model": "M", "split": split})
            images.append({"image_uid": str(i), "listing_uid": str(i), "sha256": str(i), "is_exterior": split in exterior_splits,
                           "content_review_status": "reviewed", "content_kind": "vehicle_overview" if split in exterior_splits else "interior"})
        return listings, images

    def test_listing_support_cannot_hide_missing_exterior_validation(self):
        listings, images = self.fixture({"train", "test"})
        support, eligible = assign_recognition(listings, images)
        self.assertTrue(support[0]["listing_supported"])
        self.assertFalse(eligible)
        self.assertTrue(all(not r["usable_recognition"] for r in images + listings))
        self.assertIn("insufficient_val_groups", support[0]["recognition_status"])

    def test_supported_images_keep_labels_and_listing_flags_agree(self):
        listings, images = self.fixture({"train", "val", "test"})
        _, supported = assign_recognition(listings, images)
        self.assertEqual(supported, {"A|M"})
        self.assertTrue(all(r["usable_recognition"] for r in images + listings))
        self.assertTrue(all(r["rec_label"] == "A|M" for r in images))

    def test_duplicate_images_do_not_supply_independent_groups(self):
        listings, images = self.fixture({"train", "val", "test"})
        for image in images:
            if image["listing_uid"] == "8":
                image["sha256"] = "0"
        support, eligible = assign_recognition(listings, images)
        self.assertFalse(eligible)
        self.assertEqual(support[0]["exterior_val_groups"], 1)

    def test_rare_models_can_have_supported_brand_without_merging_models(self):
        listings, images = self.fixture({"train", "val", "test"})
        for index, (listing, image) in enumerate(zip(listings, images)):
            listing["model"] = "rare-" + str(index)
            image.update(label_make="A", group_id=listing["group_id"], split=listing["split"])
        _, models = assign_recognition(listings, images)
        _, makes = assign_make_recognition(listings, images)
        self.assertFalse(models)
        self.assertEqual(makes, {"A"})
        self.assertTrue(all(r["usable_make_recognition"] for r in images))
        self.assertEqual(len({r["rec_label"] for r in images}), 12)


if __name__ == "__main__":
    unittest.main()
