"""حالات انحدار لأخطاء البيانات الفعلية التي رصدت في 20 أكتوبر."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from day20_common import public_page_data, number, inferred_model, normalize_model
from day20_recover import PublicClient

def saved_page(sid, source):
    sample = ROOT / "data/recovery/samples" / (sid + ".html")
    if sample.exists():
        return sample
    return ROOT / "data/recovery/pages" / source / (sid + ".html")

class RecoveryRegressionTests(unittest.TestCase):
    def test_new_listing_color_variant_does_not_hide_cash_price(self):
        sample = saved_page("291104", "external_group_b")
        result = public_page_data(sample.read_text(encoding="utf-8"), "291104")
        self.assertEqual(result["fields"]["sar_price"], 167900)
        self.assertEqual(result["fields"]["model"], "Taurus")
        self.assertEqual(result["fields"]["body_type"], "Sedan")
        self.assertIsNone(result["fields"]["km_mileage"])

    def test_heic_urls_are_not_silently_discarded(self):
        sample = saved_page("317755", "external_group_a")
        result = public_page_data(sample.read_text(encoding="utf-8"), "317755")
        self.assertTrue(result["photos"])
        self.assertTrue(any(".heic" in p["url"] for p in result["photos"]))
        self.assertTrue(all("/posts/317755/" in p["url"] for p in result["photos"]))

    def test_wrong_listing_is_rejected_even_with_similar_posts(self):
        sample = saved_page("291104", "external_group_b")
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            public_page_data(sample.read_text(encoding="utf-8"), "292274")

    def test_missing_mileage_is_not_fabricated_zero(self):
        self.assertIsNone(number(None))
        self.assertIsNone(number("unknown"))
        self.assertEqual(number("0"), 0)
        self.assertEqual(number("١٦٠٬٤٢٤"), 160424)

    def test_different_car_models_are_not_collapsed(self):
        self.assertEqual(normalize_model("Kia", "Optima"), "Optima")
        self.assertEqual(normalize_model("Kia", "K5"), "K5")
        self.assertNotEqual(normalize_model("BMW", "X Series"), "X5")

    def test_model_can_be_reconstructed_only_from_explicit_trim_identifier(self):
        self.assertEqual(normalize_model("BMW", "X Series", "X2 sDrive 20i"), "X2")
        self.assertEqual(normalize_model("BMW", "X Series", "4 xDrive 30i"), "X4")
        self.assertEqual(normalize_model("BMW", "X Series", "X1sDrive20i A"), "X1")
        self.assertEqual(normalize_model("BMW", "X Series", "20i"), "X Series")
        self.assertEqual(normalize_model("BMW", "X Series", "غير معروف"), "X Series")
        self.assertEqual(normalize_model("Mitsubishi", "New Outlander Series"), "Outlander")

    def test_unverified_slug_model_keeps_provenance(self):
        model, source = inferred_model({"make": "Kia", "url_source": "https://example.invalid/cardetail/kia-sonet-used-123", "model": "", "model_raw": ""})
        self.assertEqual(model, "Sonet")
        self.assertEqual(source, "listing_url_unverified")

    def test_robots_longest_rule_and_wildcards(self):
        client = PublicClient()
        client.rules["example.invalid"] = [(False, "/api/"), (False, "/*?*"), (True, "/*?page=*"), (False, "/*user/*")]
        self.assertTrue(client.allowed("https://example.invalid/cardetail/kia-123"))
        self.assertFalse(client.allowed("https://example.invalid/api/private"))
        self.assertFalse(client.allowed("https://example.invalid/cardetail/kia-123?x=1"))
        self.assertTrue(client.allowed("https://example.invalid/autos?page=2"))
        self.assertFalse(client.allowed("https://example.invalid/ar/user/profile"))

if __name__ == "__main__":
    unittest.main()
