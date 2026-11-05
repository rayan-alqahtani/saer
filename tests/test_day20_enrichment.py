import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
import io
import urllib.error
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_common import public_page_data, normalize_model
from day20_enrichment_policy import mileage_source_status, read_html, overlay_completed, verified_evidence
from day20_recover import PublicClient
from day20_clean import canonical_make


class EnrichmentPolicyTests(unittest.TestCase):
    def test_verified_make_spelling_does_not_create_identity_conflict(self):
        self.assertEqual(canonical_make("Lynk and Co"), canonical_make("Lynk & Co"))
        self.assertNotEqual(canonical_make("BAIC"), canonical_make("BMW"))
    def test_partial_collection_cannot_be_published(self):
        with patch("day20_enrichment_policy.load_json", return_value={"complete": False, "checked": 10, "queue_size": 2435}):
            with self.assertRaisesRegex(ValueError, "استكمال المصدر جارٍ"):
                overlay_completed({"pages": {}, "images": []})

    def test_modified_source_evidence_is_rejected(self):
        import day20_enrichment_policy as policy
        import hashlib
        with tempfile.TemporaryDirectory(prefix="car-evidence-") as directory:
            root = Path(directory)
            target = root / "page.html.gz"
            original = b"verified source"
            target.write_bytes(gzip.compress(original, mtime=0))
            parsed = {"page_identity_verified": True, "html_path": target.name, "html_sha256": hashlib.sha256(original).hexdigest()}
            with patch.object(policy, "ROOT", root), patch.object(policy, "RECOVERY", root):
                self.assertTrue(verified_evidence(parsed))
                target.write_bytes(gzip.compress(b"modified source", mtime=0))
                self.assertFalse(verified_evidence(parsed))

    def test_land_cruiser_seventy_is_not_collapsed_into_land_cruiser(self):
        self.assertEqual(normalize_model("Toyota", "LAND CRUISER 70"), "Land Cruiser 70")
        self.assertNotEqual(normalize_model("Toyota", "LAND CRUISER 70"), normalize_model("Toyota", "Land Cruiser"))
        self.assertEqual(normalize_model("Nissan", "Xtrail"), "X-Trail")
    def test_source_confirms_one_hundred_without_invented_correction(self):
        status, value = mileage_source_status("used", 100, {"fields": {"km_mileage": 100}})
        self.assertEqual(status, "source_confirms_record")
        self.assertEqual(value, 100)

    def test_source_change_keeps_historical_value_distinct(self):
        page = {"fields": {"km_mileage": 1200}}
        self.assertEqual(mileage_source_status("used", 1000, page), ("source_differs_historical_record", 1200))
        self.assertEqual(page["fields"]["km_mileage"], 1200)

    def test_missing_source_does_not_confirm_or_zero_fill(self):
        self.assertEqual(mileage_source_status("used", 100, None), ("source_unavailable", None))
        self.assertEqual(mileage_source_status("new", "", None), ("not_required_for_new", None))

    def test_compressed_and_original_evidence_read_identically(self):
        with tempfile.TemporaryDirectory(prefix="car-evidence-") as directory:
            root = Path(directory)
            body = '<html lang="ar">دليل المصدر</html>'.encode()
            (root / "page.html").write_bytes(body)
            (root / "page.html.gz").write_bytes(gzip.compress(body, mtime=0))
            self.assertEqual(read_html(root / "page.html"), read_html(root / "page.html.gz"))

    def test_equipment_grade_is_not_promoted_to_missing_trim(self):
        post = {"details": {"id": 1}, "analytics": {"id": 1, "options": "Standard"},
                "g4Data": {"post_id": 1, "post_ext": None}, "gallery": {"images": []}}
        body = "window.FULL_PAGE_DATA = " + json.dumps({"postDetails": post})
        self.assertEqual(public_page_data(body, "1")["fields"]["trim"], "")

    def test_compressed_s3_missing_robots_is_read_before_policy_decision(self):
        error = urllib.error.HTTPError("https://cdn.example.invalid/robots.txt", 403, "Forbidden",
                  {"Content-Encoding": "gzip"}, io.BytesIO(gzip.compress(b"<Error><Code>AccessDenied</Code></Error>")))
        client = PublicClient(compressed=True)
        client.log = lambda value: None
        with patch("urllib.request.urlopen", side_effect=error), patch("time.sleep"):
            body, status = client.get("https://cdn.example.invalid/robots.txt", robots=True)
        self.assertEqual(body, b"")
        self.assertTrue(status["missing_cdn_robots"])
        self.assertNotIn("cdn.example.invalid", client.stopped)

    def test_actual_image_access_denial_still_stops_host(self):
        url = "https://cdn.example.invalid/photos/test.webp"
        error = urllib.error.HTTPError(url, 403, "Forbidden", {"Content-Encoding": "gzip"},
                  io.BytesIO(gzip.compress(b"<Error><Code>AccessDenied</Code></Error>")))
        client = PublicClient(compressed=True)
        client.log = lambda value: None
        client.rules["cdn.example.invalid"] = []
        with patch("urllib.request.urlopen", side_effect=error), patch("time.sleep"):
            body, _ = client.get(url)
        self.assertIsNone(body)
        self.assertEqual(client.stopped["cdn.example.invalid"], 403)


if __name__ == "__main__":
    unittest.main()
