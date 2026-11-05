import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_common import ROOT, AUDIT
from day20_supplemental_validate import gallery_identity
from day20_finalize_data import publish


class SupplementalSourceTests(unittest.TestCase):
    def fixture(self):
        url = "https://cdn.example.invalid/photos-thumbs/online-v1/0x683/online/posts/123/original.heic?v=3"
        return {"source_url": url, "image_kind": "exterior"}, {"external_listing_id": "123"}, {
            "external_listing_id": "123", "page_identity_verified": True, "html_path": "data/recovery/test.html"}

    def test_actual_gallery_size_url_is_valid_despite_filename_extension(self):
        photo, parent, parsed = self.fixture()
        actual = {"photos": [{"url": photo["source_url"].replace("0x683", "0x300"), "gallery_url": photo["source_url"], "image_kind": "exterior"}]}
        with patch("day20_supplemental_validate.read_html", return_value=b"fixture"), patch("day20_supplemental_validate.public_page_data", return_value=actual):
            self.assertTrue(gallery_identity(photo, parent, parsed))

    def test_same_filename_from_another_listing_is_rejected(self):
        photo, parent, parsed = self.fixture()
        photo["source_url"] = photo["source_url"].replace("/123/", "/456/")
        self.assertFalse(gallery_identity(photo, parent, parsed))

    def test_matching_id_does_not_authorize_invented_gallery_url(self):
        photo, parent, parsed = self.fixture()
        actual = {"photos": [{"url": photo["source_url"].replace("original", "different"), "image_kind": "exterior"}]}
        with patch("day20_supplemental_validate.read_html", return_value=b"fixture"), patch("day20_supplemental_validate.public_page_data", return_value=actual):
            self.assertFalse(gallery_identity(photo, parent, parsed))

    def test_publication_refuses_outside_staging_before_mutation(self):
        with self.assertRaises(ValueError):
            publish(ROOT / "data/staging/dummy", ROOT.parent, AUDIT)


if __name__ == "__main__": unittest.main()
