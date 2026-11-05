"""تحقق من منع التسرب والإبقاء على مجموعات الإعلان، لا اختبارات شكلية."""
from collections import defaultdict
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from day20_clean import split_groups
from day20_quality import UnionFind
from day20_common import read_csv

class SplitTests(unittest.TestCase):
    def test_repeated_listing_group_cannot_cross_splits(self):
        rows = []
        for i in range(30):
            rows.append({"listing_uid": f"ad-{i}", "group_id": f"group-{i//2}", "make": "A", "model": "M"})
        mapping = split_groups(rows)
        self.assertEqual(set(mapping.values()), {"train", "val", "test"})
        self.assertEqual(mapping, split_groups(rows))
        self.assertEqual(len(mapping), 15)
        pairs = defaultdict(set)
        for r in rows:
            pairs[r["group_id"]].add(mapping[r["group_id"]])
        self.assertTrue(all(len(v) == 1 for v in pairs.values()))

    def test_transitive_duplicate_chain_is_one_group(self):
        uf = UnionFind(["c", "b", "a", "d"])
        uf.union("c", "b")
        uf.union("b", "a")
        self.assertEqual(uf.find("a"), uf.find("c"))
        self.assertNotEqual(uf.find("a"), uf.find("d"))

@unittest.skipUnless((ROOT / "data/processed/listings.csv").is_file(), "الجداول الموحدة لم تنشأ بعد؛ تبقى بوابة الدمج مغلقة")
class SavedDatasetTests(unittest.TestCase):
    def test_saved_image_and_group_splits_have_no_leakage(self):
        groups, hashes = defaultdict(set), defaultdict(set)
        for r in read_csv(ROOT / "data/processed/listings.csv"):
            groups[r["group_id"]].add(r["split"])
        for r in read_csv(ROOT / "data/processed/images.csv"):
            hashes[r["sha256"]].add(r["split"])
        self.assertTrue(all(len(v) == 1 for v in groups.values()))
        self.assertTrue(all(len(v) == 1 for v in hashes.values()))

    def test_original_rows_reconcile_to_clean_and_quarantine(self):
        accepted = read_csv(ROOT / "data/processed/listings.csv")
        quarantined = read_csv(ROOT / "data/processed/quarantine_listings.csv")
        self.assertEqual(len(accepted) + len(quarantined), 2803)
        self.assertFalse({r["listing_uid"] for r in accepted} & {r["listing_uid"] for r in quarantined})

if __name__ == "__main__":
    unittest.main()
