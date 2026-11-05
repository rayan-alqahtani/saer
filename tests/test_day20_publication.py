"""حماية الملف المفتوح وحدود نقل النسخة النهائية؛ لا تلمس بيانات المستخدم."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from day20_common import ROOT
from day20_publish_enriched import checked_move, replace_available


class PublicationSafetyTests(unittest.TestCase):
    def test_raw_and_outside_workspace_moves_are_rejected_before_mutation(self):
        with self.assertRaises(ValueError):
            checked_move(ROOT / "data/raw", ROOT / "data/staging/raw_must_not_move")
        with self.assertRaises(ValueError):
            checked_move(Path(ROOT.anchor) / "outside_workspace", ROOT / "data/staging/invalid_move")

    @unittest.skipUnless(os.name == "nt", "خاص بقفل ملفات Windows")
    def test_excel_style_reader_lock_prevents_publication(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        with tempfile.TemporaryDirectory(prefix="car-publication-") as directory:
            path = Path(directory) / "listings.csv"
            path.write_bytes(b"original data")
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 128, None)
            self.assertNotEqual(handle, wintypes.HANDLE(-1).value)
            try:
                with self.assertRaises(PermissionError):
                    replace_available([path])
                self.assertEqual(path.read_bytes(), b"original data")
            finally:
                kernel.CloseHandle(handle)
            replace_available([path])


if __name__ == "__main__":
    unittest.main()
