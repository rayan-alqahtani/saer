"""إعادة إنتاج قفل Windows الذي ظهر أثناء استرجاع الصور."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import sys
import threading
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from day20_common import save_json

@unittest.skipUnless(sys.platform == "win32", "اختبار قفل ملفات Windows")
class AtomicSaveTests(unittest.TestCase):
    def test_reader_lock_does_not_lose_manifest_update(self):
        directory = ROOT / "tmp" / ("windows-save-test-" + uuid.uuid4().hex[:10])
        directory.mkdir(parents=True)
        path = directory / "manifest.json"
        save_json(path, {"images": 1})
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 128, None)
        self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
        release = threading.Timer(.2, lambda: kernel.CloseHandle(handle))
        release.start()
        try:
            save_json(path, {"images": 2})
        finally:
            release.join()
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"images": 2})
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        history = ROOT / "data/history" / path.relative_to(ROOT)
        backups = list(history.parent.glob("manifest.json.*"))
        self.assertTrue(any(json.loads(p.read_text(encoding="utf-8")) == {"images": 1} for p in backups))

if __name__ == "__main__":
    unittest.main()
