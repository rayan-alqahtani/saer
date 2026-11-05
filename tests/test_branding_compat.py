import tempfile
import unittest
from pathlib import Path
from backend.app.branding_compat import database_path, session_token

class BrandingCompatibilityTests(unittest.TestCase):
    def test_database_name(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(database_path(Path(directory)), Path(directory) / "saer.sqlite")

    def test_session_cookie(self):
        self.assertEqual(session_token({"saer_session": "current"}), "current")
        self.assertIsNone(session_token({}))

if __name__ == "__main__":
    unittest.main()
