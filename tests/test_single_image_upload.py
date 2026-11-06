"""فحص حدود الرفع دون تحميل أوزان أو إنشاء نتائج تنبؤ بديلة."""
import ast
import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class SingleImageUploadTests(unittest.TestCase):
    def functions(self, path, names, context):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        for node in nodes:
            node.decorator_list = []
        module = ast.parse("from __future__ import annotations\n")
        module.body.extend(nodes)
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), context)
        return context

    def test_gradio_rejects_missing_out_of_range_and_invalid_before_inference(self):
        calls = []
        context = self.functions("hf_backend/app.py", {"_valid_image_path", "estimate_image_price", "identify_vehicle"}, {
            "Path": Path, "_startup_error": None,
            "services": SimpleNamespace(predict_image_price=lambda *args: calls.append(args)),
        })
        for bad in ([], ["a"] * 9):
            with self.assertRaisesRegex(ValueError, "من صورة واحدة إلى 8 صور"):
                context["estimate_image_price"](bad)
        for bad in (None, "", ["a", "b"]):
            with self.assertRaises(ValueError):
                context["estimate_image_price"](bad)
        for bad in (None, "", ["a", "b"]):
            with self.assertRaises(ValueError):
                context["identify_vehicle"](bad)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jpg"
            path.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "غير صالح"):
                context["_valid_image_path"](str(path))
            path = Path(folder) / "valid.png"
            Image.new("RGB", (4, 4)).save(path)
            self.assertEqual(context["_valid_image_path"](str(path)), path)
        self.assertEqual(calls, [])

    def test_gradio_inputs_match_live_frontend_contract(self):
        tree = ast.parse((ROOT / "hf_backend/app.py").read_text(encoding="utf-8"))
        files = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"price_files", "identity_file"} for t in node.targets)]
        self.assertEqual(len(files), 2)
        by_name = {next(t.id for t in node.targets if isinstance(t, ast.Name) and t.id in {"price_files", "identity_file"}): node for node in files}
        for name, expected_count in (("identity_file", "single"), ("price_files", "multiple")):
            node = by_name[name]
            keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.value.keywords}
            self.assertEqual(keywords.get("file_count", "single"), expected_count)
            self.assertIsNone(keywords.get("file_types"), "Gradio must defer image validation to Pillow")

    def test_rest_rejects_repeated_file_fields(self):
        from fastapi import HTTPException, File
        context = self.functions("backend/app/main.py", {"recognize", "estimate_image_price"}, {"HTTPException": HTTPException, "File": File})

        class Request:
            async def form(self):
                return SimpleNamespace(getlist=lambda key: ["first", "second"])

        for name in ("recognize", "estimate_image_price"):
            with self.assertRaises(HTTPException) as error:
                asyncio.run(context[name](Request(), "unused"))
            self.assertEqual(error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
