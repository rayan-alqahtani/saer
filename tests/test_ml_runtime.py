import sys
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import ml_common
import ml_delivery_runtime as runtime


class RuntimeSafety(unittest.TestCase):
    def test_started_run_does_not_rewrite_shared_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(ml_common,'ROOT',root), patch.object(ml_common,'verified_inputs',return_value={}), \
                 patch.object(ml_common,'sha256',return_value='digest'), \
                 patch.object(ml_common,'load_json',return_value={'training_started':True,'training_allowed':True}), \
                 patch.object(ml_common,'save_json') as save:
                ml_common.new_run('test')
                paths=[call.args[0] for call in save.call_args_list]
                self.assertEqual(len(paths),1)
                self.assertEqual(paths[0].name,'metadata.json')

    def test_package_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'changed.txt').write_text('changed')
            with patch.object(runtime,'HERE',root),patch.object(runtime,'read_json',return_value={'files_sha256':{'changed.txt':'wrong'}}):
                with self.assertRaisesRegex(ValueError,'تغير'): runtime.check_files()

    def test_portable_price_rejects_new_scope(self):
        with patch.object(runtime,'load',return_value={}):
            with self.assertRaisesRegex(ValueError,'للمستعمل'): runtime.price({'condition':'new'})


if __name__=='__main__': unittest.main()
