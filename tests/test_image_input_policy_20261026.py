import hashlib
import queue
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from image_input_policy_20261026 import decode_rgb,normalized_input_path
from model_image_tester import ModelImageTester

class InputPolicyTest(unittest.TestCase):
    def test_visible_colors_and_transparency_are_composited(self):
        im=Image.new('RGBA',(3,1))
        im.putdata([(0,0,0,0),(255,0,0,128),(3,5,7,255)])
        rgb=decode_rgb(im)
        self.assertEqual([rgb.getpixel((x,0)) for x in range(3)],[(255,255,255),(255,127,127),(3,5,7)])

    def test_opaque_source_path_is_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'opaque.png';Image.new('RGB',(8,8),'red').save(p)
            before=p.read_bytes()
            with normalized_input_path(p) as model_path:self.assertEqual(model_path,p)
            self.assertEqual(p.read_bytes(),before)

    def test_worker_normalizes_all_heads_consistently_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'transparent.png';Image.new('RGBA',(8,8),(0,0,0,0)).save(p)
            before=hashlib.sha256(p.read_bytes()).hexdigest();seen=[]
            class Engine:
                def extract(self,path):
                    seen.append(path)
                    with Image.open(path) as im:
                        assert im.mode=='RGB' and im.getpixel((0,0))==(255,255,255)
                    return {'same_features':True}
                def predict(self,task,path,k,features):
                    assert path==seen[0] and features['same_features']
                    seen.append(path);return {'task':task}
            tester=ModelImageTester.__new__(ModelImageTester);tester.engine=Engine();tester.events=queue.Queue()
            tester.predict_worker(['recognize','color','body','search'],p)
            event,results=tester.events.get_nowait()
            self.assertEqual(event,'result');self.assertEqual(len(results),4)
            self.assertNotEqual(seen[0],p);self.assertFalse(seen[0].exists())
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),before)

    def test_worker_cleanup_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'transparent.png';Image.new('RGBA',(8,8)).save(p);seen=[]
            class Engine:
                def extract(self,path):seen.append(path);raise ValueError('expected failure')
            tester=ModelImageTester.__new__(ModelImageTester);tester.engine=Engine();tester.events=queue.Queue()
            tester.predict_worker(['recognize'],p)
            self.assertEqual(tester.events.get_nowait()[0],'predict_error')
            self.assertFalse(seen[0].exists());self.assertTrue(p.exists())

    def test_new_tester_loads_the_requested_model_factory(self):
        marker=object();tester=ModelImageTester.__new__(ModelImageTester)
        tester.events=queue.Queue();tester.engine_factory=lambda:marker
        tester.load_engine();event,payload=tester.events.get_nowait()
        self.assertEqual(event,'loaded');self.assertIs(payload,marker)

    def test_model_factory_failure_is_reported(self):
        def fail():raise ValueError('model mismatch')
        tester=ModelImageTester.__new__(ModelImageTester);tester.events=queue.Queue();tester.engine_factory=fail
        tester.load_engine();event,payload=tester.events.get_nowait()
        self.assertEqual(event,'load_error');self.assertIn('model mismatch',payload)

if __name__=='__main__':unittest.main()
