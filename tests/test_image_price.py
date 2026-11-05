"""اختبارات حدود بروتوكول صورة واحدة لكل إعلان؛ لا تستعمل مجموعة test الفعلية."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import image_price_train as task
from image_price_runtime import load_image, ImagePriceRuntime


class ImagePriceProtocolTests(unittest.TestCase):
    def listing(self, uid='a', split='train'):
        return dict(listing_uid=uid,split=split,group_id=uid,usable_price='True',condition='used',sar_price='100000')

    def image(self, uid='a', number='01', **extra):
        row=dict(listing_uid=uid,image_uid=uid+':'+number,split='train',group_id=uid,sha256=uid+number,
                 is_exterior='True',content_kind='vehicle_overview',usable_visual_search='True')
        row.update(extra);return row

    def test_selection_is_one_per_ad_and_does_not_depend_on_price(self):
        ads=[self.listing()];photos=[self.image(number='02'),self.image()]
        a=task.choose_images(ads,photos)
        ads[0]['sar_price']='999';photos.reverse()
        b=task.choose_images(ads,photos)
        self.assertEqual(a,b);self.assertEqual(len(a),1);self.assertEqual(a[0]['image_uid'],'a:01')

    def test_rejects_interior_document_and_unusable(self):
        photos=[self.image(is_exterior='False'),self.image(content_kind='document'),self.image(usable_visual_search='False')]
        self.assertEqual(task.choose_images([self.listing()],photos),[])

    def test_new_car_never_enters_training_pool(self):
        ad=self.listing();ad['condition']='new'
        self.assertEqual(task.choose_images([ad],[self.image()]),[])

    def test_image_listing_split_must_agree(self):
        with self.assertRaises(ValueError):task.choose_images([self.listing()],[self.image(split='val')])

    def test_duplicate_listing_is_rejected(self):
        with self.assertRaises(ValueError):task.validate_rows([self.image(),self.image(number='02')])

    def test_sha_and_group_cross_split_rejected(self):
        a=self.image();b=self.image(uid='b',split='val',sha256=a['sha256'])
        with self.assertRaises(ValueError):task.validate_rows([a,b])
        b['sha256']='other';b['group_id']=a['group_id']
        with self.assertRaises(ValueError):task.validate_rows([a,b])

    def test_no_experiments_after_lock(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'selection_lock.json').write_text('{}')
            with self.assertRaises(ValueError):task.selection_allowed(run)

    def test_test_event_cannot_be_acquired_twice(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);task.write_json(run/'selection_lock.json',{'sha256':{}})
            (run/'test_once.json').write_text('{}')
            with patch.object(task,'check_inputs'),patch.object(task,'comparison') as evaluate:
                with self.assertRaises(FileExistsError):task.test_once(run)
                evaluate.assert_not_called()

    def test_invalid_image_has_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'invalid.jpg';p.write_bytes(b'not an image')
            with self.assertRaisesRegex(ValueError,'تعذر قراءة'):load_image(p)

    def test_runtime_refuses_tampered_artifact_before_deserialization(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'weights.pt').write_bytes(b'changed')
            task.write_json(run/'runtime_policy.json',{'artifacts_sha256':{'weights.pt':'incorrect'}})
            with self.assertRaisesRegex(ValueError,'بصمة'):
                ImagePriceRuntime(run)


if __name__=='__main__':unittest.main()
