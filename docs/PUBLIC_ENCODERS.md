<div dir="rtl">

# المشفرات العامة غير المرفقة

البيان القابل للتحقق: `PUBLIC_ENCODERS.json`، 15 ملفاً و2,920,567,687 بايت (نحو 2.72 GiB)، إضافة إلى شفرة DINOv2. لا توكنات مطلوبة ولا وزن عام كبير في Git أو ZIP التسليم.

| النموذج | المصدر الرسمي | الإصدار/التثبيت | المسار تحت منزل المستخدم |
|---|---|---|---|
| DINOv2 ViT-S/14 | dl.fbaipublicfiles.com/dinov2 | dinov2_vits14_pretrain.pth؛ SHA-256 في JSON | .cache/torch/hub/checkpoints |
| DINOv2 ViT-B/14 | المصدر نفسه | dinov2_vitb14_pretrain.pth؛ SHA-256 | المسار نفسه |
| DINOv2 ViT-L/14 | المصدر نفسه | dinov2_vitl14_pretrain.pth؛ SHA-256 | المسار نفسه |
| ResNet18 | download.pytorch.org/models | IMAGENET1K_V1، resnet18-f37072fd.pth | المسار نفسه |
| SSDLite320 MobileNet V3 Large | download.pytorch.org/models | COCO_V1، ssdlite320_mobilenet_v3_large_coco-a79551df.pth | المسار نفسه |
| CLIP ViT-B/16 | huggingface.co/openai/clip-vit-base-patch16 | 57c216476eefef5ab752ec549e440a49ae4ae5f3؛ config.json وpytorch_model.bin | .cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/REV |
| CLIP ViT-B/32 | huggingface.co/openai/clip-vit-base-patch32 | config/tokenizer عند 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268، model.safetensors عند c237dc49a33fc61debc9276459120b7eac67e7ef | models--openai--clip-vit-base-patch32/snapshots/REV تحت كاش Hugging Face نفسه |
| شفرة DINOv2 | github.com/facebookresearch/dinov2 | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 | .cache/torch/hub/facebookresearch_dinov2_main |

روابط تنزيل كل ملف وبصمته وحجمه الدقيقة داخل JSON. أسماء مستودعات CLIP التقنية ضرورية لاستعادة الأوزان كما هي. السكربت يحافظ على ملفات الترخيص داخل أرشيف شفرة DINOv2؛ راجع تراخيص المصادر قبل توزيع المشفرات أو استخدامها خارج العرض الأكاديمي.

من جذر نسخة التسليم:

```powershell
.runtime/clean-main/Scripts/python.exe tools/restore_public_encoders.py --download
.runtime/clean-main/Scripts/python.exe tools/restore_public_encoders.py
```

لا يبدأ الاستدلال أو التدريب. ينزّل المفقود فقط ويوقف العملية عند بصمة مختلفة بدلاً من استبدال ملف موجود. قد يبقى ملف .part إذا انقطع التنزيل؛ لا يستعمله التشغيل قبل التحقق. لا يعرض السكربت أسراراً أو يطلبها.

الكود الحالي للهوية يقرأ Path.home() صراحة؛ لذلك لا يُدّعى دعم كاش في مكان اختياري. الخيار --home لاختبار الاستعادة فقط. ينشئ السكربت refs/main الخاص بـCLIP B/16 عند revision المثبت، وينشئ مجلد شفرة DINOv2 بالاسم الذي يتوقعه الكود. بعد اكتمال الاستعادة يُشغّل الخادم في الوضع offline حتى لا يتغير المرجع main ضمنياً.

اختبار المرحلة الرابعة: تطابقت الأوزان والملفات الـ15 الموجودة في الكاش، وملفات Python الـ157 مع أرشيف revision الرسمي. نجح تنزيل الشفرة إلى منزل اختبار فارغ وفحصها. لم نعد تنزيل 2.72 GiB من الأوزان من الصفر؛ لم يُختبر انقطاع الشبكة أو كل مصادر التنزيل بأوزان مفقودة. أعادت البيئتان المستقلتان التنبؤات الفعلية باستخدام هذا الكاش المعلن، لا مكتبات جهاز مخفية.

</div>
