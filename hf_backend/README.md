---
title: Saer Inference API
sdk: gradio
python_version: 3.12.12
app_file: app.py
pinned: false
---

# خدمة النماذج الحية لـ «سعّر»

هذه ملفات مصدر Space بنظام Gradio وZeroGPU. لا تتضمن الأوزان الخاصة أو بيانات المشروع؛ تُحمّل الأوزان المعتمدة من مستودع Hugging Face خاص عند بدء المساحة. تُجلب أوزان DINOv2 وCLIP وResNet وSSDLite العامة من مصادرها عند التشغيل، ولا تدخل ضمن حزمة الأصول الخاصة. لا تستخدم هذه الحزمة لتدريب النماذج.

## تشغيل مجاني عبر Kaggle

إذا لم يكن Gradio Space متاحاً على حساب Hugging Face دون خطة مدفوعة، استخدم الحزمة الجاهزة `C:\car_handoff\kaggle_deployment\saer-backend-bundle.zip` والدفتر `kaggle_server.ipynb`. اجعل Kaggle Dataset خاصاً، وارفع ZIP، ثم فعّل Internet وGPU وشغّل الدفتر. يقرأ `LOCAL_ASSETS_DIR` الملفات الخاصة المحلية أولاً، لذلك لا يحتاج هذا المسار إلى `HF_TOKEN` أو مستودع Hugging Face خاص.

يدعم `app.py` تشغيل رابط Gradio Share عبر `launch_gradio_share()` أو `SAER_GRADIO_SHARE=1 python app.py`. يدعم الدفتر أيضاً خيار Cloudflare Tunnel كبديل. الروابط عامة ومؤقتة، وتتوقف عند انتهاء جلسة Kaggle.

## الإعداد المطلوب في Space

1. أنشئ مستودع Model خاصاً وارفع إليه محتويات `private_assets_to_upload` كما هي مع الحفاظ على بنية المجلدات.
2. في Space > Settings، أضف Secret باسم `HF_TOKEN` بقيمة رمز قراءة فقط، وVariable باسم `HF_MODEL_REPO` وقيمته `namespace/private-model-repo`.
3. أضف Variable باسم `FRONTEND_ORIGINS` يحتوي أصل Cloudflare Pages الفعلي، مثل `https://project.pages.dev`، ويمكن إضافة أصل المعاينة عند الحاجة. تُضاف عناوين localhost للتجربة تلقائياً. لا تستخدم `*`.
4. اختر Gradio SDK وZeroGPU من إعدادات المساحة. لا تضع أي Secret في ملفات المشروع أو سجل Git.

يستخدم `HF_TOKEN` لتنزيل ملفات محددة من المستودع الخاص عبر `snapshot_download` فقط. لا يرفع التطبيق ملفات أو مدخلات إلى ذلك المستودع. شيفرة Space عامة؛ لذلك لا تضع فيها أوزاناً خاصة. مجلد `C:\car_handoff\private_assets_to_upload` يحتوي ستة ملفات خاصة فقط؛ قائمة الأحجام والبصمات محفوظة منفصلة في `C:\car_handoff\private_assets_manifest.json`.

## نقاط API

- `/identify_vehicle`: صورة واحدة؛ يعيد اقتراح الماركة والموديل والدرجات غير المعايرة.
- `/predict_image`: من صورة واحدة إلى 8 صور؛ يعيد تقدير الصورة ونطاقيه المحفوظين في النموذج.
- `/predict_specs`: كائن JSON بمواصفات سيارة مستعملة؛ يعيد سعر الإعلان التقديري.

تقبل الصور JPG وPNG وWebP حتى 15 ميجابايت و40 مليون بكسل. عند توفر CUDA المحاكى من ZeroGPU، تُحمّل المشفّرات العامة على الجهاز وقت بدء العملية لتُعاد استخدامها في الاستدعاءات؛ التشغيل المحلي بلا GPU يستخدم CPU. قد يطول بدء Space أول مرة بسبب تنزيل الأوزان العامة والخاصة.

## تشغيل محلي للتطوير

استخدم Python 3.12.12. يثبّت Space حزم Gradio وZeroGPU SDK التي تديرها المنصة؛ ثبّت `requirements.txt` للتبعيات المثبتة، ثم `gradio` و`spaces` للتشغيل المحلي فقط. اضبط `HF_MODEL_REPO` و`HF_TOKEN` في متغيرات البيئة، واضبط `SAER_APP_ROOT` إلى مجلد `hf_backend` عند تشغيل `python app.py`. يفضّل أن يكون `HF_TOKEN` رمز قراءة للمستودع الخاص فقط.

في الواجهة الأمامية، افتح الموقع مع `?api=https://<space-name>.hf.space`. تحفظ الواجهة الرابط في `sessionStorage` لهذه الجلسة؛ لا يلزم إعادة بناء ملفات Cloudflare عند تبديل رابط Space أو Tunnel.

لخادم Kaggle: يدعم بناء الواجهة أيضاً `?apiConfig=<رابط JSON عام>`، ويمكن أن يكون الرابط واجهة Gist API عامة فيها ملف JSON يحتوي `api_base_url`. يحدث الدفتر هذا الملف عند إعادة تشغيل النفق؛ لذلك يبقى رابط Pages ثابتاً. يرسل عميل الواجهة ترويسة `ngrok-skip-browser-warning` إلى نطاقات ngrok المعروفة. النطاق المجاني الذي يعيّنه ngrok للحساب ثابت الاسم (حالياً `ngrok-free.dev`) وليس اسماً مخصصاً يختاره المستخدم. ثبات الرابط لا يبقي خادم Kaggle عاملاً عند توقف الجلسة.

## حدود التشغيل

السعر من الصورة تجريبي، ولا يحدد الممشى أو الحالة الميكانيكية. الهوية ضمن الفئات التي دُرّبت عليها والدرجات غير معايرة. أسعار المواصفات أسعار إعلان تقديرية وليست سعراً نهائياً. بيانات السوق والإعلانات والتوصيات تبقى نسخة عرض ثابتة في الواجهة.
