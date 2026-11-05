export function validateImage(file?: File): asserts file is File {
  if (!file) throw new Error('اختر صورة السيارة أولًا.');
  if (!['image/jpeg','image/png','image/webp'].includes(file.type))
    throw new Error('اختر صورة بصيغة JPG أو PNG أو WebP.');
  if (!file.size) throw new Error('ملف الصورة فارغ. اختر صورة صالحة.');
  if (file.size > 12 * 1024 * 1024) throw new Error('حجم الصورة أكبر من 12 ميجابايت. اختر صورة أصغر.');
}

export async function prepareImage(file?: File): Promise<string> {
  validateImage(file);
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (!image.naturalWidth || !image.naturalHeight) throw new Error();
    return url;
  } catch {
    URL.revokeObjectURL(url);
    throw new Error('تعذّر فتح الصورة. جرّب صورة أخرى.');
  }
}
