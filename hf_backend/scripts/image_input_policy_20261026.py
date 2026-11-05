"""Lossless source handling for the local tester; source files stay immutable."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from PIL import Image,ImageOps

def decode_rgb(source):
    im=ImageOps.exif_transpose(source)
    if 'A' in im.getbands() or 'transparency' in im.info:
        rgba=im.convert('RGBA')
        im=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba)
    return im.convert('RGB')

@contextmanager
def normalized_input_path(path):
    path=Path(path)
    with Image.open(path) as im:
        transparent='A' in im.getbands() or 'transparency' in im.info
        rgb=decode_rgb(im) if transparent else None
    if not transparent:
        yield path
        return
    with TemporaryDirectory(prefix='saer_rgb_') as temp:
        target=Path(temp)/'input.png'
        rgb.save(target,format='PNG')
        yield target
