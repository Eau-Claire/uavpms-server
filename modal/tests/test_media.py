"""Input validation for decode_image. Needs fastapi and Pillow (modal/.venv)."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from fastapi import HTTPException
    from PIL import Image
    from powerline_ai.media import decode_image
    from powerline_ai.settings import MAX_PIXELS
except ImportError as exc:  # pragma: no cover - environment without fastapi
    raise unittest.SkipTest(f'fastapi/Pillow not installed: {exc}')


def encode(image, fmt, **kw):
    buf = io.BytesIO()
    image.save(buf, fmt, **kw)
    return buf.getvalue()


class DecodeTests(unittest.TestCase):
    def rejects(self, data, status):
        with self.assertRaises(HTTPException) as ctx:
            decode_image(data)
        self.assertEqual(ctx.exception.status_code, status)

    def test_jpeg_png_mpo_accepted(self):
        im = Image.new('RGB', (64, 48), (10, 20, 30))
        for fmt in ('JPEG', 'PNG', 'MPO'):  # MPO = multi-picture JPEG written by DJI drones
            out = decode_image(encode(im, fmt))
            self.assertEqual((out.mode, out.size), ('RGB', (64, 48)), fmt)

    def test_exif_orientation_applied(self):
        im = Image.new('RGB', (64, 48))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
        self.assertEqual(decode_image(encode(im, 'JPEG', exif=exif)).size, (48, 64))

    def test_rejections(self):
        im = Image.new('RGB', (64, 48))
        self.rejects(encode(im, 'GIF'), 415)
        self.rejects(b'not an image', 415)
        self.rejects(b'', 415)
        self.rejects(encode(im, 'JPEG')[:200], 415)  # truncated
        side = int(MAX_PIXELS ** 0.5) + 10
        self.rejects(encode(Image.new('L', (side, side)), 'PNG'), 413)


if __name__ == '__main__':
    unittest.main()
