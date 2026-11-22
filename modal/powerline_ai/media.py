import io
import warnings
from fastapi import HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from .settings import IMAGE_FORMATS, MAX_BYTES, MAX_PIXELS

TOO_BIG = f'Image exceeds {MAX_BYTES // 2**20} MiB'
TOO_MANY_PIXELS = f'Image exceeds {MAX_PIXELS // 1_000_000} MP'
WRONG_FORMAT = 'Use ' + ', '.join(IMAGE_FORMATS)


async def read_body(request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, TOO_BIG)
        data.extend(chunk)
    return bytes(data)


async def read_image(request):
    return decode_image(await read_body(request))


def decode_image(data):
    """Validate and decode to RGB, applying the EXIF orientation (as OpenCV does in training and
    as browsers do when displaying), so boxes match the image the user sees."""
    if len(data) > MAX_BYTES:
        raise HTTPException(413, TOO_BIG)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in IMAGE_FORMATS:
                    raise HTTPException(415, WRONG_FORMAT)
                if im.width * im.height > MAX_PIXELS:
                    raise HTTPException(413, TOO_MANY_PIXELS)
                return ImageOps.exif_transpose(im).convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(415, 'Invalid or oversized image')
