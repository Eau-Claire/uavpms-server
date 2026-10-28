import io
import warnings
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError
from .settings import MAX_BYTES, MAX_PIXELS

async def read_image(request):
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, 'Image exceeds 10 MiB')
        data.extend(chunk)
    return decode_image(data)

def decode_image(data):
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image exceeds 10 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ('JPEG', 'PNG'):
                    raise HTTPException(415, 'Use JPEG or PNG')
                if im.width * im.height > MAX_PIXELS:
                    raise HTTPException(413, 'Image exceeds 20 MP')
                return im.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(415, 'Invalid or oversized image')
